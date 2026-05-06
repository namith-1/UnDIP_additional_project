
def _l2_distance(U, v):
    return np.sqrt(np.sum((U - v)**2, axis=0))

def pymf_sivm(X, p):
    L, N = X.shape
    EPS = 1e-16
    cur_p = 0
    d = np.zeros(N)
    for i in range(3):
        d = _l2_distance(X, X[:, cur_p:cur_p+1])
        cur_p = int(np.argmax(d))
    maxd = np.max(d)
    select = [cur_p]
    d_square = np.zeros(N)
    d_sum = np.zeros(N)
    d_i_times_d_j = np.zeros(N)
    a = np.log(maxd + EPS)
    for l in range(1, p):
        d = _l2_distance(X, X[:, select[l-1]:select[l-1]+1])
        d = np.log(d + EPS)
        d_i_times_d_j += d * d_sum
        d_sum += d
        d_square += d**2
        distiter = d_i_times_d_j + a * d_sum - (l / 2.0) * d_square
        select.append(int(np.argmax(distiter)))
    return np.array(select, dtype=int)
"""
UnDIP: Hyperspectral Unmixing Using Deep Image Prior
======================================================
Main training & evaluation pipeline.

Reference:
    B. Rasti, B. Koirala, P. Scheunders, P. Ghamisi.
    "UnDIP: Hyperspectral Unmixing Using Deep Image Prior."
    IEEE Transactions on Geoscience and Remote Sensing, 2022.

Pipeline:
    1. Load dataset (Jasper / Samson).
    2. SiVM endmember extraction (with SVD reduction).
    3. Train UnmixArch CNN (Deep Image Prior) to estimate abundances.
    4. Evaluate: SAD (Spectral Angle Distance) + RMSE.
    5. Visualise abundance maps and endmember spectra.

Usage:
    python undip_train.py --dataset jasper --num_iter 3000
    python undip_train.py --dataset samson --num_iter 3000
"""

import os
import argparse
import time

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')          # non-interactive; change to 'TkAgg' if you want pop-ups
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from sivm import reorder_endmembers
from undip_model import UnDIPNet


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark     = True


# ===========================================================================
# Data Loading
# ===========================================================================

def load_jasper(data_dir: str):
    """
    Load Jasper Ridge dataset.

    Returns:
        Y        : (L, N) float32 — normalised HSI cube (198 bands, 10000 px)
        A_gt     : (p, H, W) float32 — GT abundance maps (4, 100, 100)
        E_gt     : (L, p) float32 — GT endmembers (198, 4)
        H, W, p  : spatial dims and number of endmembers
    """
    # Raw HSI
    mat = sio.loadmat(os.path.join(data_dir, 'jasper', 'jasperRidge2_R198.mat'))
    Y   = mat['Y'].astype(np.float32)          # (198, 10000)
    nR  = int(mat['nRow'].flat[0])
    nC  = int(mat['nCol'].flat[0])

    # Normalise to [0, 1]
    Y_max = Y.max()
    if Y_max > 0:
        Y = Y / Y_max

    # Ground truth
    gt  = sio.loadmat(os.path.join(data_dir, 'jasper', 'end4.mat'))
    E_gt = gt['M'].astype(np.float32)           # (198, 4)
    A_gt = gt['A'].astype(np.float32)           # (4, 10000)
    p    = E_gt.shape[1]

    E_gt_max = E_gt.max()
    if E_gt_max > 0:
        E_gt = E_gt / E_gt_max

    A_gt_3d = A_gt.reshape(p, nR, nC)

    return Y, A_gt_3d, E_gt, nR, nC, p


def load_samson(data_dir: str):
    """
    Load Samson dataset.

    Returns:
        Y        : (L, N) float32 — normalised HSI (156 bands, 9025 px)
        A_gt     : (p, H, W) float32 — GT abundance maps (3, 95, 95)
        E_gt     : (L, p) float32 — GT endmembers (156, 3)
        H, W, p  : spatial dims and number of endmembers
    """
    mat = sio.loadmat(os.path.join(data_dir, 'samson', 'samson_1.mat'))
    Y   = mat['V'].astype(np.float32)           # (156, 9025)
    nR  = int(mat['nRow'].flat[0])
    nC  = int(mat['nCol'].flat[0])

    Y_max = Y.max()
    if Y_max > 0:
        Y = Y / Y_max

    gt  = sio.loadmat(os.path.join(data_dir, 'samson', 'end3.mat'))
    E_gt = gt['M'].astype(np.float32)           # (156, 3)
    A_gt = gt['A'].astype(np.float32)           # (3, 9025)
    p    = E_gt.shape[1]

    E_gt_max = E_gt.max()
    if E_gt_max > 0:
        E_gt = E_gt / E_gt_max

    A_gt_3d = A_gt.reshape(p, nR, nC)

    return Y, A_gt_3d, E_gt, nR, nC, p


DATASET_LOADERS = {
    'jasper': load_jasper,
    'samson': load_samson,
}


# ===========================================================================
# Evaluation Metrics
# ===========================================================================

def rmse(A_true, A_est):
    """Root Mean Squared Error × 100 (percentage)."""
    A_true = A_true.astype(np.float32)
    A_est  = np.clip(A_est,  0, 1).astype(np.float32)
    return 100.0 * float(np.sqrt(np.mean((A_true - A_est) ** 2)))


def mae(A_true, A_est):
    """Mean Absolute Error × 100 (percentage)."""
    A_true = A_true.astype(np.float32)
    A_est  = np.clip(A_est,  0, 1).astype(np.float32)
    return 100.0 * float(np.mean(np.abs(A_true - A_est)))


def sad(E_ref, E_est):
    """
    Spectral Angle Distance (SADs) for each endmember pair, in degrees.

    Args:
        E_ref : (L, p)
        E_est : (L, p)
    Returns:
        mean_sad : scalar (mean over p endmembers)
        sads     : (p,) per-endmember SADs
    """
    p = E_ref.shape[1]
    sads = np.zeros(p)
    for i in range(p):
        u = E_ref[:, i]
        v = E_est[:, i]
        cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        sads[i] = np.degrees(np.arccos(cos_theta))
    return float(np.mean(sads)), sads


# ===========================================================================
# Noise initialisation (Deep Image Prior style)
# ===========================================================================

def get_noise(channels, H, W, method='noise', seed=None):
    """
    Generate a fixed noise input tensor z of shape (1, channels, H, W).

    Args:
        channels : int — p (number of endmembers).
        H, W     : spatial dimensions.
        method   : 'noise' (uniform) or 'meshgrid'.
        seed     : optional int for reproducibility.
    Returns:
        z : torch.FloatTensor (1, channels, H, W).
    """
    if seed is not None:
        torch.manual_seed(seed)

    if method == 'noise':
        z = torch.zeros(1, channels, H, W).uniform_() * 0.1
    elif method == 'meshgrid':
        assert channels == 2
        X, Y = np.meshgrid(np.arange(W) / float(W - 1),
                            np.arange(H) / float(H - 1))
        z = torch.from_numpy(
            np.concatenate([X[None, None], Y[None, None]], axis=1)
        ).float()
    else:
        raise ValueError(f'Unknown noise method: {method}')

    return z.detach()


# ===========================================================================
# Main UnDIP training loop
# ===========================================================================

def run_undip(
    Y           : np.ndarray,   # (L, N)
    E_init      : np.ndarray,   # (L, p)  — endmembers from SiVM
    nR          : int,
    nC          : int,
    p           : int,
    num_iter    : int   = 3000,
    lr          : float = 1e-3,
    exp_weight  : float = 0.99,
    device      : str   = 'cpu',
    show_every  : int   = 500,
    seed        : int   = 42,
) -> dict:
    """
    Run the UnDIP optimisation loop.

    Mathematical model (linear mixing model):
        Y ≈ E @ A       where A are fractional abundances (ASC enforced)

    Loss:
        L = ||Y - E @ f_θ(z)||²_F

    where f_θ(z) is the CNN output (abundance maps) given fixed noise z.

    Args:
        Y          : (L, N) observed HSI.
        E_init     : (L, p) endmembers extracted by SiVM.
        nR, nC     : spatial dimensions (N = nR * nC).
        p          : number of endmembers.
        num_iter   : number of gradient-descent iterations.
        lr         : Adam learning rate (paper: 0.001).
        exp_weight : exponential smoothing weight (paper: 0.99).
        device     : 'cuda' or 'cpu'.
        show_every : print interval.
        seed       : random seed for noise initialisation.

    Returns:
        dict with keys:
            'A_est'     : (p, nR, nC) float32 — final abundance maps
            'A_est_avg' : (p, nR, nC) float32 — smoothed abundance maps
            'losses'    : list of scalar losses
            'E_init'    : (L, p) endmembers used
    """
    L, N = Y.shape
    assert N == nR * nC, f"N mismatch: {N} vs {nR*nC}"

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {dev}")

    # ------------------------------------------------------------------ #
    # Prepare tensors
    # ------------------------------------------------------------------ #
    Y_3d   = Y.reshape(L, nR, nC)                         # (L, H, W)
    Y_t    = torch.from_numpy(Y_3d[None]).to(dev)         # (1, L, H, W)
    E_t    = torch.from_numpy(E_init).to(dev)             # (L, p)

    # Fixed noise input
    z      = get_noise(p, nR, nC, method='noise', seed=seed).to(dev)

    # ------------------------------------------------------------------ #
    # Build model
    # ------------------------------------------------------------------ #
    net = UnDIPNet(
        p=p,
        num_channels_down=(256,),
        num_channels_up=(256,),
        num_channels_skip=(4,),
        filter_size_down=3,
        filter_size_up=3,
        filter_skip_size=1,
        pad='reflection',
        upsample_mode='bilinear',
        act_fun='LeakyReLU',
    ).to(dev)

    n_params = sum(param.numel() for param in net.parameters())
    print(f"UnDIPNet parameters: {n_params:,}")
    print(net)

    # ------------------------------------------------------------------ #
    # Optimiser
    # ------------------------------------------------------------------ #
    optimiser = optim.Adam(net.parameters(), lr=lr)
    mse_loss  = nn.MSELoss()

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    losses          = []
    out_avg         = None
    out_HR_avg      = None

    t0 = time.time()
    net.train()

    for it in range(1, num_iter + 1):
        optimiser.zero_grad()

        # Forward: abundance maps  (1, p, H, W)
        A_hat = net(z)

        # Reconstruct HSI: E @ A_hat → (1, L, H, W)
        # E_t: (L, p), A_hat: (1, p, H, W)
        A_flat   = A_hat.view(p, N)                        # (p, N)
        Y_hat    = torch.mm(E_t, A_flat)                   # (L, N)
        Y_hat_3d = Y_hat.view(1, L, nR, nC)               # (1, L, H, W)

        # MSE loss in the observation space
        loss = mse_loss(Y_t, Y_hat_3d)
        loss.backward()
        optimiser.step()

        # Exponential smoothing (EMA) of outputs
        with torch.no_grad():
            if out_avg is None:
                out_avg    = A_hat.detach().clone()
                out_HR_avg = Y_hat_3d.detach().clone()
            else:
                out_avg    = exp_weight * out_avg    + (1 - exp_weight) * A_hat.detach()
                out_HR_avg = exp_weight * out_HR_avg + (1 - exp_weight) * Y_hat_3d.detach()

        losses.append(float(loss.item()))

        if it % show_every == 0 or it == 1:
            elapsed = time.time() - t0
            print(f"Iter [{it:5d}/{num_iter}]  Loss: {loss.item():.6f}  "
                  f"Elapsed: {elapsed:.1f}s")

    # ------------------------------------------------------------------ #
    # Extract final abundance maps
    # ------------------------------------------------------------------ #
    net.eval()
    with torch.no_grad():
        A_final   = net(z).squeeze(0).cpu().numpy()   # (p, H, W)
    A_avg = out_avg.squeeze(0).cpu().numpy()           # (p, H, W)

    A_final   = np.clip(A_final, 0, 1)
    A_avg     = np.clip(A_avg,   0, 1)

    return {
        'A_est'     : A_final,
        'A_est_avg' : A_avg,
        'losses'    : losses,
        'E_init'    : E_init,
    }


# ===========================================================================
# Visualisation helpers
# ===========================================================================

def plot_abundance_maps(A_est, A_gt, endmember_names, save_path=None):
    """
    Side-by-side abundance map comparison: Estimated (left) vs GT (right).

    Args:
        A_est          : (p, H, W)
        A_gt           : (p, H, W)
        endmember_names: list of p strings
        save_path      : if given, saves the figure to this path
    """
    p = A_est.shape[0]
    fig, axes = plt.subplots(p, 2, figsize=(8, 3 * p))
    if p == 1:
        axes = axes[np.newaxis]

    for i in range(p):
        im1 = axes[i, 0].imshow(A_est[i], cmap='jet', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Estimated: {endmember_names[i]}', fontsize=10)
        axes[i, 0].axis('off')
        plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)

        im2 = axes[i, 1].imshow(A_gt[i], cmap='jet', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Ground Truth: {endmember_names[i]}', fontsize=10)
        axes[i, 1].axis('off')
        plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)

    plt.suptitle('UnDIP Abundance Maps', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved abundance map figure → {save_path}")
    plt.close()


def plot_endmember_spectra(E_gt, E_est, endmember_names, save_path=None):
    """Plot extracted endmember spectra vs ground truth."""
    L, p = E_gt.shape
    wavelengths = np.arange(L)

    fig, axes = plt.subplots(1, p, figsize=(5 * p, 4), squeeze=False)
    colors = plt.cm.Set1(np.linspace(0, 1, p))

    for i in range(p):
        axes[0, i].plot(wavelengths, E_gt[:, i], '--', color=colors[i],
                        label='Ground Truth', linewidth=1.5)
        axes[0, i].plot(wavelengths, E_est[:, i], '-', color=colors[i],
                        label='SiVM Extracted', linewidth=1.5, alpha=0.8)
        axes[0, i].set_title(endmember_names[i])
        axes[0, i].set_xlabel('Band index')
        axes[0, i].set_ylabel('Reflectance')
        axes[0, i].legend(fontsize=7)
        axes[0, i].grid(True, alpha=0.3)

    plt.suptitle('Endmember Spectra: SiVM vs Ground Truth', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved endmember spectra figure → {save_path}")
    plt.close()


def plot_loss_curve(losses, save_path=None):
    """Plot training loss curve."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(losses, linewidth=1.5, color='steelblue')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE Loss (log scale)')
    ax.set_title('UnDIP Training Loss Curve')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss curve figure → {save_path}")
    plt.close()


# ===========================================================================
# Dataset-specific endmember name maps
# ===========================================================================

ENDMEMBER_NAMES = {
    'jasper': ['Tree', 'Water', 'Dirt/Soil', 'Road'],
    'samson': ['Soil', 'Tree', 'Water'],
}


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='UnDIP: Hyperspectral Unmixing Using Deep Image Prior'
    )
    parser.add_argument('--dataset',   type=str, default='jasper',
                        choices=['jasper', 'samson'],
                        help='Dataset to use (default: jasper)')
    parser.add_argument('--data_dir',  type=str,
                        default=r'c:\Users\likhi\OneDrive\Documents\prev_files\UDIP\processed_data',
                        help='Root directory containing dataset sub-folders')
    parser.add_argument('--out_dir',   type=str,
                        default=r'c:\Users\likhi\OneDrive\Documents\prev_files\UDIP\results',
                        help='Output directory for figures and results')
    parser.add_argument('--num_iter',  type=int, default=3000,
                        help='Number of training iterations (default: 3000)')
    parser.add_argument('--lr',        type=float, default=1e-3,
                        help='Adam learning rate (default: 1e-3)')
    parser.add_argument('--exp_weight',type=float, default=0.99,
                        help='Exponential smoothing weight (default: 0.99)')
    parser.add_argument('--device',    type=str, default='cuda',
                        help='Compute device: cuda or cpu (default: cuda)')
    parser.add_argument('--seed',      type=int, default=42,
                        help='Random seed for noise input (default: 42)')
    parser.add_argument('--show_every',type=int, default=500,
                        help='Print loss every N iterations (default: 500)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dataset_out = os.path.join(args.out_dir, args.dataset)
    os.makedirs(dataset_out, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  UnDIP — Dataset: {args.dataset.upper()}")
    print(f"{'='*60}\n")

    loader = DATASET_LOADERS[args.dataset]
    _, A_gt, E_gt, nR, nC, p = loader(args.data_dir)
    
    # Generate Y_clean and add 30dB Gaussian Noise (as in paper)
    A_gt_flat = A_gt.reshape(p, nR * nC)
    Y_clean = E_gt @ A_gt_flat
    
    # Calculate noise variance for 30dB SNR
    np.random.seed(args.seed)
    # SNR = 10 * log10( var(signal) / var(noise) ) -> var(noise) = var(signal) / 10^(SNR/10)
    snr_db = 30.0
    signal_power = np.mean(Y_clean ** 2)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), Y_clean.shape)
    Y = Y_clean + noise
    
    Y = np.clip(Y, 0, 1).astype(np.float32)

    L, N = Y.shape
    em_names = ENDMEMBER_NAMES.get(args.dataset, [f'EM_{i}' for i in range(p)])

    print(f"HSI shape    : ({L}, {N})  ->  ({L} bands, {nR}x{nC} pixels)")
    print(f"Endmembers   : {p}")
    print(f"GT abundance : {A_gt.shape}")
    print(f"GT endmembers: {E_gt.shape}")

    # ------------------------------------------------------------------
    # 2. SiVM endmember extraction
    # ------------------------------------------------------------------
    print(f"\n[Step 1] Running SiVM (Simple Volume Maximization) ...")
    t_sivm = time.time()
    indices = pymf_sivm(Y, p)
    E_sivm = Y[:, indices]
    print(f"  SiVM done in {time.time()-t_sivm:.2f}s")
    print(f"  Selected pixel indices: {indices}")

    # Reorder extracted endmembers to match ground truth for evaluation
    order = reorder_endmembers(E_gt, E_sivm)
    E_sivm_ordered = E_sivm[:, order]

    # SAD between SiVM endmembers and GT
    mean_sad_val, sads = sad(E_gt, E_sivm_ordered)
    print(f"  SADs (per endmember): {[f'{s:.2f} deg' for s in sads]}")
    print(f"  Mean SAD: {mean_sad_val:.4f} deg")

    # Save endmember spectra figure
    plot_endmember_spectra(
        E_gt, E_sivm_ordered, em_names,
        save_path=os.path.join(dataset_out, 'endmember_spectra_sivm.png')
    )

    # ------------------------------------------------------------------
    # 3. UnDIP CNN training
    # ------------------------------------------------------------------
    print(f"\n[Step 2] Training UnDIP CNN for {args.num_iter} iterations ...")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results = run_undip(
        Y          = Y,
        E_init     = E_sivm_ordered,
        nR         = nR,
        nC         = nC,
        p          = p,
        num_iter   = args.num_iter,
        lr         = args.lr,
        exp_weight = args.exp_weight,
        device     = args.device,
        show_every = args.show_every,
        seed       = args.seed,
    )

    A_est     = results['A_est']      # (p, H, W)
    A_est_avg = results['A_est_avg']  # (p, H, W)
    losses    = results['losses']

    # ------------------------------------------------------------------
    # 4. Evaluation (with best-permutation abundance matching)
    # ------------------------------------------------------------------
    print(f"\n[Step 3] Evaluation ...")

    def best_perm_match(A_gt_3d, A_est_3d):
        """Find permutation of A_est channels minimising RMSE vs A_gt."""
        from itertools import permutations
        p_loc = A_gt_3d.shape[0]
        best_r = np.inf
        best_A = A_est_3d
        best_perm_idx = list(range(p_loc))
        A_gt_f  = A_gt_3d.reshape(p_loc, -1)
        A_est_f = A_est_3d.reshape(p_loc, -1)
        for perm in permutations(range(p_loc)):
            A_p = A_est_f[list(perm), :]
            r = 100.0 * float(np.sqrt(np.mean((A_gt_f - A_p) ** 2)))
            if r < best_r:
                best_r = r
                best_A = A_est_3d[list(perm), :, :]
                best_perm_idx = list(perm)
        return best_A, best_r, best_perm_idx

    A_est_matched, rmse_est, perm_est = best_perm_match(A_gt, A_est)
    A_avg_matched, rmse_avg, _        = best_perm_match(A_gt, A_est_avg)
    mae_est = mae(A_gt, A_est_matched)
    mae_avg = mae(A_gt, A_avg_matched)

    print(f"  Best channel permutation (final): {perm_est}")
    print(f"  RMSE (final, best perm): {rmse_est:.4f}% | MAE: {mae_est:.4f}%")
    print(f"  RMSE (avg,   best perm): {rmse_avg:.4f}% | MAE: {mae_avg:.4f}%")

    matched_em_names = [em_names[i] for i in perm_est]
    for i, name in enumerate(matched_em_names):
        r = rmse(A_gt[i:i+1], A_est_matched[i:i+1])
        m = mae(A_gt[i:i+1], A_est_matched[i:i+1])
        print(f"    {name:12s}: RMSE = {r:.4f}%, MAE = {m:.4f}%")

    # ------------------------------------------------------------------
    # 5. Save figures & results
    # ------------------------------------------------------------------
    print(f"\n[Step 4] Saving figures and results ...")

    plot_abundance_maps(
        A_est_matched, A_gt, matched_em_names,
        save_path=os.path.join(dataset_out, 'abundance_maps_final.png')
    )
    plot_abundance_maps(
        A_avg_matched, A_gt, matched_em_names,
        save_path=os.path.join(dataset_out, 'abundance_maps_avg.png')
    )
    plot_loss_curve(
        losses,
        save_path=os.path.join(dataset_out, 'loss_curve.png')
    )

    # Save numerical results
    np.save(os.path.join(dataset_out, 'A_est.npy'), A_est_matched)
    np.save(os.path.join(dataset_out, 'A_est_avg.npy'), A_avg_matched)
    np.save(os.path.join(dataset_out, 'E_sivm.npy'), E_sivm_ordered)
    np.save(os.path.join(dataset_out, 'losses.npy'), np.array(losses))

    print(f"\n{'='*60}")
    print(f"  Results saved to: {dataset_out}")
    print(f"{'='*60}")
    print(f"\n  Final Summary:")
    print(f"    Dataset       : {args.dataset}")
    print(f"    Mean SAD (SiVM endmembers): {mean_sad_val:.4f} deg")
    print(f"    RMSE (UnDIP final, best perm) : {rmse_est:.4f}%")
    print(f"    RMSE (UnDIP avg, best perm)   : {rmse_avg:.4f}%")
    print()


if __name__ == '__main__':
    main()
