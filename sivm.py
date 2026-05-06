"""
Simple Volume Maximization (SiVM) for Endmember Extraction.

Implements the algorithm described in:
    C. Thurau, K. Kersting, and C. Bauckhage.
    "Yes we can - Simplex Volume Maximization for Descriptive Web-Scale
    Matrix Factorization". CIKM 2010.

As used in the UnDIP paper:
    B. Rasti, B. Koirala, P. Scheunders, P. Ghamisi.
    "UnDIP: Hyperspectral Unmixing Using Deep Image Prior."
    IEEE Transactions on Geoscience and Remote Sensing, 2022.

Pipeline (matches UnDIP paper exactly):
    1. SVD dimensionality reduction to (p-1) components.
    2. Greedy simplex-volume maximization in reduced space.
    3. Return endmember indices and the (L × p) spectral matrix.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _euclidean_sq(a, b):
    """Squared Euclidean distance between two column vectors."""
    diff = (a - b).ravel()
    return float(diff @ diff)


def _endmember_extract(X, p):
    """
    Core SiVM greedy search as implemented in the UnDIP repository.

    Follows the original pymf.sivm logic ported to pure NumPy.

    Args:
        X : (L, N) float array — each column is a pixel spectrum.
        p : int — number of endmembers.

    Returns:
        I : (p,) int array  — selected pixel indices (sorted).
        d : (p, N) float array — squared distances used in the search.
    """
    L, N = X.shape
    d = np.zeros((p, N), dtype=np.float64)
    I = np.zeros(p, dtype=int)
    Z = np.zeros((L, 1), dtype=np.float64)

    # Step 1: index of pixel farthest from the origin
    for i in range(N):
        d[0, i] = _euclidean_sq(X[:, i:i+1], Z)
    I[0] = int(np.argmax(d[0]))

    # Step 2: distance from I[0]
    for i in range(N):
        d[0, i] = _euclidean_sq(X[:, i:i+1], X[:, I[0]:I[0]+1])

    # Steps 3..p: iterative Cayley-Menger volume maximization
    for v in range(1, p):
        D1 = np.concatenate(
            [d[:v, I[:v]].reshape(v, v), np.ones((v, 1))], axis=1
        )
        D2 = np.concatenate([np.ones((1, v)), np.zeros((1, 1))], axis=1)
        D4 = np.linalg.inv(np.concatenate([D1, D2], axis=0))  # (v+1, v+1)

        V = np.zeros((1, N), dtype=np.float64)
        for i in range(N):
            D3 = np.concatenate([d[:v, i:i+1], np.ones((1, 1))], axis=0)
            V[0, i] = float(np.squeeze(D3.T @ D4 @ D3))

        I[v] = int(np.argmax(V))

        # Update distances for newly selected point
        for i in range(N):
            d[v, i] = _euclidean_sq(X[:, i:i+1], X[:, I[v]:I[v]+1])

    # Sort indices for reproducibility
    sort_order = np.argsort(I)
    I = I[sort_order]
    d = d[sort_order, :]
    return I, d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sivm_extract(Y, num_endmembers):
    """
    Extract endmembers from hyperspectral data using SiVM with SVD reduction.

    This matches the UnDIP paper pipeline exactly:
        1. Clip negatives and normalise to [0, 1].
        2. SVD → keep `num_endmembers` principal components.
        3. Greedy simplex-volume search in reduced space.
        4. Return endmember spectra in *original* band space.

    Args:
        Y              : (L, N)  float array — L bands, N pixels.
        num_endmembers : int p  — number of endmembers to extract.

    Returns:
        indices : list of p column indices selected from Y.
        E       : (L, p) ndarray — endmember spectra (columns of Y).
    """
    L, N = Y.shape
    p = num_endmembers

    # ------------------------------------------------------------------ #
    # Pre-processing: clip and normalise (UnDIP clips to [0, 1])
    # ------------------------------------------------------------------ #
    Y_proc = np.clip(Y, 0, None)
    max_val = Y_proc.max()
    if max_val > 0:
        Y_proc = Y_proc / max_val

    # ------------------------------------------------------------------ #
    # SVD dimensionality reduction to p components (as in UnDIP)
    # ------------------------------------------------------------------ #
    U, s, Vt = np.linalg.svd(Y_proc, full_matrices=False)
    n_comp = min(p, L, N)
    # Project to reduced space: (n_comp, N)
    Y_reduced = np.diag(s[:n_comp]) @ Vt[:n_comp, :]

    # ------------------------------------------------------------------ #
    # SiVM greedy search in reduced space
    # ------------------------------------------------------------------ #
    indices, _ = _endmember_extract(Y_reduced, p)

    E = Y[:, indices.astype(int)]   # (L, p) in original band space
    return list(indices.astype(int)), E


def reorder_endmembers(E_ref, E_est):
    """
    Reorder columns of E_est to best match E_ref using l2 norm.
    Useful for evaluation against ground truth.

    Args:
        E_ref : (L, p) reference endmembers.
        E_est : (L, p) estimated endmembers.

    Returns:
        order : permutation array such that E_est[:, order] ≈ E_ref column-wise.
    """
    p = E_ref.shape[1]
    order = np.zeros(p, dtype=int)
    error = np.zeros((1, p))
    for l in range(p):
        for n in range(p):
            diff = E_ref[:, l] - E_est[:, n]
            error[0, n] = float(diff @ diff)
        order[l] = int(np.argmin(error))
    return order
