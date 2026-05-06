# UnDIP: Hyperspectral Unmixing Using Deep Image Prior

This repository contains an end-to-end Python/PyTorch implementation of the **UnDIP (Hyperspectral Unmixing using Deep Image Prior)** framework, based on the 2022 paper by Behnood Rasti et al.

> **AI Context Note:** This document serves as the primary metadata reference for any AI agent or developer working in this repository. It documents the exact directory structures, the mathematical quirks of the algorithms used, and the empirical baselines achieved.

## Directory Structure

*   `undip_model.py`: Contains the `UnmixArch` CNN (Encoder-Decoder with skip connections).
*   `sivm.py`: Contains the Simplex Volume Maximization (SiVM) algorithm used for extracting pure material signatures (Endmembers) prior to CNN training.
*   `undip_synthetic.py` / `undip_train.py`: The main execution pipelines for optimizing the Deep Image Prior network.
*   `processed_data/` (Ignored in Git): Contains the raw hyperspectral data cubes.
*   `results/` (Ignored in Git): Contains the output abundance maps, endmember spectra charts, and loss curves.

---

## 1. Data Metadata (`processed_data/`)

The repository uses standard hyperspectral benchmarks. Because of GitHub size limits, the data is stored locally and git-ignored.

### Samson Dataset (`processed_data/samson/`)
*   **Raw Data (`samson_1.mat`):** 156 bands, 95 x 95 pixels.
*   **Endmembers ($p=3$):** Soil, Tree, Water (`end3.mat`).
*   **Paper Evaluation Quirk:** The original UnDIP paper does *not* evaluate on the raw noisy `samson_1.mat`. Instead, it synthesizes a clean image using `Y_clean = E_GT x A_GT`, adds 30dB of Gaussian noise, and evaluates on that. This is crucial for matching the paper's single-digit MAE scores.

### Jasper Ridge Dataset (`processed_data/jasper/`)
*   **Raw Data:** 198 bands, 100 x 100 pixels.
*   **Endmembers ($p=4$):** Tree, Water, Road, Dirt/Soil (`end4.mat`).
*   **Evaluation:** Similar to Samson, evaluated using synthetic 30dB noisy data generated from ground truths.

### Urban Dataset (`processed_data/urban/`)
*   **Raw Data:** Large-scale hyperspectral image (143 MB `.npy` file, explicitly blocked from Git tracking).
*   **Endmembers ($p=4$):** Asphalt, Grass, Tree, Roof.

---

## 2. Algorithm & Methodology Metadata

### SiVM (Endmember Extraction)
The `sivm.py` implementation is mathematically identical to the Python Matrix Factorization (`pymf`) library used in the original paper.
*   **Initialization:** Uses a "Fastmap" sequence (3 iterations of picking the furthest Euclidean point) to stabilize the first selected node.
*   **Volume Proxy:** Instead of calculating exact geometric volumes (which is numerically unstable in high dimensions), it maximizes a robust proxy using the sum of the log-distances.
*   **Performance:** Consistently extracts endmembers with a Mean Spectral Angle Distance (SAD) of **~4° - 6°** on synthesized data.

### UnDIP CNN (Abundance Estimation)
*   **Deep Image Prior:** The architecture of the network replaces the traditional explicit Bayesian prior (like Total Variation).
*   **Constraints:**
    *   **ANC (Non-negativity):** Enforced via the exponential numerator in the final Softmax layer.
    *   **ASC (Sum-to-One):** Enforced by dividing by the sum in the final Softmax layer, guaranteeing abundances equal 1.0.

---

## 3. Results Metadata (`results/`)

When the pipeline is executed (`python undip_synthetic.py --dataset samson --num_iter 3000 --device cuda`), it dumps all artifacts here. Future AI agents can use these baselines to verify code changes:

### Baseline Benchmarks (Synthetic 30dB Noise)
*   **Samson:**
    *   Mean SAD: ~5.37°
    *   RMSE (Average): ~12.4%
    *   **MAE (Average): ~8.0%** (Matches paper's ~8% baseline)
*   **Jasper Ridge:**
    *   Mean SAD: ~6.82°
    *   RMSE (Average): ~14.9%
    *   **MAE (Average): ~7.3%** (Matches paper's 6% - 9% baseline)

### Generated Output Files
*   `A_est.npy`: The final estimated abundance map tensor.
*   `E_sivm.npy`: The extracted endmember signatures.
*   `abundance_maps_final.png`: Visual grids comparing ground truth abundances against UnDIP estimates.
*   `endmember_spectra_sivm.png`: Spectral signature comparisons ($E_{est}$ vs $E_{GT}$).
*   `loss_curve.png`: MSE reconstruction loss over 3000 iterations.
