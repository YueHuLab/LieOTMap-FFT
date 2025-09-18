# Readme for LieOTMap-FFT (v4)

---

## 1. Algorithm Introduction

LieOTMap-FFT is a novel, fully differentiable framework for rigid body fitting of atomic models into cryo-EM density maps. The framework synergistically combines four key mathematical and computational techniques:

1.  **Lie Algebra:** The SE(3) rigid body transformation (rotation and translation) is parameterized using Lie algebra. This ensures a continuous and singularity-free optimization landscape, making it highly suitable for modern gradient-descent-based optimizers.

2.  **Optimal Transport (OT):** The mobile atomic structure and the target density map are treated as voxelized probability distributions on a 3D grid. Their similarity is measured using an Optimal Transport (OT) score, which is efficiently calculated by the Sinkhorn algorithm. This provides a robust similarity measure between the two maps.

3.  **Fast Fourier Transform (FFT):** A direct application of OT to high-resolution maps is computationally prohibitive. LieOTMap-FFT accelerates the core convolution operations within the Sinkhorn iterations by using the Fast Fourier Transform (FFT), reducing the complexity from O(N^2) to O(N log N) and making the method practical for large-scale data.

4.  **Map Representation & Scoring Kernel:** A TM-align-inspired similarity kernel is used to define the cost for optimal transport. This kernel, combined with a two-stage sigma-based thresholding of the maps, provides a robust score landscape that allows for effective global search.

By integrating these four components, LieOTMap-FFT creates a smooth and accurate optimization process capable of refining a structure from a random initial placement to a high-accuracy final pose.

---

## 2. Program Usage (fitter_sinkhorn_fft_v4.py)

This script performs the map-to-map fitting.

**Required Arguments:**

*   `--mobile_structure`: Path to the mobile structure file to be fitted (e.g., `1aon.cif`). Accepts `.cif` or `.pdb` formats.
*   `--target_map`: Path to the target cryo-EM density map file (e.g., `EMD-1046.map`). Accepts `.mrc` format.
*   `--gold_standard_structure`: Path to the gold-standard structure file for calculating the final RMSD to validate the result (e.g., `1GRU.cif`).

**Optional Arguments:**

*   `--output`: Path to save the fitted PDB structure. If not provided, a descriptive name will be generated automatically (e.g., `1aon_tm_fft_v4_rmsd_4.81.pdb`).
*   `--lr`: Learning rate for the Adam optimizer. Default: `0.01`.
*   `--steps`: Total number of optimization steps. Default: `100`.
*   `--d0`: The distance-scaling parameter in the TM-align-style scoring kernel. This is a crucial hyperparameter. Default: `8.0`.
*   `--score_scale`: A large scaling factor for the raw OT score to produce a numerically stable loss for the optimizer. Default: `10000.0`.
*   `--sinkhorn_iter`: Number of internal iterations for the Sinkhorn algorithm in each step. Default: `10`.
*   `--sigma_level`: The sigma level used for thresholding the target density map to reduce noise. Default: `3.0`.
*   `--mobile_sigma_level`: The sigma level used for thresholding the voxelized mobile structure map in each optimization step. Default: `1.0`.

---

## 3. Command Line Used for the Successful Run

The following command was used to generate the result with a final RMSD of 4.81 Å, as documented in the paper:

```bash
python fitter_sinkhorn_fft_v4.py --mobile_structure 1aon.cif --target_map EMD-1046.map --gold_standard_structure 1GRU.cif --lr 0.03 --d0 2.0 --steps 3000 --mobile_sigma_level 1.2
```
