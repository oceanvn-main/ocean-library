# Kernel Topology Ablation Study

## Overview

This guide explains the **Kernel Topology Ablation Study**, an experiment that isolates **matrix sparsity** as the definitive control variable governing manifold learning quality.

The experiment implements ESL §14.5.4 (Figure 14.30) and extends it with a cross-injection ablation (Stage 4) to prove that algorithm names are irrelevant — the topology of the matrix is what determines geometric success.

---

## The Core Argument

Two algorithms (KPCA and Spectral Clustering) both hinge on eigenvalue decomposition of an N-by-N matrix. The question is: **what happens when you feed the wrong matrix topology into each pipeline?**

| Branch | Matrix | Topology | Eigen direction |
|--------|--------|----------|-----------------|
| KPCA | K_tilde (double-centered RBF) | Dense — global | Top (largest) |
| Spectral | L = G - W (kNN Laplacian) | Sparse — local | Bottom (smallest) |

---

## Stage 1: Data and Kernel Generation

Three concentric rings are synthesized in R^2 with injected Gaussian noise:

```
ring k: X(i,1) = r_k * cos(theta) + eps,  eps ~ N(0, sigma^2)
         X(i,2) = r_k * sin(theta) + eps
radii: r = [1, 3, 5]
```

The dense RBF kernel matrix K is computed as:

```
K(i,j) = exp( -||xi - xj||^2 / (2 * c^2) )
```

**Key invariants enforced by assertion:**
- `diag(K) = 1.0` exactly (exp(0) = 1)
- All off-diagonal elements > 0 (RBF is strictly positive)

The bandwidth parameter `c` is swept over `[0.5, 1, 2, 5, 10]`.

> **Note:** `FullyConnectedRBFStrategy` zeroes the diagonal for graph use. For KPCA, the kernel diagonal must be 1. The script computes K directly without calling the strategy.

---

## Stage 2: Topological Manipulation

### KPCA Branch — Double-Centering

The centering projection operator is:

```
M = (1/N) * ones(N, N)
H = I_N - M
K_tilde = H * K * H
```

`K_tilde` is fully dense (sparsity ≈ 0%). It preserves global pairwise distances.

### Spectral Branch — Sparse Laplacian

A binary kNN mask is generated via `MutualKNNStrategy` and applied element-wise to K:

```
W = mask .* K         (RBF weights, truncated to kNN neighborhood)
G = diag(sum(W, 2))   (degree matrix)
L = G - W             (unnormalized Laplacian)
```

`L` is sparse (sparsity >> 50% for k=10 on N=900). It encodes only local structure.

**Sparsity is logged for both matrices** at the end of Stage 2.

---

## Stage 3: Spectral Analysis

### Eigenvalue Decomposition

- `K_tilde`: extract **top** m eigenvectors (largest eigenvalues) — these are the KPCA principal components.
- `L`: extract **bottom** m eigenvectors (smallest eigenvalues) — eigenvector 1 is trivial (constant), so clusters are found in eigenvectors 2 and 3.

### Eigengap Test

For 3 perfectly separated rings, the graph Laplacian L has exactly 3 zero eigenvalues (one per connected component). A pronounced gap between eigenvalue 3 and eigenvalue 4 signals the number of clusters:

```
n_components = sum(lambda_L < 1e-8)   % should equal 3
eigengap = lambda_L(4) - lambda_L(3)  % should be >> 0
```

### Indicator Functions

- **Spectral eigenvectors** form **discrete step-functions** (block-constant per cluster) — a consequence of the sparse binary topology.
- **KPCA eigenvectors** form **continuous sinusoidal waves** — a consequence of the global dense kernel.

---

## Stage 4: Cross-Injection Ablation

### Cross A: Sparse W into KPCA Pipeline

```
K_tilde_W = H * W * H
Extract top eigenvectors
```

**Failure mode:** The weak block structure of W cannot sustain a stable global projection. Result: V-shaped collapse or overlapping blobs.

### Cross B: Dense K into Laplacian Pipeline

```
G_K = diag(sum(K, 2))
L_K = G_K - K
Extract bottom eigenvectors
```

**Failure mode:** The dense degree matrix G_K overwhelms W. The eigengap disappears; no meaningful cluster structure in the projection.

---

## Conclusion

> The control variable is matrix sparsity, not the algorithm name.

| Property | Dense (KPCA) | Sparse (Spectral) |
|----------|--------------|-------------------|
| Memory | O(N^2) | O(N * k) |
| Projection | Continuous (inductive) | Discrete (transductive only) |
| Bandwidth sensitivity | High | Low |
| Nested manifold handling | Requires careful c tuning | Robust by topology |

---

## See Also

- `recipes/kpca_vs_spectral.md`
- `+OceanStats/+Unsupervised/SpectralClustering.m`
- `+OceanStats/+Unsupervised/+Affinity/MutualKNNStrategy.m`
- `+OceanStats/+Unsupervised/+Affinity/FullyConnectedRBFStrategy.m`
- `+OceanStats/+Pipeline/PCA.m`
- ESL §14.5.4
