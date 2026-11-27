# Factor Analysis

## Overview

Factor Analysis models observed variables as linear combinations of a small number of **latent factors** plus unique noise:

```
X = A * S + ε,    Σ = A * A^T + D_ε   (ESL Eq. 14.81)
```

| Symbol | Shape | Meaning |
|--------|-------|---------|
| A      | p × q | Factor loading matrix |
| S      | N × q | Latent factor scores (unobserved) |
| D_ε    | p × p | Diagonal specific variance matrix |
| q      | scalar | Number of factors (q << p) |

**Reference**: ESL §14.7.1, Mardia et al. (1979).

---

## Key Contrast with PCA

| Property | PCA | Factor Analysis |
|---|---|---|
| Objective | Maximize variance | Explain correlation structure |
| Finds | Max-variance directions | Shared latent factors |
| High-variance isolated variables | Captured in PC1 | Relegated to specific noise D_ε |
| Rotation invariance | Unique (SVD) | Not unique (rotations preserve fit) |

**ESL Ex. 14.15 demonstration**: With X3=10*Z3 (high variance, uncorrelated):
- PC1 ≈ [0, 0, 1] — points along X3 (max variance wins)
- FA1 ≈ [0.7, 0.7, 0] — recovers shared latent Z1 driving X1≈X2

---

## Usage

```matlab
fa = OceanStats.Unsupervised.FactorAnalysis();
fa.fit(X, 1, 'Rotation', 'none');    % q=1 factor, no rotation

A = fa.getLoadings();      % (p x q) loading matrix
S = fa.transform(X_new);   % (M x q) Thompson factor scores
Sigma = fa.getCovModel();  % (p x p) model covariance = A*A' + diag(Psi)
fa.printSummary();
```

---

## See Also

- `Experiments/Scripts/Unsupervised/exp_factor_analysis.m` — ESL Ex. 14.15 experiment
- `OceanStats.Unsupervised.SparsePCA` — sparse loading variant of PCA
- `OceanStats.Unsupervised.NMF` — non-negative factorization
