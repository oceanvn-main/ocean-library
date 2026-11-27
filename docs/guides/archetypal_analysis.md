# Archetypal Analysis

## Overview

Archetypal Analysis (Cutler & Breiman, 1994) finds **r archetypes** that are convex combinations of the data points, then represents each data point as a convex combination of those archetypes. This forces archetypes to lie on the **convex hull** of the data — they are "pure" or "extreme" examples.

**Reference**: ESL §14.6.1, Cutler & Breiman (1994).

---

## Formulation (ESL Eq. 14.75–14.77)

```
X ≈ W * H,   where H = B * X
J(W, B) = ||X - W * B * X||_F^2
```

| Matrix | Shape  | Constraints                        | Meaning                                  |
|--------|--------|------------------------------------|------------------------------------------|
| W      | N × r  | w_ik >= 0,  rows sum to 1          | Each sample as convex combo of archetypes |
| B      | r × N  | b_ki >= 0,  rows sum to 1          | Each archetype as convex combo of data    |
| H=B*X  | r × p  | Derived from B                     | Archetype images (on convex hull)         |

---

## Comparison: AA vs NMF vs K-Means

| Property              | Archetypal Analysis | NMF                | K-Means          |
|-----------------------|---------------------|--------------------|------------------|
| W constraints         | rows sum to 1       | non-negative only  | one-hot rows     |
| H constraints         | H = B*X (data hull) | free non-negative  | free centroids   |
| Prototype location    | Convex hull         | Anywhere >=0       | Data interior    |
| Reconstruction        | Convex superposition| Additive parts     | Nearest centroid |

---

## Algorithm

```
Initialize B randomly (rows uniform on simplex)
Repeat:
  W step — fix H=B*X, per row i:
    w_i = lsqnonneg(H', x_i') then normalize sum(w_i)=1

  B step — fix W, per row k:
    Stage 1: h_k_ols = W[:,k]' * R_k / ||W[:,k]||^2   (OLS in data space)
    Stage 2: b_k = lsqnonneg(Xc', h_k_ols') then normalize sum(b_k)=1
    Update H[k,:] = b_k * Xc

Until convergence of ||X - WH||_F
```

Run multiple random restarts (`NumRestarts`) and keep the best solution.

---

## Usage

```matlab
aa = OceanStats.Unsupervised.ArchetypalAnalysis();
aa.fit(X, r, 'NumRestarts', 5, 'MaxIter', 300, 'Verbose', true);

H     = aa.getArchetypes();     % (r x p) archetype images
X_hat = aa.reconstruct();       % (N x p) training reconstruction
W_new = aa.transform(X_test);   % (M x r) convex weights for new data
aa.printSummary();
```

---

## Expected Behavior

- **2D synthetic data**: Archetypes appear at or near the vertices of the data's convex hull, while K-Means centroids sit in the interior.
- **Image data (digit 3)**: With r=2, archetypes show contrasting extreme styles (e.g., thin vs fat, left-leaning vs right-leaning). With r=4, four more distinct extremes.
- **Reconstruction error**: Monotonically decreases with iterations; higher r gives lower error.

---

## See Also

- `OceanStats.Unsupervised.NMF` — non-negative factorization (parts-based)
- `Experiments/Scripts/Unsupervised/exp_archetypal_digits.m` — full experiment (Figs 14.35 & 14.36)
