# Sparse PCA Guide

## Overview

`OceanStats.Unsupervised.SparsePCA` implements Sparse Principal Component Analysis (SPCA) following **Zou, Hastie & Tibshirani (2006)**. It produces loading vectors with many **exactly-zero** entries, making principal components easier to interpret than standard PCA.

**Reference**: ESL §14.5.5, Eq. 14.70–14.71.

### When to use it

- You have **p >> N** (more features than samples) and need unique, interpretable components.
- You want to identify which features *actually drive* each principal axis.
- You need a sparse approximation for downstream models or visualization.

---

## Algorithm

For a single sparse PC, SPCA minimizes:

```
min over (theta, v):  sum_i ||x_i - theta * v^T x_i||^2 + lambda*||v||^2 + lambda1*||v||_1
subject to: ||theta||_2 = 1
```

This is solved by **alternating minimization** (two steps, iterated to convergence):

| Step | Variable Fixed | Update |
|------|---------------|--------|
| **A — Procrustes** | V | theta = Xc*v / ||Xc*v|| |
| **B — Elastic-Net** | Theta | v_ridge = (X^T X + lambda*I)^-1 X^T theta ; v = SoftThreshold(v_ridge, lambda1/2) |

Multi-component extraction uses **deflation**: after fitting component k, subtract its reconstruction from X before fitting k+1.

---

## Usage

```matlab
spca = OceanStats.Unsupervised.SparsePCA();

% Basic fit
spca.fit(X, K, 'Lambda', 1e-4, 'Lambda1', 0.1);

% Access results
V     = spca.Loadings;            % (p x K) sparse loading matrix
Z     = spca.Scores;              % (N x K) projection scores
sparsity = spca.sparsityRatio();  % (1 x K) fraction of zero loadings

% Project new data
Z_new = spca.transform(X_test);

% Print summary
spca.printSummary();
```

### Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `Lambda`  | `1e-4`  | L2 ridge: ensures uniqueness when p > N. Set to 0 if N > p and you want exact PCA recovery. |
| `Lambda1` | `0.5`   | L1 lasso: controls sparsity. `0` = dense (close to PCA); larger = sparser loadings. |
| `MaxIter` | `500`   | Max iterations per component. |
| `Tol`     | `1e-6`  | Convergence tolerance on `||v_new - v_old||_2`. |
| `Verbose` | `false` | Print per-component convergence info. |

---

## Examples

### Lambda1 sweep (sparsity control)

```matlab
lambda1_vals = [0, 0.05, 0.1, 0.3, 0.6, 1.0];
for i = 1:numel(lambda1_vals)
    s = OceanStats.Unsupervised.SparsePCA();
    s.fit(X, 3, 'Lambda1', lambda1_vals(i));
    fprintf('lambda1=%.2f | sparsity=%.1f%%\n', lambda1_vals(i), 100*s.sparsityRatio()(1));
end
```

### Recovering standard PCA

```matlab
% Set both penalties to 0 (and N > p) to recover standard PCA
spca = OceanStats.Unsupervised.SparsePCA();
spca.fit(X, K, 'Lambda', 0, 'Lambda1', 0);
% spca.Loadings should match the first K columns of V from svd(X - mean(X))
```

### Comparison figure (standard vs sparse loadings as images)

```matlab
viz = OceanViz.Core('Theme', 'Dark');
figure;
for k = 1:K
    % Row 1: standard PCA
    ax = subplot(2, K, k);
    imagesc(ax, reshape(V_pca(:,k), 16, 16)'); colormap(ax, gray);
    % Row 2: sparse PCA
    ax2 = subplot(2, K, K+k);
    imagesc(ax2, reshape(spca.Loadings(:,k), 16, 16)'); colormap(ax2, gray);
end
```

---

## See Also

- `OceanStats.Pipeline.PCA` — standard PCA wrapper
- `OceanStats.Regularization.L1Regularizer` — soft-thresholding (used internally)
- `OceanStats.Regularization.L2Regularizer` — ridge shrinkage (used internally)
- `exp_pca_digits.m` — standard PCA baseline experiment
- `exp_spca_digits.m` — sparse PCA experiment with lambda1 sweep
- ESL §14.5.5, Zou et al. (2006) J. Comput. Graph. Stat.
