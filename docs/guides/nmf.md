# NMF — Non-negative Matrix Factorization

## Overview

Non-negative Matrix Factorization (NMF) approximates a non-negative data matrix `X` as a
product of two non-negative matrices:

```
X  ≈  W * H
```

| Symbol | Shape   | Meaning                                         |
|--------|---------|-------------------------------------------------|
| X      | (N x p) | Input data (must be non-negative, e.g. pixels)  |
| W      | (N x r) | Activation matrix — coefficients per sample     |
| H      | (r x p) | Basis matrix — r learned basis images           |
| r      | scalar  | Rank of factorization                           |

The non-negativity constraint forces **parts-based** representations: components can only
add to, never cancel each other, unlike PCA which allows mixed-sign loadings.

**Reference**: Lee & Seung (1999, 2001). ESL §14.6.

---

## Algorithm — Lee & Seung Multiplicative Updates

**Objective** (Frobenius norm):

```
min_{W >= 0, H >= 0}  ||X - W * H||_F^2
```

**Update rules** (alternating, each iteration monotonically decreases the objective):

```
H  ←  H  .*  (W^T * X)          ./  (W^T * W * H  +  eps)
W  ←  W  .*  (X  * H^T)         ./  (W  * H * H^T +  eps)
W  :=  normalize columns of W to unit norm
```

The `.* ... ./` notation is element-wise multiply and divide.  
`eps` (1e-10) prevents division by zero; does not affect correctness.  
Normalization after each W update transfers scale to H and prevents unbounded growth.

**Convergence**: checked as relative change in `||X - WH||_F^2` between successive iterations.

---

## Usage

```matlab
% Preprocessing: X must be non-negative
X_nn = (X_raw + 1) / 2;   % shift from [-1,1] to [0,1]

% Fit NMF with r=25 basis images
nmf = OceanStats.Unsupervised.NMF();
nmf.fit(X_nn, 25, 'MaxIter', 300, 'Tol', 1e-4, 'Verbose', true);

% Access results
H     = nmf.getBasisImages();   % (25 x p) basis images
X_hat = nmf.reconstruct();      % (N x p) training reconstruction = W*H
W_new = nmf.transform(X_test);  % (M x 25) activate on new data

% Summary
nmf.printSummary();
```

---

## Examples

### Basic digit decomposition

```matlab
% Load USPS digit data
raw  = load('Data/zip/zip.train/zip.train');
mask = raw(:,1) == 3;
X_nn = (raw(mask, 2:end) + 1) / 2;    % shift to [0,1]

nmf = OceanStats.Unsupervised.NMF();
nmf.fit(X_nn, 25, 'MaxIter', 500, 'Verbose', true);
nmf.printSummary();
```

### Visualize basis images

```matlab
H  = nmf.getBasisImages();    % (r x p)
img_dim = 16;
for k = 1:size(H, 1)
    img = reshape(H(k, :), img_dim, img_dim)';
    subplot(5, 5, k); imagesc(img); colormap hot; axis image off;
end
```

### Compare with PCA and VQ

See full comparison experiment: `Experiments/Scripts/Unsupervised/exp_nmf_digits.m`

---

## Key Properties vs PCA and VQ

| Property                 | NMF             | PCA             | VQ (K-Means)      |
|--------------------------|-----------------|-----------------|-------------------|
| Loading signs            | Non-negative    | Mixed ±         | Non-negative      |
| Reconstruction type      | Soft (additive) | Soft (subspace) | Hard (nearest 1)  |
| Components interpretable | Parts-based     | Holistic        | Prototype images  |
| MSE vs optimal           | Higher than PCA | Optimal (L2)    | Higher than NMF   |
| Uniqueness               | Not guaranteed  | Unique          | Local optima      |
| Non-negativity required  | Yes (input)     | No              | Yes (pixel data)  |

**Expected MSE ordering**: `PCA <= NMF <= VQ`

---

## See Also

- `OceanStats.Pipeline.PCA` — standard PCA via economy SVD
- `OceanStats.Unsupervised.SparsePCA` — sparse PCA with L1 penalty on loadings
- `Experiments/Scripts/Unsupervised/exp_nmf_digits.m` — full NMF vs PCA vs VQ comparison
