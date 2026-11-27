# Scattered Data Interpolation Guide

## Overview

Scattered data interpolation addresses the problem of reconstructing a continuous scalar field `f(x)` from `N` irregularly sampled observation points `(X, y)` in `D` dimensions. Unlike grid-based interpolation (e.g., bilinear or bicubic interpolation on structured pixel grids), scattered data interpolation requires mesh-free or triangulation-based methods that do not assume a regular spatial layout.

This guide details the mathematical foundations and usage of the `+OceanMath/+Interpolation/` package, covering Radial Basis Functions (RBF), Nadaraya-Watson Kernel Regression, Delaunay Triangulation (Natural Neighbour), K-Nearest Neighbours (KNN), and 1D Smoothing Splines.

---

## Mathematical Models

### 1. Radial Basis Function (RBF) Interpolation (Szeliski §4.1.1, eq. 4.3–4.11)

An RBF interpolant models the unknown function as a linear superposition of radially symmetric basis functions centered at each training point:

`f(x) = sum_{k=1}^N w_k * phi(||x - x_k||)`

Where `phi(r)` is the radial kernel. For exact interpolation (`lambda = 0`), the weights `w` are determined by solving the `N x N` linear system:

`K * w = y,   where K_ij = phi(||x_i - x_j||)`

For noisy data, Tikhonov regularization (quadratic penalty `p=2`, eq. 4.11) is added to the diagonal:

`(K + lambda * I) * w = y`

Supported RBF kernels in `+OceanMath/+Image/+Resampling/`:
- **Gaussian** (eq. 4.4): `phi(r) = exp(-r^2 / c^2)`
- **Hardy Multiquadric** (eq. 4.5): `phi(r) = sqrt(r^2 + c^2)`
- **Inverse Multiquadric** (eq. 4.6): `phi(r) = 1 / sqrt(r^2 + c^2)`
- **Thin Plate Spline** (eq. 4.7): `phi(r) = r^2 * log(r)`

### 2. Nadaraya-Watson Kernel Regression (Szeliski §4.1.1, eq. 4.12–4.14)

Instead of solving an `N x N` linear system, kernel regression estimates `f(x)` at query locations directly as a normalized distance-weighted average of all training points:

`f(x) = (sum_k y_k * phi(||x - x_k||)) / (sum_l phi(||x - x_l||))`

The denominator enforces a partition of unity (`sum_k phi'_k(x) = 1`), guaranteeing that estimates are bounded within `[min(y), max(y)]`. This method requires zero training phase (lazy evaluation).

### 3. Delaunay Triangulation & Natural Neighbour (Szeliski §4.1)

Partitions the domain into non-overlapping simplices (triangles in 2D) whose vertices are the training points. Inside each simplex:
- **Natural Neighbour**: Computes Sibson area-overlap weights from the Voronoi tessellation, yielding `C1` smooth interpolation.
- **Linear**: Uses barycentric coordinates within each simplex (`C0` continuous).
- **Nearest**: Piecewise-constant nearest-vertex value (`C-1`).

### 4. K-Nearest Neighbour (KNN) Regression

Estimates `f(x)` from the `K` closest training points:

`f_hat(x) = (1 / Z) * sum_{k in N_K(x)} w_k * y_k`

Supports uniform weights (`w_k = 1/K`) or distance weights (`w_k = 1/||x - x_k||`), using Euclidean, Manhattan, or Chebyshev distance metrics.

### 5. 1D Smoothing Splines (Szeliski §4.1, ESL §5.4)

For 1D inputs (`D=1`), minimizes the regularized energy:

`E(f) = sum_{i=1}^N (y_i - f(x_i))^2 + lambda * integral (f''(x))^2 dx`

By the Representer Theorem, the minimizer is a natural cubic spline with knots at all training locations, solved via `(B + lambda * Omega) * c = y`.

---

## Usage

Use `OceanMath.Interpolation.ScatteredInterpolant` as the unified master class:

```matlab
import OceanMath.Interpolation.ScatteredInterpolant

% 1. Create an RBF model with Gaussian kernel directly
model_rbf = ScatteredInterpolant('Method', 'rbf', 'Kernel', 'gaussian', 'Epsilon', 0.5, 'Lambda', 1e-3);
model_rbf.fit(X_train, Y_train);
Y_pred = model_rbf.predict(X_query);

% 2. Create a Nadaraya-Watson kernel regression model
model_kr = ScatteredInterpolant('Method', 'kernel-regression', 'Kernel', 'gaussian', 'Epsilon', 0.2);
model_kr.fit(X_train, Y_train);
Y_pred_kr = model_kr.predict(X_query);

% 3. Create a Natural Neighbour interpolant
model_nat = ScatteredInterpolant('Method', 'natural', 'SubMethod', 'natural');
model_nat.fit(X_train, Y_train);
Y_pred_nat = model_nat.predict(X_query);

% 4. Create a KNN regression model
model_knn = ScatteredInterpolant('Method', 'knn', 'K', 5, 'Weights', 'distance');
model_knn.fit(X_train, Y_train);
Y_pred_knn = model_knn.predict(X_query);

% 5. Access pedagogical metadata for GUI display
meta = OceanMath.Interpolation.ScatteredInterpolant.getMetadata('rbf');
disp(meta.display_name);
disp(meta.pedagogy.motivation);
```

---

## Examples

### Complete Experiment Pipeline with Cross-Validation

```matlab
import OceanMath.Interpolation.ScatteredInterpolant

% 1. Generate synthetic 2D scattered data
gen = OceanData.Generators.SmoothFunctionGenerator('N', 60, 'Dim', 2, 'Mode', 'mixed', 'NoiseStd', 0.05);
[X_tr, Y_tr] = gen.generate();

% 2. Fit multiple models using ScatterFitter orchestrator
fitter = OceanStats.Interpolation.ScatterFitter('Dim', 2, 'NGrid', 100);
fitter.addModel('RBF-Gaussian', ScatteredInterpolant('Method', 'rbf', 'Kernel', 'gaussian', 'Epsilon', 0.3));
fitter.addModel('KernelRegression', ScatteredInterpolant('Method', 'kernel-regression', 'Kernel', 'gaussian', 'Epsilon', 0.3));
fitter.addModel('Natural', ScatteredInterpolant('Method', 'natural'));
fitter.fitAll(X_tr, Y_tr);

% 3. Evaluate errors against ground truth handle
fitter.evaluateAll(gen.MetaData.f_true);
fitter.printSummary();

% 4. Tune RBF Epsilon using 5-Fold Cross-Validation
cv = OceanStats.Interpolation.InterpolationCV('ModelType', 'rbf', 'TuneParam', 'rbf-epsilon', 'KFolds', 5);
cv.fit(X_tr, Y_tr);
fprintf('Best tuned Epsilon = %.4f\n', cv.BestParam);
```

---

## See Also

- `OceanMath.Interpolation.ScatteredInterpolant` — Unified master scattered data interpolant.
- `OceanStats.Interpolation.ScatterFitter` — Multi-model evaluation orchestrator.
- `OceanStats.Interpolation.InterpolationCV` — K-Fold cross-validation hyperparameter tuner.
- `OceanViz.Strategies.ScatterInterpStrategy` — Dark-theme multi-panel visualization strategy.
