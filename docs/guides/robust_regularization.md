# Robust Image Regularization

## Overview

Implements **Szeliski §4.2 Robust Regularization** — the framework for replacing
the standard quadratic (L2) smoothness penalty with non-quadratic functions that
better preserve sharp edges and handle outliers.

The core idea: instead of penalising all image differences equally with rho(d) = d^2,
use a penalty that grows more slowly for large differences (edges):

```
E_IR(f) = sum_{i,j}  s_x * rho(f(i+1,j) - f(i,j))
                    + s_y * rho(f(i,j+1) - f(i,j))
```

**Package locations:**
- `+OceanMath/+Image/+Regularization/` — Penalty functions & strategies
- `+OceanMath/+Image/+Variational/` — Energy formulation & solvers (`+Solvers/`)

---

## Penalty Function Family

| Class | Formula | Convex | Edge Behaviour |
|---|---|---|---|
| `QuadraticPenalty` | rho(d) = d^2 | Yes | Blurs all edges equally |
| `L1Penalty` | rho(d) = |d| | Yes | Total Variation — sharp edges, staircase |
| `PNormPenalty` | rho(d) = |d|^p, p in (0,2] | p >= 1 | Hyper-Laplacian for p < 1 |
| `HuberPenalty` | piecewise L2/L1 blend | Yes | Smooth compromise |
| `GNCPenalty` | min(d^2, V) | No | Hard outlier rejection |

Use `PenaltyFactory.create(name)` for string-based construction.

---

## Usage

### Basic TV Denoising (IRLS)

```matlab
import OceanMath.Image.Regularization.*
import OceanMath.Image.Variational.*
import OceanMath.Image.Variational.Solvers.*

pen    = L1Penalty();
energy = ImageRegularizationEnergy(pen, 'Lambda', 0.1);
solver = IRLSSolver('MaxIter', 100);
f      = solver.solve(energy, g_noisy);
```

### Total Variation via Primal-Dual (Chambolle-Pock)

```matlab
solver = PrimalDualSolver('MaxIter', 1000);
f      = solver.solve(energy, g_noisy);
```

### GNC for Non-Convex Outlier Rejection

```matlab
pen    = GNCPenalty('V', 0.3);
energy = ImageRegularizationEnergy(pen, 'Lambda', 0.05);
solver = GNCSolver('NumScales', 10, 'VMax', 200);
f      = solver.solve(energy, g_noisy);
```

---

## Solver Selection Guide

| Penalty | Recommended Solver | Why |
|---|---|---|
| `QuadraticPenalty` | `IRLSSolver` (1 iter) | Exact direct solve in one iteration |
| `L1Penalty` (TV) | `PrimalDualSolver` | Avoids IRLS weight singularity at d=0 |
| `L1Penalty` (TV) | `IRLSSolver` | Also valid, numerically stabilised with Epsilon |
| `HuberPenalty` | `IRLSSolver` | Smooth weights; converges in ~10–30 iters |
| `PNormPenalty` p >= 1 | `IRLSSolver` | Convex; typically 20–50 iters |
| `PNormPenalty` p < 1 | `IRLSSolver` | Non-convex; use warm restart |
| `GNCPenalty` | `GNCSolver` | Non-convex; continuation required |

---

## IRLS Connection

The `IRLSSolver` iterates the weighted normal equations:

```
(I  +  lambda * D' * W_k * D) * f_{k+1}  =  g

where W_k = diag(w(D * f_k)),   w(d) = rho(d) / d^2
```

The weight function w(d) is exposed via `penalty.weight(d)`.
For `QuadraticPenalty`, w = 1 (constant), so the system converges in exactly 1 step
and matches a direct MATLAB `\` solve — verified in the unit test suite.

---

## Examples

See experiment scripts:

- [`exp_robust_penalties.m`](file:///c:/Ocean%20Library/Matlab%20code%20practice/Experiments/Scripts/Computer%20Vision/exp_robust_penalties.m) — Penalty family visualisation + 1D edge recovery
- [`exp_total_variation_denoising.m`](file:///c:/Ocean%20Library/Matlab%20code%20practice/Experiments/Scripts/Computer%20Vision/exp_total_variation_denoising.m) — TV denoising: IRLS vs Primal-Dual on 2D images
- [`exp_gnc_continuation.m`](file:///c:/Ocean%20Library/Matlab%20code%20practice/Experiments/Scripts/Computer%20Vision/exp_gnc_continuation.m) — Blake-Zisserman GNC scale annealing

## See Also

- [`image_denoising.md`](image_denoising.md) — Spatial filter denoising (Gaussian, Bilateral, NLM)
- [`pixel_transforms.md`](pixel_transforms.md) — Point transforms and tone mapping
- Szeliski (2021) §4.2 "Robust Regularization"
