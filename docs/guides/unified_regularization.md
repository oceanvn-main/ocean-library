# Domain-Agnostic Regularization Architecture

## Overview

The **Domain-Agnostic Regularization Architecture** unifies 1D time-series signal filtering (L1 trend filtering / Hodrick-Prescott), 2D image denoising (Total Variation / edge-preserving regularization), 3D volumetric smoothing, and N-D Graph ML regularization under a single reusable strategy pattern.

By expressing every variational objective function using linear difference operators ($D$), all variational solvers (`IRLSSolver`, `PrimalDualSolver`, `GNCSolver`) and penalty strategies (`IRobustPenalty`) become **100% domain-agnostic**.

---

## Core Equation

$$\min_f \frac{1}{2} \|f - g\|^2 + \lambda \sum_k \rho( (D f)_k )$$

$$\text{IRLS Update Step: } (I + \lambda D^T W D) f_{k+1} = g$$

where $W = \text{diag}(w(D f))$ and $w(d) = \rho(d) / d^2$.

---

## Architecture Components

### 1. Robust Penalties (`OceanMath.Regularization.*`)
- `IRobustPenalty`: Abstract interface (`evaluate`, `weight`, `derivative`, `isConvex`).
- Strategies: `QuadraticPenalty`, `L1Penalty`, `PNormPenalty`, `HuberPenalty`, `GNCPenalty`, `PenaltyFactory`.

### 2. Difference Operators (`OceanMath.Operators.*`)
- `IDifferenceOperator`: Abstract interface for linear difference matrices.
- `DifferenceOperator1D`: 1D Order 1 (TV) and Order 2 (L1 trend / Hodrick-Prescott).
- `DifferenceOperator2D`: 2D image spatial differences.
- `DifferenceOperatorGraph`: Graph incidence matrix ($E \times V$).

### 3. Variational Energy & Solvers (`OceanMath.Variational.*`)
- `VariationalEnergy`: Combines any `IDifferenceOperator` and `IRobustPenalty`.
- `ImageRegularizationEnergy`: Specializes `VariationalEnergy` for 2D images.

---

## Examples

```matlab
% 1D Trend Filtering Example
import OceanMath.Operators.DifferenceOperator1D
import OceanMath.Regularization.L1Penalty
import OceanMath.Variational.VariationalEnergy
import OceanMath.Image.Variational.Solvers.IRLSSolver

op1d   = DifferenceOperator1D('Order', 2);
pen    = L1Penalty();
energy = VariationalEnergy(op1d, pen, 'Lambda', 15.0);

solver = IRLSSolver('MaxIter', 100);
f_trend = solver.solve(energy, g_noisy_signal);
```

See [`exp_1d_trend_filtering.m`](file:///c:/Ocean%20Library/Matlab%20code%20practice/Experiments/Scripts/Time%20Series%20Analysis/exp_1d_trend_filtering.m) for a complete working demonstration.
