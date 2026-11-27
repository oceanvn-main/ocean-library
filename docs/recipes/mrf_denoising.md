# Recipe: MRF Spatial Prior Energy Denoising

This recipe demonstrates how to solve 2D Markov Random Field (MRF) spatial prior energy minimization using `OceanMath.Variational` and plot the resulting state transitions using `OceanViz.Core`.

```matlab
import OceanMath.Operators.DifferenceOperator2D
import OceanMath.Regularization.L1Penalty
import OceanMath.Variational.VariationalEnergy
import OceanMath.Image.Variational.Solvers.IRLSSolver
import OceanViz.Core

% 1. Create energy formulation: E(X) = 0.5*||X - Y||^2 + lambda * sum( rho(D*X) )
op2d   = DifferenceOperator2D();
pen    = L1Penalty('Epsilon', 1e-5);
energy = VariationalEnergy(op2d, pen, 'Lambda', 0.8);

% 2. Solve using OceanMath IRLSSolver
solver = IRLSSolver('MaxIter', 30, 'Tol', 1e-6);
X_restored = solver.solve(energy, Y_noisy);

% 3. Render result using OceanViz Engine
viz  = Core('Theme', 'Dark');
fig  = figure('Color', viz.Colors.Background);
ax   = axes('Parent', fig);
data = struct('Image', X_restored, 'Title', 'MRF Restored Matrix');

viz.plot(ax, "Image", data);
```

See script [`exp_mrf_denoising.m`](file:///c:/Ocean%20Library/Matlab%20code%20practice/Experiments/Scripts/Computer%20Vision/exp_mrf_denoising.m) for the full 6-panel experiment.
