# Likelihood Surface Optimization Recipes

Copy-paste recipes for running numerical optimization algorithms (Steepest Descent, Newton-Raphson, DFP, and BFGS) on likelihood surfaces using `OceanAlgo.Optimization.MatlabOptimizationWrapper` and `OceanStats.Time.ARMALikelihood`.

---

## 1. Minimizing a Standard Function with BFGS or DFP

```matlab
% Objective function: Rosenbrock function
rosenbrock = @(x) 100*(x(2) - x(1)^2)^2 + (1 - x(1))^2;
x0 = [-1.2, 1.0];

wrapper = OceanAlgo.Optimization.MatlabOptimizationWrapper();

% Run BFGS
[x_bfgs, f_bfgs, hist_bfgs] = wrapper.solve(rosenbrock, x0, ...
    'Method', 'bfgs', ...
    'IsMaximization', false);

% Run DFP
[x_dfp, f_dfp, hist_dfp] = wrapper.solve(rosenbrock, x0, ...
    'Method', 'dfp', ...
    'IsMaximization', false);

fprintf('BFGS iterations: %d, Optimal f: %.6f\n', hist_bfgs.iterations, f_bfgs);
fprintf('DFP iterations:   %d, Optimal f: %.6f\n', hist_dfp.iterations, f_dfp);
```

---

## 2. Maximizing ARMA(1,1) Log-Likelihood

```matlab
% 1. Generate synthetic ARMA(1,1) series
y = OceanStats.Time.ARMAProcess.simulate(0.7, -0.4, 1.0, 500, 42);

% 2. Define likelihood handle
likeFun = @(p) OceanStats.Time.ARMALikelihood.logLikelihood(p, y, 1.0);

% 3. Run solver with Steepest Descent
wrapper = OceanAlgo.Optimization.MatlabOptimizationWrapper();
[p_opt, max_L, history] = wrapper.solve(likeFun, [0.0, 0.0], ...
    'Method', 'steepdesc', ...
    'IsMaximization', true);

fprintf('Estimated parameters: phi = %.4f, theta = %.4f\n', p_opt(1), p_opt(2));
fprintf('Max Log-Likelihood: %.4f across %d iterations\n', max_L, history.iterations);
```

---

## 3. Sweeping Log-Likelihood Surface and Rendering OceanViz Dashboard

```matlab
% 1. Synthesize series
y = OceanStats.Time.ARMAProcess.simulate(0.7, -0.4, 1.0, 500, 42);

% 2. Sweep 2D terrain grid
[PhiGrid, ThetaGrid, LGrid] = OceanStats.Time.ARMALikelihood.gridSweep(y, 1.0, [-0.9, 0.9], [-0.9, 0.9], 100);

% 3. Run all four solvers
likeFun = @(p) OceanStats.Time.ARMALikelihood.logLikelihood(p, y, 1.0);
wrapper = OceanAlgo.Optimization.MatlabOptimizationWrapper();
start_param = [-0.6, 0.7];

histories = struct();
[~, ~, histories.Steepest] = wrapper.solve(likeFun, start_param, 'Method', 'steepdesc', 'IsMaximization', true);
[~, ~, histories.Newton]   = wrapper.solve(likeFun, start_param, 'Method', 'newton',    'IsMaximization', true);
[~, ~, histories.DFP]      = wrapper.solve(likeFun, start_param, 'Method', 'dfp',       'IsMaximization', true);
[~, ~, histories.BFGS]     = wrapper.solve(likeFun, start_param, 'Method', 'bfgs',      'IsMaximization', true);

% 4. Render OceanViz Dark Dashboard
vizData = struct();
vizData.PhiGrid    = PhiGrid;
vizData.ThetaGrid  = ThetaGrid;
vizData.LGrid      = LGrid;
vizData.TrueParam  = [0.7, -0.4];
vizData.StartParam = start_param;
vizData.Histories  = histories;

viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('Likelihood Optimization Dashboard');
ax = axes('Parent', fig);
viz.plot(ax, "LikelihoodOptimization", vizData);
```

---

## See Also
* [`exp_likelihood_surface_optimization.m`](file:///c:/Ocean%20Library/Matlab%20code%20practice/Experiments/Scripts/Time%20Series%20Analysis/exp_likelihood_surface_optimization.m)
* [`MatlabOptimizationWrapper.m`](file:///c:/Ocean%20Library/Matlab%20code%20practice/+OceanAlgo/+Optimization/MatlabOptimizationWrapper.m)
* [`ARMALikelihood.m`](file:///c:/Ocean%20Library/Matlab%20code%20practice/+OceanStats/+Time/ARMALikelihood.m)
