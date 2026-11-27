# Gambler's Problem Value Iteration Recipes

Copy-paste code snippets for solving the Gambler's Problem and rendering dark-theme OceanViz figures.

---

## Recipe 1: Run Gambler's Problem Experiment Script

```matlab
% Executes Ex 4.9 experiment for ph = 0.40, ph = 0.25, and ph = 0.55
exp_gamblers_problem;
```

---

## Recipe 2: Run Generic Tabular Value Iteration

```matlab
% Create environment and generic ValueIterationSolver
env = OceanStats.RL.Env.Classic.ModifiedCarRentalEnv('MaxCars', 10);
solver = OceanStats.RL.DP.ValueIterationSolver(env, 'DiscountFactor', 0.9, 'Theta', 1e-4);

% Run Value Iteration loop
[V, pi, history] = solver.run();
```

---

## Recipe 3: Plot Dark-Theme Gambler's Problem Figure

```matlab
% Initialize OceanViz Core Dark theme
viz = OceanViz.Core('Theme', 'Dark');
theme = viz.Colors;

% Render Figure 4.3 styled 2-panel dark-mode figure
% Panel 1: Value estimates for sweeps 1, 2, 3, 32, final
% Panel 2: Final optimal policy stake vs capital
```
