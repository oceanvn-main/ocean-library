# Tabular Reinforcement Learning Recipes

Copy-paste code snippets for solving tabular MDPs and visualizing RL outputs in Ocean Library.

---

## Recipe 1: Solve Jack's Car Rental (Exercise 4.7)

```matlab
% Instantiate environment with Ex 4.7 parameters
env = OceanStats.RL.Env.Classic.ModifiedCarRentalEnv(...
    'MaxCars', 20, ...
    'MaxMove', 5, ...
    'RentReward', 10, ...
    'MoveCost', 2, ...
    'ParkingThreshold', 10, ...
    'ParkingPenalty', 4, ...
    'FreeShuttle', true ...
);

% Create solver and execute policy iteration
solver = OceanStats.RL.DP.PolicyIterationSolver(env, 'DiscountFactor', 0.9, 'Theta', 1e-4);
[V, pi, history] = solver.run();

% Display summary
fprintf('Policy Iteration Converged in %d steps (%d evaluation sweeps total).\n', ...
    history.PolicySteps, sum(history.EvalSweeps));
```

---

## Recipe 2: Render High-Contrast Dark-Theme RL Visualizations

```matlab
% Pass the solver object directly to plot_tabular_rl
fig = plot_tabular_rl(solver, 'Title', 'Exercise 4.7 Optimal Control');

% Or pass individual matrices
fig = plot_tabular_rl(solver.ValueMatrix, solver.PolicyMatrix);
```

---

## Recipe 3: Parameter Sweep over Parking Penalty

```matlab
% Study policy sensitivity to parking penalty values [0, 2, 4, 8]
penalties = [0, 2, 4, 8];
for p = penalties
    env = OceanStats.RL.Env.Classic.ModifiedCarRentalEnv('ParkingPenalty', p);
    solver = OceanStats.RL.DP.PolicyIterationSolver(env);
    [V, pi] = solver.run();
    fprintf('Penalty $%.0f: Max Return V(20,20) = %.2f\n', p, V(21,21));
end
```
