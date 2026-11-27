# Real-time dynamic programming on an exact tabular model

`OceanStats.RL.DP.RTDPSolver` performs expected Bellman backups along sampled greedy trials. It reads an exact transition model; it does not call the environment's `step` method. A trial samples the next state only to decide where to back up next. The backed-up value uses the entire outcome distribution.

## Run and inspect one trial

```matlab
env = OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner', 19);
model = OceanStats.RL.Dyna.Models.ExactModelAdapter(env);
solver = OceanStats.RL.DP.RTDPSolver(model, ...
    'DiscountFactor', 1, 'Seed', 7, 'MaxDepth', 50);

startState = env.state_to_index(env.get_all_states(10,:));
trace = solver.runTrial(startState);
disp(trace.TerminationReason)
disp(trace.Steps)
```

Each `trace.Steps` entry shows the state, selected action, action values, sampled successor and reward, value before and after the backup, and absolute change (`Residual`). `trace.Backups` counts updates in this trial. A trial can stop at a terminal transition or its depth limit. `solver.getLastTrajectory()` returns the same trace.

## Solve within explicit budgets

```matlab
solver = OceanStats.RL.DP.RTDPSolver(model, ...
    'DiscountFactor', 1, 'Seed', 7, 'MaxDepth', 50);
[values, policy, result] = solver.solve(startState, ...
    'MaxTrials', 100, 'MaxBackups', 2000);

disp(result.Trials)
disp(result.Backups)
disp(result.TerminationReason)
disp(result.ExploredStates)
```

`values` has one entry per indexed model state. `policy(s)` is an action index for a visited state and zero for an unvisited state. An unvisited state's value remains its `InitialValue`. `result.Converged` is false when a trial or backup budget ends; this RTDP class does not certify solved states. `result.LastTrajectory` exposes the final trial. Use `solver.getValue(s)` and `solver.getPolicy(s)` for individual indexed states.

The constructor also accepts `InitialValue` (a finite column vector, one value per state) and `TieTolerance` (actions within this distance of the best value use the first legal action index). `solve` accepts `Tolerance` for result metadata; it is not a convergence certificate. An `Env.IModelEnvironment` is supplied through the read-only `ExactModelAdapter`; an exact `Dyna.Models.IWorldModel` can be passed directly. Reusing a solver continues from its current values and random stream. Construct a fresh solver with the same seed to repeat the same trial sequence.

For a single exact backup outside a trial, `OceanStats.RL.DP.BellmanOptimality.backup(model, stateIndex, values, discount)` returns the backed-up value, selected action, per-action values, and transition distributions. Its `actionValue(outcomes, values, discount)` computes one action's full expected return. Terminal successors contribute their reward with zero continuation.

See the [RTDP build contract](../../Planning/RL/Planning%20and%20Learning%20with%20Tabular%20method/RTDP_Build_Contract.md) for scope and verification requirements. LRTDP solved labeling is a separate phase.
