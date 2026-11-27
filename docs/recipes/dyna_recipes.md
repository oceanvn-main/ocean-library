# Tabular Dyna recipes

## Dyna-Q on a finite environment

```matlab
env = OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner', 19);
trace = OceanStats.RL.Dyna.Observers.Recorder();
learner = OceanStats.RL.Dyna.Factory.dynaQ(env, ...
    'Alpha', 0.1, 'Epsilon', 0.1, 'Gamma', 1, ...
    'PlanningSteps', 10, 'Seed', 7, 'Observer', trace);
result = learner.run('NumEpisodes', 50, 'MaxStepsPerEpisode', 1000);
```

`result.Q` is the learned action-value table. `trace.UpdateEvents` distinguishes real and planning updates and retains the target, TD error, step size, and before/after value for every mutation.

For a changing environment, construct the textbook Dyna-Q+ composition with
`OceanStats.RL.Dyna.Factory.dynaQPlus`. It gives every legal action at an encountered
state a default self-loop model and applies the recency bonus only to planning rewards.

## Replace uniform planning with prioritized sweeping

```matlab
env = OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner', 19);
mask = OceanStats.RL.Dyna.Models.ExactModelAdapter.deriveMask(env);
model = OceanStats.RL.Dyna.Models.DeterministicTabularModel(21, 2, mask);
planner = OceanStats.RL.Dyna.Planning.PrioritizedSweeping(1e-6);
learner = OceanStats.RL.Dyna.DynaLearner(env, ...
    'Model', model, 'Planning', planner, 'PlanningSteps', 10, 'Gamma', 1);
result = learner.run('NumEpisodes', 50);
```

Use `planner.inspect()` to inspect queue activity. True termination suppresses bootstrapping; external time limits remain truncations and retain the final-state bootstrap.
