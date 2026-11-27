# Imported racetrack experiments

## Overview

Reusable implementations now live in the `ocean_rl` package; the original
Python filenames are compatibility entry points. See the
[extension guide](../../Experiments/Scripts/RL/Python/README.md) for the package
layout, algorithm registration and current timeout semantics.

The Python environment separates map loading (`racetrack_maps.py`), dynamics
(`racetrack_env.py`), Monte Carlo control (`offpolicy_mc_racetrack.py`), and the
CLI experiment (`exp_multi_map_racetrack.py`). Each map/seed gets a fresh policy.
This is point-to-point navigation, not a vacuum coverage or vehicle steering model.

## Usage

Objectives are stateless strategies registered in `ocean_rl.objectives`.
Minimum time remains default. Geometric distance uses negative physical segment
length with gamma=1, counts final boundary contact and excludes reset teleportation.
The car dynamics and crash resets remain; this is not a collision-free planner.
See the Python README objective section for extension contracts and limitations.

The Python project now supports installation with `python -m pip install -e .`
from its directory. `ocean-rl --system racetrack` launches the experiment.
Wheel distributions include default maps; results default to the caller's
`./results`. See the Python CONTRIBUTING.md for checks and release instructions.

To draw maps visually, use the [Tiled workflow](../recipes/tiled_racetrack.md).
The runner accepts .tmj and Tiled JSON exports directly.

Run from the repository root:

```powershell
py -3.12 Experiments/Scripts/RL/Python/exp_multi_map_racetrack.py --maps Experiments/Scripts/RL/Python/maps/track_1.json --episodes 100 --seeds 0 1 --max-steps 100000
```

Install NumPy for training; Pillow for images; Gymnasium for declared spaces;
Matplotlib only for optional `--plot` (saved figures) or `--show`. The Python
renderer remains optional because OceanViz.Core is MATLAB-only; a future Python
adapter to OceanViz would allow reuse of its themes and rendering strategies.

`--eval-max-steps` separately bounds evaluation rollouts (default 1000).
`--maps` accepts multiple files; CLI movement, speed, failure probability and
step-limit overrides take precedence over each JSON map's environment settings.
Output JSON includes the actual grid, grid hash, configuration, training episode
statistics, evaluation trajectories and success/crash/truncation metrics. NPZ
files store sparse state rows and corresponding Q/log_C action vectors. These
are inspection checkpoints, not a resumable training API. Outputs with the same
map index, basename and seed in the same directory are replaced; choose a new
`--output-dir` to preserve a run.

## Examples

See [copy-paste recipes](../recipes/racetrack_maps.md) for importing your own map.

### Semantics and limits

Coordinates are zero-based `(row, column)` with row increasing downward. States
are `(row, column, vx, vy)`; positive vy moves upward. `max_velocity` is exclusive:
5 means component speeds 0 through 4 in textbook mode, -4 through 4 in
bidirectional mode. Actions are accelerations, not direct position increments.

The default textbook mode allows only upward/rightward motion. Bidirectional
mode permits all four directions with inertia. Maps must have explicit start
and finish cells. Finish cells are terminal zones, not oriented boundary edges.
Paths check every touched grid cell in order; simultaneous corner contact with
a wall takes precedence over finish contact. A crash resets to a random start
with zero velocity and the episode continues. Failed acceleration leaves the
previous velocity unchanged, including rest at the start. Terminal observations
remain on a finish cell. Reward is -1 on nonterminal transitions and 0 on finishing (the existing reward convention).

MC requires a complete return for an update. The current learner records
truncated episodes without updating them and continues training. If every
episode truncates, it raises `EpisodeTruncatedError` and the CLI writes failure
diagnostics with a nonzero exit status. With partial truncation, updates use
only completed episodes, which can bias estimates toward shorter trajectories.
Increase the step limit or choose a bootstrapping algorithm for difficult maps.
Completing a training budget does not establish convergence or optimality.

In the original verification, the bundled house and maze hit a 100,000-step first-episode
limit with seed 0. They import and pass movement reachability validation, but
random inertial exploration is expensive; these examples are not evidence of
successful learning. The textbook preset completed 100 training episodes for
each of seeds 0 and 1, with 10 of 14 starts finishing in noise-free evaluation
within 300 steps for each seed. These are smoke-run results, not convergence
guarantees.

Behavior is epsilon-greedy with uniform exploitation over tied maximizers. Its
actual probability is saved during rollout. The target breaks ties by action
order. Cumulative importance weights use logarithms to avoid overflow. Sparse
tables allocate only visited states and accept negative velocities without
array-index aliasing. Evaluation uses a separate seeded environment and does
not mutate training RNG or tables. Acceleration noise is off during evaluation,
but crash resets remain stochastic.

## See Also

- [Recipes](../recipes/racetrack_maps.md)
- [Architecture](../../.agents/architecture.md)
- `Experiments/Scripts/RL/Python/tests/test_environment.py`
- `Experiments/Scripts/RL/Python/tests/test_agent.py`

