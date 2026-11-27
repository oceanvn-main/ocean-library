# Reinforcement learning workspace

This folder contains experiment entry points. Python implementations are grouped
in `Python/ocean_rl`; MATLAB implementations already live in `+OceanStats/+RL`.

```text
RL/
  README.md
  exp_*.m                      MATLAB experiment entry points
  Python/
    ocean_rl/                  Reusable Python implementation package
      contracts.py             Environment and agent interfaces
      registry.py              Register/select learning algorithms
      algorithms/monte_carlo/  Off-policy MC implementation and metadata
      environments/            State transitions, dynamics and rewards
      maps/                    Map data/loading and Tiled adapter
      evaluation/              Policy rollouts and evaluation metrics
      visualization/           Optional plots
      experiments/             Racetrack experiment orchestration
      tools/                   Image-map preprocessing
      paths.py                 Stable data/output locations
    configs/                   Algorithm parameter files
    maps/                      User map assets, images and editable Tiled files
    results/                   Generated reports, checkpoints and figures
    tests/                     Regression tests
    exp_multi_map_racetrack.py  Existing CLI, forwarding into the package
    offpolicy_mc_racetrack.py   Existing import, forwarding into the package
    racetrack_env.py           Existing import, forwarding into the package
    racetrack_maps.py           Existing import, forwarding into the package
    tiled_maps.py              Existing import, forwarding into the package
    import_dot_map.py          Existing CLI, forwarding into the package
    diag_trajectory.py         Existing checkpoint trajectory inspection tool
  slprj/                       Generated MATLAB/Simulink build artifacts
```

`Python/maps/` contains data; `Python/ocean_rl/maps/` contains map-loading code.
Keep learned results out of reusable modules. The small legacy files contain
imports and launch calls so existing commands and scripts continue to work.

## MATLAB experiment index

| Topic | Experiments | Reusable implementation |
|---|---|---|
| Dynamic programming | `exp_gamblers_problem.m`, `exp_jack_car_rental.m` | `+OceanStats/+RL/+DP` |
| Blackjack Monte Carlo | `exp_ch5_01_blackjack_mc_prediction.m`, `exp_ch5_02_off_policy_bj_value_estimation.m` | `+OceanStats/+RL/+MonteCarlo` |
| Toolbox comparison | `exp_ch5_03_matlab_rl_toolbox_comparison.m` | Experiment-specific toolbox integration |
| Importance sampling | `exp_discounting_aware_is.m`, `exp_discounting_aware_control.m` | `+MonteCarlo/+Weighting` and `+Strategies` |
| Racetrack control | `exp_racetrack_offpolicy_mc.m` | `+OceanStats/+RL/+MonteCarlo` |

The MATLAB scripts keep their existing locations. `slprj` and Python
`__pycache__` directories are generated caches and are not implementation homes.

## Adding future work

- A new learning algorithm belongs in `ocean_rl/algorithms/<family>/`.
- A new world or physical system belongs in `ocean_rl/environments/`.
- A new map format belongs in `ocean_rl/maps/`.
- A new study belongs in `ocean_rl/experiments/` with a small launcher if needed.

See [Python usage and extension instructions](Python/README.md). Only off-policy
MC is currently implemented in the Python registry. Q-learning, SARSA, DQN and
PPO are future implementations, not aliases for the existing MC algorithm.
