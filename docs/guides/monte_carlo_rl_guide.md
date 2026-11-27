# Monte Carlo Reinforcement Learning Guide

## Overview

The `OceanStats.RL.MonteCarlo` package extends its Chapter 4 Dynamic Programming (DP)
foundation with a full **Monte Carlo (MC)** framework for Chapter 5 of
Sutton & Barto (2018). Unlike DP, MC methods require **no model of the
environment** — they learn purely from sampled episodes of experience.

### Key Distinction: DP vs MC

| Property | Dynamic Programming (Ch.4) | Monte Carlo (Ch.5) |
|---|---|---|
| Requires model? | Yes — full P(s'\|s,a) | No — only step/reset |
| Learns from | Analytical Bellman backups | Sampled episode returns |
| Interface | `TabularEnvironment` | `IEpisodeEnvironment` |
| Solver | `PolicyIterationSolver` / `ValueIterationSolver` | `MonteCarloSolver` |
| Environment | `ModifiedCarRentalEnv` | `BlackjackEnv` |

### Speedup Techniques (Modern Implementation)

1. **Incremental mean update** (S&B §5.6): `V(s) += (1/N) * (G - V(s))` — eliminates return-list storage (O(1) memory per state vs O(N_episodes))
2. **Weighted IS incremental** (S&B §5.6 Eq 5.8): `C(s,a) += W; Q += (W/C)*(G-Q)` — no full-episode IS product recomputation
3. **O(1) state indexing**: `BlackjackEnv` uses a pre-computed 3D lookup table (`StateIndexLUT`)
4. **Pre-allocated buffers**: Episode arrays allocated once per solver run, trimmed at end

---

## Classes

### `IEpisodeEnvironment` (Abstract)

Abstract base class for episodic, sample-based environments. Concrete subclasses
must implement:

| Method | Signature | Description |
|---|---|---|
| `reset` | `() → state` | Begin new episode, return initial state |
| `step` | `(action) → [next_state, reward, done]` | Take one step |
| `get_valid_actions` | `(state) → actions` | Valid actions for state |
| `get_all_states` | `() → (N x D)` | Full state space matrix |
| `state_to_index` | `(state) → idx` | O(1) index lookup |
| `action_to_index` | `(action) → idx` | 1-based action column |
| `get_num_actions` | `() → n` | Action space size |
| `is_terminal` | `(state) → tf` | Terminal state check |

The interface also provides a concrete `generate_episode(policy_fn)` method
with pre-allocated buffers that all six MC algorithms use.

### `BlackjackEnv`

Canonical Chapter 5 environment. State = `[player_sum, dealer_showing, usable_ace]`.

- **200 states** total: 10 (sums 12-21) × 10 (dealer 1-10) × 2 (ace flag)
- **2 actions**: 0 = stick, 1 = hit
- **Infinite deck**: cards sampled uniformly with replacement
- **Dealer policy**: deterministic — hits until sum ≥ 17
- **Reward**: +1 (win), 0 (draw), −1 (loss) at terminal step; 0 otherwise

### `RacetrackEnv`

Sutton & Barto Exercise 5.12 environment. State = `[x, y, vx, vy]`.

- **State space**: 4D continuous grid position and 2D velocity components $v_x, v_y \in \{0,1,2,3,4\}$, $(v_x, v_y) \neq (0,0)$.
- **9 actions**: $(a_x, a_y) \in \{-1,0,+1\}^2$.
- **Sub-pixel Ray-Grid Collision Detection**: Pre-computed ray traversal checks boundary collision vs finish line crossing.
- **Noise / Failure**: $P(\text{fail}) = 0.1$ (10% chance acceleration fails, velocity updates by zero).
- **Reward**: $-1$ per step until crossing finish line. Crash resets position to starting line with zero velocity.


### `MonteCarloSolver`

Unified solver Context class delegating to modular strategy algorithms (`+Strategies/`) and importance sampling weighting schemes (`+Weighting/`).

```matlab
solver = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
    'Algorithm',        'OffPolicyPrediction', ...
    'Gamma',            0.9,                   ...
    'DiscountingAware', true,                  ... % Sutton & Barto Section 5.8
    'NumEpisodes',      500000,                ...
    'TargetPolicy',     target_pi,             ...
    'BehaviorPolicy',   behavior_b,            ...
    'Verbose',          true                   ...
);
[V, Q, pi, history] = solver.run();
```

---

## Section 5.8: Discounting-Aware Importance Sampling

Standard off-policy importance sampling scales returns $G_t$ as unitary wholes using full product ratios $\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}$. When episodes are long ($T \gg 1$) and $\gamma < 1$, factors $k \gg t$ introduce massive extraneous variance without changing the expected update.

Section 5.8 treats discounting $\gamma$ as a degree of partial termination, decomposing return $G_t$ into flat partial returns $\bar{G}_{t:h} = \sum_{k=t+1}^{h} R_k$ bounded by horizon $h$:

$$G_t = (1 - \gamma) \sum_{h=t+1}^{T-1} \gamma^{h-t-1} \bar{G}_{t:h} + \gamma^{T-t-1} \bar{G}_{t:T}$$

### Weighting Schemes (`+OceanStats/+RL/+MonteCarlo/+Weighting/`)

- **`OrdinaryISWeighting`** (Eq 5.5): Unweighted empirical mean of full returns.
- **`WeightedISWeighting`** (Eq 5.6): Bounded-variance normalized weighted average of full returns.
- **`DiscountingAwareOrdinaryISWeighting`** (Eq 5.9): Unweighted mean of flat partial returns.
- **`DiscountingAwareWeightedISWeighting`** (Eq 5.10): Normalized weighted average of flat partial returns.


---

## Usage

### 1. First-Visit MC Prediction

Estimate V(s) under a fixed policy (e.g., "stick on 20-21"):

```matlab
env    = OceanStats.RL.Env.Classic.BlackjackEnv();
all_states = env.get_all_states();

% Fixed target policy: stick if player_sum >= 20, else hit
target_pi = @(state, env_) double(state(1) < 20);   % 1=hit if sum < 20

solver = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
    'Algorithm',    'FirstVisitPrediction', ...
    'NumEpisodes',  500000,                ...
    'TargetPolicy', target_pi              ...
);
[V, ~, ~, history] = solver.run();
```

### 2. MC Control with Exploring Starts

Find optimal policy (replicates S&B Fig 5.2):

```matlab
env    = OceanStats.RL.Env.Classic.BlackjackEnv();
solver = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
    'Algorithm',   'ExploringStarts', ...
    'NumEpisodes', 5000000            ...
);
[V_star, Q_star, pi_star, history] = solver.run();
```

### 3. On-Policy Epsilon-Greedy Control

```matlab
solver = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
    'Algorithm',   'OnPolicyControl', ...
    'Epsilon',      0.1,              ...
    'NumEpisodes',  500000            ...
);
[V, Q, pi, history] = solver.run();
```

### 4. Off-Policy Prediction with Weighted IS

```matlab
% Target: stick on 20-21 (deterministic)
target_pi = @(s_idx, a_idx) double(a_idx == 1);   % always stick (action_idx=1)

% Behavior: uniform random (0.5, 0.5)
behavior_b = @(s_idx, a_idx) 0.5;

solver = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
    'Algorithm',      'OffPolicyPrediction', ...
    'TargetPolicy',    target_pi,            ...
    'BehaviorPolicy',  behavior_b,           ...
    'NumEpisodes',     1000000               ...
);
[V, ~, ~, history] = solver.run();
```

### 5. Off-Policy Control

```matlab
behavior_b = @(s_idx, a_idx) 0.5;   % Uniform random (covers all actions)

solver = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
    'Algorithm',     'OffPolicyControl', ...
    'BehaviorPolicy', behavior_b,        ...
    'NumEpisodes',    1000000            ...
);
[V_star, Q_star, pi_star, history] = solver.run();
```

### 6. Discounting-Aware Off-Policy MC Control (Exercise 5.14)

```matlab
solver = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
    'Algorithm',        'DiscountingAwareControl', ...
    'Gamma',            0.95,                      ...
    'Epsilon',          0.10,                      ...
    'NumEpisodes',      50000                      ...
);
[V_star, Q_star, pi_star, history] = solver.run();
```

---

## Visualization

```matlab
% Prepare data struct for MonteCarloRLStrategy
viz_data = struct( ...
    'VTable',      V,                          ...
    'QTable',      Q,                          ...
    'PolicyTable', pi,                         ...
    'AllStates',   env.get_all_states(),       ...
    'History',     history                     ...
);

viz  = OceanViz.Core('Theme', 'Dark');
fig  = figure('Color', viz.Colors.Background, 'Position', [50,50,1400,820]);
ax   = subplot(1,1,1);

% 4-panel dashboard
viz.plot(ax, 'MonteCarloRLStrategy', viz_data, struct('Mode', 'Combined'));

% Individual modes
viz.plot(ax, 'MonteCarloRLStrategy', viz_data, struct('Mode', 'ValueSurface3D',  'UsableAce', 0));
viz.plot(ax, 'MonteCarloRLStrategy', viz_data, struct('Mode', 'PolicyHeatmap',   'UsableAce', 1));
viz.plot(ax, 'MonteCarloRLStrategy', viz_data, struct('Mode', 'ConvergenceCurve'));
```

---

## See Also

- [Monte Carlo RL Recipes](../recipes/monte_carlo_rl_recipes.md)
- [Tabular RL Guide (DP)](tabular_rl_guide.md)
- [Architecture Reference](../../.agents/architecture.md)
