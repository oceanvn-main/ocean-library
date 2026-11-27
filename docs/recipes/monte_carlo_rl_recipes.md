# Monte Carlo RL Recipes

Copy-paste recipes for the `OceanStats.RL` Monte Carlo framework (Ch.5).

---

## Recipe 1 — Create Blackjack Environment

```matlab
env = OceanStats.RL.Env.Classic.BlackjackEnv();
fprintf('States: %d | Actions: %d\n', env.get_num_states(), env.get_num_actions());
% → States: 200 | Actions: 2

% Inspect state layout
all_s = env.get_all_states();  % (200 x 3): [player_sum, dealer_showing, usable_ace]
disp(all_s(1:5, :));

% Roll out one random episode manually
state = env.reset();
done  = false;
while ~done
    action             = randi(2) - 1;   % 0=stick, 1=hit (random)
    [state, r, done]   = env.step(action);
    fprintf('  Action: %d → Sum: %d | Reward: %d | Done: %d\n', action, state(1), r, done);
end
```

---

## Recipe 2 — First-Visit MC Prediction (Simple Policy)

```matlab
env = OceanStats.RL.Env.Classic.BlackjackEnv();

% Target policy: hit if sum < 20, stick otherwise
target_pi = @(state, env_) double(state(1) < 20);

solver = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
    'Algorithm',    'FirstVisitPrediction', ...
    'NumEpisodes',  500000,                ...
    'TargetPolicy', target_pi              ...
);
[V, ~, ~, history] = solver.run();

% Inspect V at a specific state [sum=20, dealer=5, ace=0]
s_idx = env.state_to_index([20, 5, 0]);
fprintf('V([20,5,0]) = %.4f\n', V(s_idx));
```

---

## Recipe 3 — MC Exploring Starts Control (Optimal Policy)

```matlab
env    = OceanStats.RL.Env.Classic.BlackjackEnv();
solver = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
    'Algorithm',   'ExploringStarts', ...
    'NumEpisodes',  5000000,          ...
    'CheckInterval', 100000,          ...
    'Verbose',       true             ...
);
[V_star, Q_star, pi_star, history] = solver.run();

% Check optimal action for sum=18, dealer=6, no ace
s_idx = env.state_to_index([18, 6, 0]);
fprintf('Optimal action: %d (1=stick, 2=hit)\n', pi_star(s_idx));
```

---

## Recipe 4 — On-Policy Epsilon-Greedy Control

```matlab
env    = OceanStats.RL.Env.Classic.BlackjackEnv();
solver = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
    'Algorithm',   'OnPolicyControl', ...
    'Epsilon',      0.05,             ...
    'NumEpisodes',  1000000           ...
);
[V, Q, pi, history] = solver.run();
```

---

## Recipe 5 — Off-Policy Prediction (Weighted IS)

```matlab
env = OceanStats.RL.Env.Classic.BlackjackEnv();

% Target pi: always stick (action 0 = action_idx 1)
target_pi  = @(s_idx, a_idx) double(a_idx == 1);
% Behavior b: uniform random
behavior_b = @(s_idx, a_idx) 0.5;

solver = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
    'Algorithm',      'OffPolicyPrediction', ...
    'TargetPolicy',    target_pi,            ...
    'BehaviorPolicy',  behavior_b,           ...
    'NumEpisodes',     500000                ...
);
[V, ~, ~, history] = solver.run();
```

---

## Recipe 6 — Off-Policy Control (WIS)

```matlab
env = OceanStats.RL.Env.Classic.BlackjackEnv();

% Behavior policy: soft random (covers both actions equally)
behavior_b = @(s_idx, a_idx) 0.5;

solver = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
    'Algorithm',     'OffPolicyControl', ...
    'BehaviorPolicy', behavior_b,        ...
    'NumEpisodes',    2000000            ...
);
[V_star, Q_star, pi_star, history] = solver.run();
```

---

## Recipe 7 — Visualise with OceanViz

```matlab
% After running any solver:
viz_data = struct( ...
    'VTable',      V,                    ...
    'QTable',      Q,                    ...
    'PolicyTable', pi,                   ...
    'AllStates',   env.get_all_states(), ...
    'History',     history               ...
);

viz  = OceanViz.Core('Theme', 'Dark');
strat = OceanViz.Strategies.MonteCarloRLStrategy();

% 4-panel dashboard (creates its own figure)
strat.draw([], viz_data, struct('Mode', 'Combined'), viz.Colors);

% Individual panels on existing axes
fig = figure('Color', viz.Colors.Background, 'Position', [50,50,1400,450]);
ax1 = subplot(1,3,1);
ax2 = subplot(1,3,2);
ax3 = subplot(1,3,3);

strat.draw(ax1, viz_data, struct('Mode','ValueSurface3D', 'UsableAce', 0), viz.Colors);
strat.draw(ax2, viz_data, struct('Mode','PolicyHeatmap',  'UsableAce', 1), viz.Colors);
strat.draw(ax3, viz_data, struct('Mode','ConvergenceCurve'),               viz.Colors);
```

---

## Recipe 8 — Implement a Custom Episode Environment

```matlab
% Skeleton for a new IEpisodeEnvironment (e.g., GridWorldEnv)
classdef MyEnv < OceanStats.RL.MonteCarlo.IEpisodeEnvironment
    % ...
    properties
        % Define state, grid dimensions, etc.
    end
    methods
        function state = reset(obj)         % return start state
        end
        function [s,r,d] = step(obj, a)    % apply action
        end
        function acts = get_valid_actions(obj, state)
        end
        function all = get_all_states(obj)
        end
        function idx = state_to_index(obj, state)
        end
        function i = action_to_index(obj, action)
        end
        function n = get_num_actions(obj)
        end
        function tf = is_terminal(obj, state)
        end
    end
end
```

---

## Recipe 9 — Racetrack Off-Policy MC Control (Exercise 5.12)

```matlab
% Create Racetrack environment (Track 1 or Track 2)
env = OceanStats.RL.Env.Classic.RacetrackEnv('Track1', 'FailProb', 0.1);

% Configure Off-Policy MC Control with Epsilon-Greedy Behavior Policy
solver = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
    'Algorithm',     'OffPolicyControl', ...
    'NumEpisodes',   150000,             ...
    'Epsilon',       0.1,                ...
    'InitialQValue', -1000.0,            ... % Pessimistic init for negative return MDP
    'Verbose',       true                ...
);
[V_star, Q_star, pi_greedy, history] = solver.run();

% Evaluate rollout from a starting position with noise disabled
env.set_fail_prob(0.0);
env.reset();
greedy_policy_fn = @(s, e) e.index_to_action(pi_greedy(e.state_to_index(s)));
ep = env.generate_episode(greedy_policy_fn, 'ForceStartState', [env.StartCells(1,:), 0, 0]);
fprintf('Finished from start cell in %d steps!\n', ep.Length);
```

---

## Recipe 10 — Discounting-Aware Importance Sampling (Section 5.8)

```matlab
env = OceanStats.RL.Env.Classic.BlackjackEnv();

target_pi  = @(s_idx, a_idx) double(a_idx == 1);   % Always stick
behavior_b = @(s_idx, a_idx) 0.5;                  % Uniform random

% Discounting-Aware Off-Policy Prediction (gamma = 0.9)
solver = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
    'Algorithm',        'OffPolicyPrediction', ...
    'Gamma',            0.9,                   ...
    'DiscountingAware', true,                  ... % Enables Section 5.8 flat partial returns
    'ISType',           'Weighted',            ... % Eq 5.10 (or 'Ordinary' for Eq 5.9)
    'TargetPolicy',     target_pi,             ...
    'BehaviorPolicy',   behavior_b,            ...
    'NumEpisodes',      100000                 ...
);
[V, ~, ~, history] = solver.run();
```

---

## Recipe 11 — Discounting-Aware Off-Policy MC Control (Exercise 5.14)

```matlab
% Exercise 5.14: Off-Policy Control with Truncated Weighted Estimator for Q(s,a)
env = OceanStats.RL.Env.Classic.RacetrackEnv('TrackId', 1, 'NoiseProb', 0.1);

solver = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
    'Algorithm',        'DiscountingAwareControl', ...
    'Gamma',            0.95,                      ...
    'Epsilon',          0.10,                      ...
    'NumEpisodes',      50000,                     ...
    'Verbose',          true                       ...
);
[V_star, Q_star, pi_star, history] = solver.run();
```

---

## See Also

- [Monte Carlo RL Guide](../guides/monte_carlo_rl_guide.md)
- [Tabular RL Recipes (DP)](tabular_rl_recipes.md)


