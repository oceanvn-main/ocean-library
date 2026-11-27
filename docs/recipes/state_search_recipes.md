# State-space search and decision-time planning

`OceanAlgo.StateSearch` holds general search engines that plan on a model, never on a live environment. RL uses them through `OceanStats.RL.DecisionTime`. Plan: `Planning/System/Search/State_Search_Build_Plan.md`.

| Engine | Model it needs | Source |
|---|---|---|
| `AStarSearch` (A*, weighted A*) | deterministic `IDistributionModel`, reward <= 0 | Hart, Nilsson & Raphael 1968; Pohl 1970 |
| `IDAStarSearch` | deterministic, reward <= 0 | Korf 1985 |
| `BeamSearch` (and diverse beam) | deterministic | Russell & Norvig ch. 4; Vijayakumar et al. 2018 |
| `DepthLimitedSearch` | `IDistributionModel` | Sutton & Barto §8.9 |
| `RolloutSearch` | `ISampleModel` (a distribution model is sampled automatically) | Sutton & Barto §8.10 |
| `MonteCarloTreeSearch` | `ISampleModel` | Sutton & Barto §8.11; Kocsis & Szepesvári 2006 |

Every engine returns an `OceanAlgo.StateSearch.SearchResult`: `Action`, `Value`, `Actions`, `ActionValues`, `ActionVisits`, `Path`, `Status` (`ok`, `budget_exhausted`, `no_path`, `no_legal_action`, `at_goal`), `Work` and an optional `Trace`.

## Plan on an RL environment

```matlab
env = OceanStats.RL.Env.Classic.WindyGridworldEnv();
pm  = OceanStats.RL.DecisionTime.WorldModelSearchModel(env);     % exact model -> planning model
start = env.idx(4, 1);

hCost = @(s) abs(8 - (mod(s - 1, env.Cols) + 1));                % column distance: admissible, consistent
r = OceanAlgo.StateSearch.AStarSearch('Heuristic', hCost).search(pm, start);
disp(r.Value)              % -15: the optimal return
disp(r.Path.Actions)       % the 15 actions
disp(r.Work)               % Expansions, Generated, Reopened, MaxFrontier, DistinctStates
```

`WorldModelSearchModel` also accepts a Dyna `IWorldModel`, including a learned one. A learned model only knows observed pairs, so planning on an unobserved pair raises the model's own error. Exact (read-only) models are cached per (s, a) for speed; the draws are identical.

## Act with decision-time planning (S&B 8.8)

```matlab
env = OceanStats.RL.Env.Classic.WindyGridworldEnv('KingMoves', true, 'StochasticWind', true);
pm  = OceanStats.RL.DecisionTime.WorldModelSearchModel(env);
leaf = @(s) -abs(8 - (mod(s - 1, env.Cols) + 1));                % imperfect value estimate

mcts  = OceanAlgo.StateSearch.MonteCarloTreeSearch('NumIterations', 200, 'ExplorationConstant', 5, ...
    'MaxDepth', 30, 'LeafValue', leaf, 'Seed', 1);
agent = OceanStats.RL.DecisionTime.DecisionTimeAgent(mcts, pm);   % plan, act once, replan
ep = agent.runEpisode(env, 'Seed', 1, 'MaxSteps', 100);
disp([ep.Return, ep.Steps]); disp(ep.Ended)
```

The agent calls `advance` on MCTS after each real step, so the subtree below the real successor is reused (`ReuseTree`, default true).

## Depth-limited heuristic search (S&B 8.9)

```matlab
dls = OceanAlgo.StateSearch.DepthLimitedSearch('Depth', 5, 'LeafValue', leaf, 'RecordBackups', true);
r = dls.search(pm, start);
disp(r.ActionValues)       % q_5(start, a) for every legal action
disp(r.Trace)              % one-step backups in depth-first order (Figure 8.9)
```

With an exact model, `q_D` equals the root row after `D` value-iteration sweeps from `LeafValue`. Pass `'MaxExpansions', n` to search with iterative deepening and return the deepest completed depth.

The horizon effect is real. With `LeafValue = 0` and reward -1 per step, a path cut off at depth `D` scores the same as one that reaches the goal in `D` steps. On Windy Gridworld, `Depth = 15` from the start ties every action, and the agent loops forever. `Depth = 16` is optimal.

## Rollout algorithm (S&B 8.10)

```matlab
base = OceanStats.RL.DecisionTime.TableRolloutPolicy(piVector);  % or a NumStates x NumActions table
ro = OceanAlgo.StateSearch.RolloutSearch('RolloutPolicy', base, 'NumRollouts', 50, 'Horizon', 50, ...
    'LeafValue', leaf, 'Seed', 1);
r = ro.search(pm, start);
disp([r.ActionValues; r.ActionVisits; r.Work.StandardErrors])
```

The greedy action improves on the rollout policy (policy improvement theorem). The default rollout policy is `OceanAlgo.StateSearch.UniformRandomRolloutPolicy`.

## MCTS: exploration constant and tree inspection

```matlab
mcts = OceanAlgo.StateSearch.MonteCarloTreeSearch('NumIterations', 500, 'ExplorationConstant', 5, ...
    'RecordIterations', true);
r = mcts.search(pm, start);
t = mcts.getTree();        % Key, Parent, ParentAction, Depth, Visits, per-node EdgeVisits / EdgeValues
r.Trace(1)                 % Nodes, Actions, Rewards, ExpandedNode, Tail, Return of iteration 1
```

`ExplorationConstant` depends on the scale of the returns. UCB1's `sqrt(2)` assumes returns in [0, 1]. Set c near the spread of returns. With returns spread over ±10 and c = 2, an action whose first random rollouts were poor stayed unexplored for 80,000 iterations; c = 5 found the optimum. `FinalSelection` is `"most_visited"` (default) or `"max_value"`.

## Write your own planning model

Implement `OceanAlgo.StateSearch.IDistributionModel` and/or `ISampleModel`:

```matlab
% actions(s) -> legal action labels;  key(s) -> numeric or string key;  isGoal(s) (optional)
% outcomes(s, a) -> struct array Next, Reward, Terminated, Probability
% sample(s, a, stream) -> struct Next, Reward, Terminated
```

States are opaque, so a vector or struct state works if `key` returns a unique scalar or string. Rewards are maximised. A* and IDA* use cost = -reward and reject positive rewards.

## Experiment

`Experiments/Scripts/RL/exp_ch8_decision_time_planning.m` (set `QuickMode = true` for a short run) compares:

- path search on deterministic Windy Gridworld: plan quality and expansions;
- decision-time agents on stochastic wind with king moves: return against model calls per decision, with V* and the random policy as references.
