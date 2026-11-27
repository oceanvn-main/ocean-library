# PAC exploration: R-max and E³

`OceanStats.RL.PAC` holds two model-based exploration algorithms with polynomial (PAC) guarantees. Both learn a tabular count model and decide which unknown parts of the MDP to visit.

| Agent | Idea | Source |
|---|---|---|
| `RMaxAgent` | Unknown pairs are worth an upper bound U = Rmax/(1-γ). Act greedily on the optimistic model, so exploration follows from optimism. | Brafman & Tennenholtz 2002; pseudo-code from Strehl, Li & Littman 2009, Algorithm 1 |
| `E3Agent` | Balanced wandering in unknown states. In known states, compute exploitation and exploration policies explicitly, and explore while escaping to an unknown state is likely enough. | Kearns & Singh 2002, Sections 5.5 and 6 |

Both expect a tabular environment whose observation is the state index (`OceanStats.RL.Env.TabularEnvironment` family).

## R-max on the Chain

```matlab
env = OceanStats.RL.Env.Classic.ChainEnv();            % Strens 2000: 5 states, slip 0.2, rewards 2 / 10
ag  = OceanStats.RL.PAC.RMaxAgent(5, 2, 'KnownCount', 20, 'MaxReward', 10, 'DiscountFactor', 0.95);
tr  = ag.run(env, 'Steps', 10000, 'Seed', 1);           % Rewards, States, Actions, Modes ("known"/"optimistic")
disp(ag.greedyPolicy()')                                % [1 1 1 1 1]: walk to the end
disp(ag.isKnown())                                      % pairs tried m times
```

- A pair is **known** after `KnownCount` (m) tries. Only its first m samples enter R̂ and T̂ (`ag.Counts.N` never exceeds m; `ag.Counts.Seen` counts all tries).
- Every time a pair becomes known, value iteration runs `ceil(ln(1/(ε1(1-γ)))/(1-γ))` sweeps (`Epsilon1`, default 0.01).
- `Optimism` replaces U = Rmax/(1-γ) with a scalar or an S x A heuristic upper bound.
- A reward above `MaxReward` raises an error, because the optimism would no longer be an upper bound.

When `MaxReward` equals the true maximum reward, a pair that already earns it ties every unknown pair, and R-max stops exploring there. That behaviour is still optimal from that state, but parts of the MDP stay unknown. A slightly larger bound keeps the optimism strict.

## E³ on the Chain

```matlab
ag = OceanStats.RL.PAC.E3Agent(5, 2, 'KnownCount', 20, 'MaxReward', 10, 'DiscountFactor', 0.95, 'Epsilon', 0.1);
tr = ag.run(env, 'Steps', 10000, 'Seed', 1);
disp(tr.Modes(1:20))            % "wander" | "explore" | "exploit"
disp(ag.Known')                 % known states: every action tried m times (Definition 8)
disp(ag.escapeProbability()')   % max probability of leaving the known set within Horizon steps
disp(ag.exploitationPolicy()')
disp(ag.Decisions(end-4:end,:)) % [step, state, 1 = explore / 0 = exploit]
```

- `Horizon` defaults to the ε-horizon time T = ceil((1/(1-γ)) ln(Rmax/(ε(1-γ)))) (Kearns & Singh, Lemma 2).
- `EscapeThreshold` defaults to ε/(2·T·Rmax), the bound of the Explore-or-Exploit Lemma.
- `Strategy` options:
  - `"exploration_biased"` (default, Section 6): explore whenever escape is likely enough.
  - `"schedule"`: explore with probability α(t) = 1/t.
- On the Chain, the 1/t schedule gets stuck in the "go back" loop. Exploitation values the unknown part of the chain pessimistically (`MinReward`/(1-γ)), and 1/t explores too rarely to escape. The exploration-biased rule does not get stuck.
- Rewards must lie in [`MinReward`, `MaxReward`]; Kearns & Singh assume [0, Rmax].

## Known-state planning on its own

```matlab
[R, P, Pend] = ag.Counts.estimates();
plans = OceanStats.RL.PAC.knownStatePlans(R, P, Pend, ag.Known, true(5, 2), 0.95, ag.Horizon);
plans.ExploitPolicy, plans.Escape, plans.ExplorePolicy(:, end)
```

## Experiment

`Experiments/Scripts/RL/exp_pac_exploration_chain.m` (`QuickMode = true` for a short run) compares cumulative reward on the Chain for:

- R-max;
- E³ with both strategies;
- an ε-greedy Q-learning baseline;
- the exact optimal and always-go-back reward rates, as reference lines.
