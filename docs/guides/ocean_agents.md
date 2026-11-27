# OceanAgents — Agents, Minds and Runners

## Overview

`+OceanAgents` runs agents in environments while keeping every agent's memory private and every run reproducible. It is the first part of the game-theory agent system planned in `Planning/System/Game Theory/`. Milestones A1 (agent host, swappable mind, single-agent and multi-agent runners) and A2 (fixed and evidence-adaptive trust) are done. Plans and skills (A3) and enforceable commitments (A4) come later.

An **agent** is an `AgentHost`: an identity (`AgentID`), a **mind** (`IDecisionProcess`, e.g. `PolicyMind` wrapping one `IPolicy`), an optional **learner** (`ILearner`), a discount `Gamma` and a random stream. The agent never holds the environment. A **runner** is the only thing that calls `reset`/`step`. It passes each agent an `ActorView` of what it may see, gets back one `Decision`, advances the world once, and returns `Feedback`. The host turns that feedback into a private `Transition` for its learner.

```text
environment ──ActorView──► AgentHost ──► mind ──► policy ──Decision──► runner ──► environment.step (once)
     ▲                          │                                                     │
     └──────────── runner ◄─────┴──── Feedback ──► TransitionBuilder ──► learner (after the step)
```

| Package | Contents |
|---|---|
| `OceanAgents.Core` | Value records (`ActorView`, `Decision`, `Feedback`, `Transition`, `DecisionFrame`, `ActionBatch`, `EnvSpec`, `CapabilitySet`), interfaces (`IPolicy`, `ILearner`, `IDecisionProcess`, `IJointEnvironment`, `ISingleAgentEnvironment`), `assertValueData` |
| `OceanAgents.Env` | `NativeSingleAgentBridge` (any existing `OceanStats.RL.Env.IEnvironment`), `Reference.RepeatedMatrixGameEnv` (any `NormalFormGame`, simultaneous play) |
| `OceanAgents.Policies` | `RandomLegalPolicy`, `TablePolicy` (acts from a saved table), `EpsilonGreedyQPolicy` |
| `OceanAgents.Learners` | `OneStepQLearner` (Q-learning over the agent's own `TabularQ`) |
| `OceanAgents.Minds` | `PolicyMind` (the reactive profile: one policy, nothing else) |
| `OceanAgents.Artifacts` | `ActionCodec`, `DiscreteStateCodec`, `PolicyArtifact` (saved policy with a declared label/column convention) |
| `OceanAgents.Runtime` | `AgentHost`, `SingleAgentRunner`, `JointRunner`, `StreamRegistry`, `TransitionBuilder`, `RunRecord`, `IsolationChecker` |
| `OceanAgents.Beliefs` (A2) | `BeliefRecord`, `IBeliefStore`, `FixedBeliefStore` (imported scores, never fitted), `BetaTrustStore` (Beta–Bernoulli per subject and context) |
| `OceanAgents.Minds` (A2) | `TrustMind` (beliefs → choice), `TrustRules.ExpectedUtilityRule` (optional risk aversion), `TrustRules.ThompsonRule` |
| `OceanAgents.Env.Reference` (A2) | `DelegationEnv`: each round, delegate the task to a provider or take the outside option; outcomes become evidence |

Rules the code enforces:

- **Labels, not columns.** A `Decision` carries the environment's external action label. Tables are column-indexed, and `ActionCodec` maps between the two. A saved action table must declare `Representation` `"Label"` or `"ColumnIndex"`. For example, TD's `pi` stores labels and MC's `PolicyTable` stores columns, and they cannot be told apart from the numbers.
- **Private state.** Records hold value data only (no handles at any depth). A mind or learner belongs to one host, and a host binds to one runner at a time. At construction, runners refuse agents that share any handle with each other or with the environment (`Runtime:SharedComponent`).
- **One rollout owner.** Each decision causes exactly one environment step. Learning happens only after the step is committed. `LearningEnabled = false` changes no learned parameter.
- **Termination vs truncation.** `Terminated` means a true terminal (no bootstrap). `Truncated` (e.g. `TimeLimit`) keeps the final observation and passes successor legality in `NextView.Info.successor_action_mask`, so a learner can bootstrap.
- **Reproducible randomness.** `StreamRegistry` gives each purpose (`"env"`, `"agent:<id>:action"`) its own `mrg32k3a` stream, derived from the run seed and the purpose name. The global stream is never touched. Adding a purpose never shifts another, and the order in which agents are asked cannot change any draw.
- **Fail-stop.** If a callback or environment step throws, the run stops. `RunRecord.Status = "Failed"` records the stage and the `MException`, and the host and runner become `Invalid`. Nothing is retried.
- **Checks before the first reset.** Role, action count, label set and state range are checked when the runner is constructed. A host left at role `"default"` takes its slot's role.

## Beliefs and trust (A2)

- **Evidence** (`Core.EvidenceRecord`) reaches an agent only in its own `ActorView.Evidence`. Each record says who it is about (`SubjectID`), in which context, whether it was observed directly or reported, and the outcome (`success`, `failure` or `unknown`).
- **Stores** consume each evidence ID exactly once. The possible dispositions are `applied`, `recorded-only` (fixed store), `duplicate`, `unknown`, `report`, `ineligible` and `frozen`. The same ID with different content is a fault. Beliefs are keyed by subject, never by slot, and contexts are independent. Capacities are explicit and nothing is silently evicted.
- **Beta update**: success α+1, failure β+1. The estimate is α/(α+β) and the variance is αβ/((α+β)²(α+β+1)). The variance can *rise* after contradictory evidence.
- **Unknown stays unknown.** Obstruction is not failure, and reports are recorded but never applied in A2.
- **Profiles**: the same `TrustMind` is `fixed-belief-reactive` with a `FixedBeliefStore` and `evidence-trust-adaptive` with a `BetaTrustStore`. Rules: expected utility U = p̃·S + (1−p̃)·F with p̃ = p − λσ, deterministic lowest-label ties; or Thompson sampling, which has no closed-form likelihood and so records none.
- **Frozen-belief evaluation**: `store.setUpdatesEnabled(false)`. This is separate from `LearningEnabled`, which controls parameter learning.
- **Evidence IDs are scoped per world instance** (`DelegationEnv` `RunTag`). A trained store reused with a new world keeps learning instead of rejecting or dropping the new events.
- **Behaviour note**: with payoffs [10 −20 0] and a Beta(1,1) prior, a λ = 0 expected-utility mind never delegates on its own (prior utility −5 < 0). It learns only from evidence others make public. Thompson sampling explores.

## World kernel (W1)

`OceanAgents.World` is a generic, hierarchical world. It holds **rules and registered components**, not domain logic. Adding a new kind of thing means new files, never an edit to the kernel. Design: `Planning/System/Game Theory/World_Kernel_Design.md`. Contract: `W1_Kernel_Contract.md`.

- **Timeline ("GOD")** owns time only. It never inspects commands, and nothing polls the clock.
- **Doors** are guards *and* containers. Each `DoorSpec` declares:
  - the components and resources it alone writes (one writer per component);
  - its membership;
  - its rules (pure functions that return proposed writes, events, commands and predictions);
  - its axioms;
  - when it is driven.
  
  Doors can nest two levels deep.
- **Components are plain arrays**, one row per entity. They come in three kinds:
  - `value`: changed only by writes;
  - `linear`: Base + Rate·(now − T0);
  - `clock`: now − T0.
  
  Time-bound values move with the timeline at no cost. Predictors say when a discontinuity will happen, and the world **skips** instants where nothing is due. Skipping is proven identical to stepping, including runs where a monitored value crosses its bound.
- **Instants** run phase by phase: due items, then event delivery, then rules (reading the phase-start state), then one all-or-nothing synchronisation point. That point applies writes, then commands in tie-break order, then monitor axioms, predictors and events. An event is never seen in its own phase at the same instant. Any error restores the last completed state and fails the world (fail-stop).
- **The guard.** Admission checks types with physical dimensions (exponent algebra over L, M, T, E, C, N) and axioms, never names:
  - `static` axioms are checked at admission;
  - `monitor` axioms are checked at every synchronisation point;
  - modules are admitted by `probe` axioms and receive a `Certificate`;
  - `assumed` axioms are never checked, and certificates never report them as verified.
  
  Commands never change values: attach and detach are allowed only on the issuing door's own components, and an attach of a component the entity already has is refused.
- **Hidden data:**
  - hidden values and hidden presence never reach a view directly;
  - information derived from another door's hidden data flows only through rules that declare it in `Reveals`;
  - axioms and memberships may not name another door's hidden components;
  - `self` visibility is enforced at views only: a rule may copy a `self` value into a public component.
- **Limits in W1:**
  - no agent face yet (the Agency door and `IJointEnvironment` come in W2);
  - fixed-roster agents;
  - integer ticks only;
  - linear and clock time functions only;
  - fail-stop copies the state once per phase (≈ 7 ms per instant at 10⁴ entities);
  - admitting 10⁴ spawns takes ≈ 8 s.

## Grid worlds built from doors (W2)

The three native grid environments are rebuilt from doors, and proven to be the same environments.

| Piece | Role |
| --- | --- |
| `World.Doors.Agency` | Marks an entity as an agent (hidden `AgentID`); routes an action label to an event |
| `World.Doors.GridMotion` | Owns the cell; moves, walls, wind (one uniform per step, as natively) |
| `World.Doors.RewardTermination` | Step and cell rewards, goal, cliff reset, optional time limit (truncation) |
| `World.Doors.GridWorlds` | Factories `cliffWalking`, `randomWalk`, `windy` with the native options; also the exact model |
| `World.AgentWorld` | Presents a world as an `IJointEnvironment` (one `World.advance` per step) |
| `Env.WorldAsEnvironment` | Reverse bridge: a door-built grid world used as a native `IEnvironment` + `IModelEnvironment` |
| `World.Doors.RLDoor` | Guard for RL environments: module axioms RL-1…RL-9, checked by probing, never by class name |

Evidence (tests `OceanTest.OceanAgents.Env.World*ParityTest`): trajectories bit-identical to `env_golden.mat`; the existing `TDLearner` reproduces `td_replay_golden.json` byte for byte; value iteration returns the same values.

Limits: a bridge step costs about 6.5 ms against about 20 µs natively (the kernel snapshots state every phase), so use native environments for long training and door worlds when you need composition. Doors that reproduce a native environment use `'Stream', "legacy-mt19937ar"`; new doors keep the default keyed streams. Contract: `Planning/System/Game Theory/W2_Grid_Doors_Contract.md`.

## Mind, body, memory and signals (AB1)

An agent can be built from three parts that talk only through signals: a **mind** that decides and holds its learners, a **body** that turns commands into the world's actions and the world's view into senses, and **memories** the mind queries. The pairing of one mind with one body is a `Dualism`. Design: `Planning/System/Game Theory/Agent_Body_Signal_Design.md`; contract: `AB1_Signal_Body_Contract.md`.

| Piece | Role |
| --- | --- |
| `Minds.Signals.SignalType`, `Signal`, `Wave` | What a signal is (event, stream or wave; flag, number, vector, label or record; physical dimension; role). A wave is a formula of time whose threshold crossings are solved exactly |
| `Minds.Signals.Router`, `Port` | One per agent. Routes are fixed at join; the sender's type is negotiated once; every signal is stamped with the true sender; a flood at one instant stops the run |
| `Components.IComponent` | Verify, then act. A refused signal changes nothing; there are no setters. `ThresholdConverter` (with hysteresis) and `Sampler` turn waves into events and streams |
| `Memory.TableMemory`, `EpisodeMemory` | Anyone may query; only the one bound writer may write |
| `Body.EmptyBody`, `GridRobotBody` | An empty body forwards the command; a robot body has a motor and a position sense, and runs behaviors (`RepeatMove`) started and stopped by signals |
| `Minds.QSignalMind` | The A1 epsilon-greedy Q agent as a signal mind. `AgentHost` still decides when learning happens |
| `Embodied.Dualism` | Pairs a mind with a body when the body accepts every command the mind sends; a body swap happens between episodes, and the mind sees the new body through its signature |

Evidence (`OceanTest.OceanAgents.Integration.EmbodiedRegressionTest`): the cliff-walking Q-learning run through an empty body and through a robot body is identical to the PolicyMind run (returns, lengths, Q table, update counts). Cost: about 1.8 to 2.2 times the PolicyMind agent per step.

## Observations: the mind sees only through its body (AB2)

The body carries **sensors**; each turns what the world shows into a typed **observation** and sends it to the mind through the router. The mind keeps a **perception** for each kind of observation and decides **and learns** only from what it perceives: it never reads the world's view or action mask itself. Contract: `Planning/System/Game Theory/AB2_Observation_Contract.md`.

| Piece | Role |
| --- | --- |
| `Body.Sensors.DigitObservation` | A number or a vector the world shows (a cell index, four CartPole numbers) |
| `Body.Sensors.MovesObservation` | What the **Laws** (the environment's rules) permit now: flags named by the move labels |
| `Body.Sensors.BehaviorObservation` | Whether a body behavior is running (its own type, a flag named "running") |
| `Body.Sensors.VisionObservation` + `GridRenderer` | An image of the scene of a declared size (learning from images is not built yet) |
| `Minds.Perception.DigitPerception`, `MovesPerception`, `BehaviorPerception` | The mind's side: each reads one kind of observation |
| `Minds.PolicySignalMind` | Any existing policy as a mind that sees only through observations |
| `Env.Reference.TicTacToeEnv` | Two players taking turns; the Laws refuse the wrong player and occupied cells |

Rules:

- **Pairing is by type only.** Each perception must find exactly one sensor of a type it can read; none, or more than one, refuses the pairing. Two sensors of the same kind need different types, for example a physical dimension.
- **A frame is sensed once.** The decision and the learning after the step share the same observations.
- **Behaviors never talk to the mind.** The mind sees a running behavior through its observation and lets it continue. If the Laws forbid the behavior's next move, an interruptible behavior is stopped and a committed one raises `Body:BehaviorBlocked`.
- **Policy** keeps its reinforcement-learning meaning: the mind's own rule for choosing; the environment's rules are the **Laws**.
- **Learners belong to the mind.** Give them with `mind.addLearner(learner, id)`, never to the host; the mind builds its own experience from what it perceived and did, and the host only says when a step is committed.
- **Several policies.** `mind.addPolicy(name, policy)`; settings choose which acts (`mind.usePolicy(name)`), or any `Minds.Selection.IPolicySelector` (a learned one later). `mind.actedPolicy()` names the policy behind the last decision.

Evidence: the cliff-walking Q agent seeing only through a digit and a moves observation reproduces the original run exactly (`Integration.EmbodiedRegressionTest`); a moves-only agent plays legal, complete, reproducible tic-tac-toe (`Integration.TicTacToeSmokeTest`). Cost: about 3 to 4 times the PolicyMind agent per step.

## The world side as devices (DV1)

A world's rules can live in **devices**: components on the world side that verify, then act, and talk through signals, with no environment class and no door holding the rules. `Devices.DeviceWorld.cliffWalking()` is Sutton & Barto's cliff walking built this way. Contract: `Planning/System/Game Theory/DV1_Device_World_Contract.md`.

| Piece | Role |
| --- | --- |
| `Components.GridPosition` | The body's own cell. Anyone may ask where it is; it moves only on a request from the terrain device |
| `Devices.Terrain` | The ground: size, edges, cliff, start, and cells the Laws forbid entering. Moves the body's position |
| `Devices.Reward` | Pays (-1 a step, -100 for the cliff), ends the episode at the goal or at its time limit |
| `Devices.DeviceWorld` | Wires the devices on one router and presents the world as a native environment, so tests, solvers and runners use it unchanged |

Evidence: every recorded seed of `cliff`, `cliff3x5` and `cliff_limit20` is reproduced bit for bit, the model equals the recording and value iteration the native result (`Devices.DeviceWorldTest`); the AB2 agent runs identically on it, with its robot body's own position being the one the terrain moves (`Integration.DeviceAgentParityTest`). Cost: a few milliseconds a step, cheaper than the door-built world, far from the native class.

## Players, factored actions and fictitious play (PG1)

A **`Player`** is one participant: a mind, its body (an empty one when none is given), their pairing and the host that gates learning. Experiments create players and hand them to the runners. Observers (`Observe.IPlayerObserver`, e.g. `Observe.PlayerTrace`) can be attached for display or recording; they see value records only and can never change a run. Contract: `Planning/System/Game Theory/PG1_Parity_Guess_Contract.md`.

| Piece | Role |
| --- | --- |
| `Core.ActionFactors` | An action made of parts (e.g. put and guess); joint actions numbered with the first part fastest |
| `Policies.FactoredPolicy` | Each part decided by its own policy; the joint action is their combination |
| `Learners.FactorLearner` | A learner that learns about one part only |
| `OceanStats.GameTheory.Learning.EmpiricalBelief`, `FictitiousPlay` | Fictitious play (Brown 1951), generic: beliefs about every opponent, best or smooth (logit) response, any number of players |
| `OceanStats.GameTheory.Solvers.FictitiousPlaySolver` | Every player of a normal-form game plays fictitious play; exploitability per round (converges in zero-sum games) |
| `Memory.BeliefMemory`, `Learners.FictitiousPlayLearner`, `Policies.FictitiousPlayPolicy` | Fictitious play inside a player: the learner updates the beliefs from the other's revealed actions, the policy best-responds |
| `Env.Reference.ParityGuessEnv` | Two players put a number and guess the parity of the other's; one right guess wins, both right is a draw |

Evidence (`Integration.ParityGuessTest`): against a player who puts even 90 % of the time and guesses at random, a player who learns from the other wins about 0.47 of games (theory p2/(1+p2) with p2 = 0.9), against 1/3 without learning; fictitious play reaches the best response in fewer games than Q-learning.

## Usage

1. Wrap the environment: `NativeSingleAgentBridge(env, 'AgentID', id)` for an existing single-agent environment, or a joint environment such as `RepeatedMatrixGameEnv`. The bridge's `AgentID` must equal the host's.
2. Build the agent's components fresh for each agent: value table → policy (and learner) → `PolicyMind` → `AgentHost(id, mind, 'Learner', learner, 'Gamma', g)`. The learner's `Gamma` must equal the host's.
3. Build a runner with an explicit `'Seed'` and call `run(numEpisodes)`. Read the returned `RunRecord`: `Status`, `EpisodeReturns`, `EpisodeLengths`, `StepCalls`, `UpdateCounts`, `FailureStage`, `Failure`.
4. To reuse a host with another runner, call `runner.release()` first.

## Examples

See [OceanAgents recipes](../recipes/agent_recipes.md): Q-learning on CliffWalking, evaluation without learning, acting from a saved table, two agents in repeated matching pennies, isolation checks, a recorded failure, and (A2) a Beta trust store, imported scores, fixed vs adaptive trust in one delegation world, frozen-belief evaluation, and (W1) a draining battery world that jumps time and the guard refusing by axiom.

## Limits (A1)

- Discrete actions only (`Discrete` action spaces), numeric observations, unit-duration steps, action phase only (no message phases), simultaneous or single-agent timing (turn-based is later).
- `IsolationChecker` compares handles with `==`. A class that overloads `eq` by content can produce false positives. It does not look inside `Constant` properties, `persistent` variables or the internals of Java objects.
- A saved TD `pi` with 0 in terminal rows is rejected by `PolicyArtifact`. The "no action here" marker is decided in A5.
- Checkpoints, message phases, skills and commitments are not implemented yet.
- Trust (A2): direct evidence only (no indirect reputation), no forgetting or windowing, no identity defence. A store's `version()` counts applied updates only and does not change on an episode reset.

## See Also

- Design and contract: `Planning/System/Game Theory/Implementation_Build_Plan_A0_A1.md`, `A0_Contract_Freeze.md`, `A2_Beliefs_Contract.md`, `Agent_Model_Architecture_and_Problem_Analysis.md`
- Environments reused: [Monte Carlo RL guide](monte_carlo_rl_guide.md), `OceanStats.RL.Env`
- Tests: `runtests('OceanTest.OceanAgents','IncludeSubpackages',true)`
