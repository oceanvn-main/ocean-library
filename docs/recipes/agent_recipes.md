# OceanAgents Recipes

Copy-paste snippets for `+OceanAgents` (see the [guide](../guides/ocean_agents.md)). Every snippet was run in MATLAB R2026a on 2026-09-28; the outputs shown are from that run.

## 1. Q-learning on CliffWalking through an agent

```matlab
env     = OceanStats.RL.Env.Classic.CliffWalkingEnv();
bridge  = OceanAgents.Env.NativeSingleAgentBridge(OceanStats.RL.Env.Wrappers.TimeLimit(env, 200));
codec   = OceanAgents.Artifacts.DiscreteStateCodec(48);
q       = OceanStats.RL.TD.Values.TabularQ(env.ValidMask, 0);      % the agent's own value table
policy  = OceanAgents.Policies.EpsilonGreedyQPolicy(q, 1:4, 'Epsilon', 0.1, 'StateCodec', codec);
learner = OceanAgents.Learners.OneStepQLearner(q, 1:4, 'Alpha', 0.5, 'Gamma', 1, 'StateCodec', codec);
host    = OceanAgents.Runtime.AgentHost("agent1", OceanAgents.Minds.PolicyMind(policy), ...
              'Learner', learner, 'Gamma', 1);
runner  = OceanAgents.Runtime.SingleAgentRunner(bridge, host, 'Seed', 7);
rec     = runner.run(60);
fprintf('%s: %d episodes, %d steps, mean return first/last 10: %.1f / %.1f\n', rec.Status, ...
    rec.NumEpisodes, rec.StepCalls, mean(rec.EpisodeReturns(1:10)), mean(rec.EpisodeReturns(end-9:end)));
% Completed: 60 episodes, 2885 steps, mean return first/last 10: -290.8 / -124.0
```

The policy and the learner share one `TabularQ`, which belongs to this one agent. Build a new table for every agent.

## 2. Evaluate without learning

```matlab
runner.release();                         % a host binds to one runner at a time
host.setLearningEnabled(false);
before  = q.Q;
evalRec = OceanAgents.Runtime.SingleAgentRunner(bridge, host, 'Seed', 99).run(10);
fprintf('eval: %s, Q unchanged: %d, updates: %d\n', evalRec.Status, isequal(q.Q, before), evalRec.UpdateCounts);
% eval: Completed, Q unchanged: 1, updates: 0
```

## 3. Act from a saved table

```matlab
env2 = OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner', 5);
art  = OceanAgents.Artifacts.PolicyArtifact('Kind', "ActionTable", 'Representation', "Label", ...
    'Table', [2; 2; 2; 2; 2; 2; 2], 'ActionLabels', [1 2], 'NumStates', 7, 'StateStart', 1, ...
    'Role', "default", 'Version', "1", 'Provenance', struct('source', "hand-written: always right"));
host2 = OceanAgents.Runtime.AgentHost("walker", OceanAgents.Minds.PolicyMind(OceanAgents.Policies.TablePolicy(art)));
rec2  = OceanAgents.Runtime.SingleAgentRunner( ...
    OceanAgents.Env.NativeSingleAgentBridge(env2, 'AgentID', "walker"), host2, 'Seed', 1).run(3);
fprintf('returns %s, lengths %s\n', mat2str(rec2.EpisodeReturns'), mat2str(rec2.EpisodeLengths'));
% returns [1 1 1], lengths [3 3 3]
```

`Representation` is required. Use `"Label"` when the entries are environment action labels (e.g. TD `pi`) and `"ColumnIndex"` when they are table columns (e.g. MC `PolicyTable`). A mismatch against the environment (role, label set, state range) is rejected when the runner is built, before any reset.

## 4. Two agents play repeated matching pennies

```matlab
A     = [1 -1; -1 1];
game  = OceanStats.GameTheory.Core.NormalFormGame('PayoffA', A, 'PayoffB', -A);
world = OceanAgents.Env.Reference.RepeatedMatrixGameEnv(game, 'Roster', ["p1" "p2"], 'Rounds', 50);
hosts = {OceanAgents.Runtime.AgentHost("p1", OceanAgents.Minds.PolicyMind(OceanAgents.Policies.RandomLegalPolicy())), ...
         OceanAgents.Runtime.AgentHost("p2", OceanAgents.Minds.PolicyMind(OceanAgents.Policies.RandomLegalPolicy()))};
rec3  = OceanAgents.Runtime.JointRunner(world, hosts, 'Seed', 3).run(20);
fprintf('%s, p1 total %d, p2 total %d\n', rec3.Status, sum(rec3.EpisodeReturns(:,1)), sum(rec3.EpisodeReturns(:,2)));
% Completed, p1 total -16, p2 total 16
```

Both players decide from the same frame. The runner validates the whole batch before one joint step, so the order in which agents are asked does not change the result.

## 5. Check that two agents share nothing

```matlab
paths = OceanAgents.Runtime.IsolationChecker.sharedHandles(hosts{1}, hosts{2});
fprintf('shared handles: %d\n', numel(paths));      % 0; otherwise paths name where the shared object sits
% shared handles: 0
```

Runners run this check themselves at construction and refuse shared components (`OceanAgents:Runtime:SharedComponent`).

## 6. A failure is recorded, not retried

```matlab
loop = OceanAgents.Artifacts.PolicyArtifact('Kind', "ActionTable", 'Representation', "Label", ...
    'Table', [2; 2; 2; 2; 1; 2; 2], 'ActionLabels', [1 2], 'NumStates', 7, 'StateStart', 1, ...
    'Role', "default", 'Version', "1");                  % state 4 -> right, state 5 -> left: never ends
h = OceanAgents.Runtime.AgentHost("w2", OceanAgents.Minds.PolicyMind(OceanAgents.Policies.TablePolicy(loop)));
r = OceanAgents.Runtime.SingleAgentRunner(OceanAgents.Env.NativeSingleAgentBridge(env2, 'AgentID', "w2"), ...
    h, 'Seed', 1, 'MaxStepsPerEpisode', 100);
rec = r.run(1);
fprintf('%s at stage "%s": %s\n', rec.Status, rec.FailureStage, rec.Failure.identifier);
% Failed at stage "step": OceanAgents:Runtime:StepBudgetExceeded
```

After a failure the host and runner are `Invalid`. Build fresh objects to run again. Use `TimeLimit` when you want a deliberate truncation instead.

## 7. A Beta trust store by hand (A2)

```matlab
s = OceanAgents.Beliefs.BetaTrustStore('Prior', [1 1]);
e = @(id, out) OceanAgents.Core.EvidenceRecord('EvidenceID', id, 'SubjectID', "p1", 'Context', "cheap", ...
    'Outcome', out, 'SourceID', "me", 'EpisodeID', 1);
d = s.observe([e("E1", "success"), e("E2", "success"), e("E3", "failure"), e("E1", "success"), e("E4", "unknown")]);
b = s.belief("p1", "cheap");
fprintf('%s | Beta(%g,%g) mean %.2f var %.3f mass %d\n', strjoin(d, ' '), b.Alpha, b.Beta, b.Estimate, b.Variance, b.EvidenceMass);
% applied applied applied duplicate unknown | Beta(3,2) mean 0.60 var 0.040 mass 3
```

Each event is consumed once:
- A repeated ID with the same content is a `"duplicate"`.
- A repeated ID with different content raises an `OceanAgents:Beliefs:EvidenceConflict` fault.
- An `"unknown"` outcome, such as an obstruction, never counts as a failure.

## 8. Imported scores need a declared scale

```matlab
scores = struct('SubjectID', {"p1", "p2"}, 'Context', "cheap", 'Score', {80, 40}, 'ScaleMin', 0, 'ScaleMax', 100, ...
    'Conversion', "linear-scale", 'SourceID', "survey");
f  = OceanAgents.Beliefs.FixedBeliefStore(scores);
fb = f.belief("p2", "cheap");
fprintf('p2: raw %g -> estimate %.2f (%s, %s)\n', fb.RawScore, fb.Estimate, fb.Conversion, fb.SourceKind);
% p2: raw 40 -> estimate 0.40 (linear-scale, import)
```

A 0–100 score with `'Conversion', "none"` is refused with `Beliefs:BadImport`. A score is not a probability unless you declare how it becomes one.

## 9. A fixed-belief agent and an adaptive agent in one delegation world

```matlab
pay = struct('cheap', [2 -1 0], 'expensive', [10 -20 0]);            % [success failure outside]
env = OceanAgents.Env.Reference.DelegationEnv('Roster', ["fixed" "adaptive"], 'Providers', ["p1" "p2"], ...
    'Contexts', ["cheap" "expensive"], 'Reliability', [0.9 0.9; 0.4 0.4], ...
    'Schedule', struct('FromRound', 41, 'Reliability', [0.1 0.1; 0.8 0.8]), ...   % p1 and p2 swap at round 41
    'Payoffs', pay, 'ObstructionProb', 0.1, 'Visibility', "public", 'Rounds', 80);
fixedScores = struct('SubjectID', {"p1", "p1", "p2", "p2"}, 'Context', {"cheap", "expensive", "cheap", "expensive"}, ...
    'Score', {0.9, 0.9, 0.4, 0.4}, 'ScaleMin', 0, 'ScaleMax', 1, 'Conversion', "none", 'SourceID', "config");
fixedMind = OceanAgents.Minds.TrustMind(OceanAgents.Beliefs.FixedBeliefStore(fixedScores), ...
    OceanAgents.Minds.TrustRules.ExpectedUtilityRule(), pay);
trust     = OceanAgents.Beliefs.BetaTrustStore('Prior', [1 1]);
adaptMind = OceanAgents.Minds.TrustMind(trust, OceanAgents.Minds.TrustRules.ThompsonRule(), pay);
hosts = {OceanAgents.Runtime.AgentHost("fixed", fixedMind), OceanAgents.Runtime.AgentHost("adaptive", adaptMind)};
rec = OceanAgents.Runtime.JointRunner(env, hosts, 'Seed', 3).run(1);
b1 = trust.belief("p1", "expensive");  b2 = trust.belief("p2", "expensive");
fprintf('%s; returns fixed %g, adaptive %g; adaptive trust (expensive): p1 %.2f, p2 %.2f\n', rec.Status, ...
    rec.EpisodeReturns(1), rec.EpisodeReturns(2), b1.Estimate, b2.Estimate);
% Completed; returns fixed -147, adaptive -58; adaptive trust (expensive): p1 0.51, p2 0.56
```

This is one seed and one run. It illustrates the mechanism; it is not evidence that one mind is better in general. Each decision's `Provenance` records the beliefs, scores and evidence mass that produced it.

## 10. Frozen-belief evaluation

```matlab
trust.setUpdatesEnabled(false);
v  = trust.version();
d2 = trust.observe(OceanAgents.Core.EvidenceRecord('EvidenceID', "late-1", 'SubjectID', "p2", 'Context', "cheap", ...
    'Outcome', "failure", 'SourceID', "me", 'EpisodeID', 9));
fprintf('%s, version %d -> %d\n', d2, v, trust.version());
% frozen, version 138 -> 138
```

The event is still consumed once, so it will not be re-applied later, but the beliefs do not move.


## 11. A world kernel: a battery that drains, and time that jumps (W1)

A world is built from **doors**, each a small container of rules and the components it alone writes. The clock never polls. `Charge` is a *linear* component (value = Base + Rate·(now − T0)), so it drains with no work at all. A predictor states when it will hit zero, and the world jumps straight to that instant.

```matlab
D = OceanAgents.World.Guard.Dimension;
chargeT = OceanAgents.World.Guard.ComponentType('Kind', "linear", 'Dimension', D.E());
predict = OceanAgents.World.RuleSpec('Name', "ZeroCrossing", 'Phase', "settle", 'Kind', "predictor", ...
    'Requires', "Charge", 'Reads', "Charge", 'Predicts', "Charge", 'DueEvent', "Empty", 'Fn', @zeroCrossing);
drain = OceanAgents.World.RuleSpec('Name', "Drain", 'Phase', "resolve", 'Consumes', "Empty", ...
    'Commands', "remove", 'Fn', @drainFn);
battery = OceanAgents.World.DoorSpec('Name', "battery", 'Owns', struct('Name', "Charge", 'Type', chargeT), ...
    'Membership', "Charge", 'Rules', [predict drain], 'Drive', "on-event", ...
    'Axioms', OceanAgents.World.Guard.AxiomSpec('ID', "BT-1", 'Mode', "monitor", 'Component', "Charge", 'Range', [0 Inf]), ...
    'Init', @(stream) OceanAgents.World.Command.spawn(struct('Charge', ...
        struct('Value', struct('Value', 10, 'Rate', -0.5), 'Type', chargeT))));
w = OceanAgents.World.World(battery, 'SkipIdle', true);
w.reset(1);
reports = w.run(100);
fprintf('instants processed: %s; entity alive afterwards: %d\n', mat2str([reports.Time]), w.Registry.isAlive(1));

% local functions (put them at the end of your script file)
function out = zeroCrossing(ctx)
c = ctx.Read.Charge;  r = ctx.Read.Charge_Rate;
times = inf(numel(ctx.Rows), 1);
f = r < 0;
times(f) = ceil(ctx.Now + c(f) ./ (-r(f)));        % first whole tick at or after the crossing
out = struct('Predictions', struct('Rows', ctx.Rows, 'Times', times));
end

function out = drainFn(ctx)
ids = reshape(unique([ctx.Events.SourceEntity]), 1, []);
cmds = OceanAgents.World.Command.empty(1, 0);
for id = ids, cmds(end + 1) = OceanAgents.World.Command.remove(id); end %#ok<AGROW>
out = struct('Commands', cmds);
end
```
Output: `instants processed: 20; entity alive afterwards: 0`. Instants 0–19 were skipped, and the result is identical to stepping every tick (kernel test K9).

## 12. The guard admits by types and axioms, never by names (W1)

A door is also a guard. An offered entity must have components whose **types, including physical dimensions**, match the owner's declaration, and it must satisfy the door's **axioms**. No name is ever compared.

```matlab
speedT = OceanAgents.World.Guard.ComponentType('Dimension', D.L() / D.T());
massT = OceanAgents.World.Guard.ComponentType('Dimension', D.M());
human = OceanAgents.World.DoorSpec('Name', "human", 'Owns', struct('Name', {"Speed", "Mass"}, 'Type', {speedT, massT}), ...
    'Membership', ["Speed" "Mass"], 'Drive', "never", 'Axioms', [ ...
    OceanAgents.World.Guard.AxiomSpec('ID', "HB-1", 'Component', "Speed", 'Range', [0 12]), ...
    OceanAgents.World.Guard.AxiomSpec('ID', "HB-2", 'Reads', "Mass", 'Predicate', @(v) v.Mass <= 300)], ...
    'Init', @(stream) [ ...
        OceanAgents.World.Command.spawn(struct('Speed', struct('Value', 1.4, 'Type', speedT), 'Mass', struct('Value', 70, 'Type', massT))), ...
        OceanAgents.World.Command.spawn(struct('Speed', struct('Value', 30, 'Type', speedT), 'Mass', struct('Value', 1200, 'Type', massT))), ...
        OceanAgents.World.Command.spawn(struct('Speed', struct('Value', 5, 'Type', massT), 'Mass', struct('Value', 70, 'Type', massT)))]);
w2 = OceanAgents.World.World(human);
w2.reset(1);
rr = w2.ResetReport;
refused = rr.CommandsRefused;
fprintf('admitted: %d; refused:\n', w2.Registry.numAlive());
for k = 1:numel(refused)
    fprintf('  %s  %s\n', refused(k).Reason, strjoin(string(refused(k).AxiomIDs), '+'));
end
```
Output:
```text
admitted: 1; refused:
  axiom  HB-1+HB-2
  type-mismatch
```
The walker is admitted. The "car" (30 m/s, 1200 kg) is refused by axioms HB-1 and HB-2. The third offer gives a speed with a mass dimension and is refused as a type mismatch.

## 13. A door-built grid world is the native environment (W2)

`GridWorlds` assembles Agency + GridMotion + RewardTermination; `WorldAsEnvironment` presents the result as a native environment. Same seed and actions give the same rows, and the exact model is the same.

```matlab
native = OceanStats.RL.Env.Classic.WindyGridworldEnv('StochasticWind', true);
world = OceanAgents.Env.WorldAsEnvironment(OceanAgents.World.Doors.GridWorlds.windy('StochasticWind', true));
pick = RandStream('mt19937ar', 'Seed', 1);
a = native.reset('Seed', 7);  b = world.reset('Seed', 7);
same = isequal(a, b);  steps = 0;
for t = 1:200
    act = randi(pick, 4);
    [a, ra, ta] = native.step(act);  [b, rb, tb] = world.step(act);
    same = same && isequal([a ra ta], [b rb tb]);  steps = steps + 1;
    if ta, break; end
end
[n1, p1, r1] = native.get_transitions(31, 2);  [n2, p2, r2] = world.get_transitions(31, 2);
fprintf('steps compared: %d; identical: %d; model identical: %d
', steps, same, isequal({n1, p1, r1}, {n2, p2, r2}));
```
Output: `steps compared: 200; identical: 1; model identical: 1`.

## 14. The RL door admits an environment by probing its behaviour (W2)

The RL door holds no rules, only axioms RL-1…RL-9 (reset and step shape, determinism under a seed, refusal of illegal steps, terminal handling, …). Any object that behaves correctly is admitted, whatever its class.

```matlab
door = OceanAgents.World.Doors.RLDoor.spec('Probes', 4);
cert = OceanAgents.World.Guard.GuardEngine.admitModule(world, door, 'ModuleVersion', "1");
fprintf('admitted: %d; axioms: %s; statuses: %s
', cert.Admitted, strjoin([cert.Axioms.ID], ' '), ...
    strjoin(unique([cert.Axioms.Status]), ' '));
```
Output: `admitted: 1; axioms: RL-1 RL-2 RL-3 RL-4 RL-5 RL-6 RL-7 RL-8 RL-9; statuses: assumed probed`. An axiom that cannot be probed is reported as `assumed`, never as checked.

## 15. An agent as mind, body and memory (AB1)

The A1 Q-learning agent rebuilt as a signal mind with a grid robot body. The run is identical to recipe 1; the mind asks its value memory by signal.

```matlab
env     = OceanStats.RL.Env.Classic.CliffWalkingEnv();
codec   = OceanAgents.Artifacts.DiscreteStateCodec(48);
q       = OceanStats.RL.TD.Values.TabularQ(env.ValidMask, 0);
mind    = OceanAgents.Minds.QSignalMind(q, 1:4, 'Epsilon', 0.1, 'StateCodec', codec);
OceanAgents.Embodied.Dualism.form(mind, OceanAgents.Body.GridRobotBody(string(1:4)));
learner = OceanAgents.Learners.OneStepQLearner(q, 1:4, 'Alpha', 0.5, 'Gamma', 1, 'StateCodec', codec);
mind.addLearner(learner, "q");
host    = OceanAgents.Runtime.AgentHost("agent1", mind, 'Gamma', 1);
bridge  = OceanAgents.Env.NativeSingleAgentBridge(OceanStats.RL.Env.Wrappers.TimeLimit(env, 200));
rec     = OceanAgents.Runtime.SingleAgentRunner(bridge, host, 'Seed', 7).run(60);
best    = mind.query("values", struct('Op', "best", 'State', 37));
fprintf('%s: mean return first/last 10: %.1f / %.1f; best move at the start cell: %s\n', rec.Status, ...
    mean(rec.EpisodeReturns(1:10)), mean(rec.EpisodeReturns(end-9:end)), best.Label);
```
Output: `Completed: mean return first/last 10: -290.8 / -124.0; best move at the start cell: 1`.

## 16. Moving a mind to a different body (AB1)

A body answers the mind's probe with its signature, so the mind can compare bodies. A body may offer more than the mind uses; a body that lacks one of the mind's commands is refused.

```matlab
four = string(1:4);  king = string(1:9);
d1 = OceanAgents.Embodied.Dualism.form(mind, OceanAgents.Body.GridRobotBody(four));
d2 = OceanAgents.Embodied.Dualism.form(mind, OceanAgents.Body.GridRobotBody(king));
delta = d1.Signature.diff(d2.Signature);
fprintf('new commands: %s; lost: %s\n', strjoin(delta.Added, ' '), strjoin([delta.Removed "none"], ' '));
try
    OceanAgents.Embodied.Dualism.form(OceanAgents.Minds.QSignalMind(OceanStats.RL.TD.Values.TabularQ(true(4, 9), 0), 1:9), ...
        OceanAgents.Body.GridRobotBody(four));
catch err
    fprintf('%s\n', err.identifier);
end
```
Output: `new commands: 5 6 7 8 9; lost: none`, then `OceanAgents:Embodied:Incompatible`.

## 17. A wave crosses a threshold at an exact time (AB1)

A wave is a formula, so the next crossing is solved, not found by stepping.

```matlab
W = OceanAgents.Minds.Signals.Wave.sine(2, 1/40, 'Offset', 0.5);
t = W.nextCrossing(2, 0, "up");
fprintf('the wave first reaches 2 at t = %.4f (value there %.4f)\n', t, W.valueAt(t));
```
Output: `the wave first reaches 2 at t = 5.3989 (value there 2.0000)`.

## 18. A moves-only agent plays tic-tac-toe (AB2)

An empty body with one sensor, what the Laws permit now. The agent does not know the board; it only proves the pieces work end to end.

```matlab
env  = OceanAgents.Env.Reference.TicTacToeEnv('Roster', ["x" "o"]);
mind = OceanAgents.Minds.PolicySignalMind(OceanAgents.Policies.RandomLegalPolicy(), 1:9);
OceanAgents.Embodied.Dualism.form(mind, OceanAgents.Body.EmptyBody(string(1:9), ...
    'Sensors', {OceanAgents.Body.Sensors.MovesObservation(string(1:9))}));
x = OceanAgents.Runtime.AgentHost("x", mind);
o = OceanAgents.Runtime.AgentHost("o", OceanAgents.Minds.PolicyMind(OceanAgents.Policies.RandomLegalPolicy()));
rec = OceanAgents.Runtime.JointRunner(env, {x, o}, 'Seed', 3).run(200);
fprintf('%s: x won %d, drew %d, lost %d of %d games\n', rec.Status, nnz(rec.EpisodeReturns(:, 1) == 1), ...
    nnz(rec.EpisodeReturns(:, 1) == 0), nnz(rec.EpisodeReturns(:, 1) == -1), rec.NumEpisodes);
```
Output: `Completed: x won 127, drew 27, lost 46 of 200 games`.

## 19. The mind acts on what it observes, and needs the right sensors (AB2)

The moves sensor says only move 2 is permitted, so that is what the mind chooses. A Q mind needs a state (a digit observation); a body without one is refused.

```matlab
labels = string(1:4);
q    = OceanStats.RL.TD.Values.TabularQ(true(48, 4), 0);
mind = OceanAgents.Minds.QSignalMind(q, 1:4, 'Epsilon', 0);
body = OceanAgents.Body.EmptyBody(labels, 'Sensors', {OceanAgents.Body.Sensors.DigitObservation(), ...
    OceanAgents.Body.Sensors.MovesObservation(labels)});
OceanAgents.Embodied.Dualism.form(mind, body);
v = OceanAgents.Core.ActorView('AgentID', "a", 'EpisodeID', 1, 'FrameID', 1, 'Tick', 0, 'Observation', 37, ...
    'ActionLabels', 1:4, 'ActionMask', logical([0 1 0 0]));
mind.beginEpisode(v);
d = mind.decide(v, RandStream('mrg32k3a', 'Seed', 1));
fprintf('the Laws permit only move 2 here; the mind chose %d\n', d.Action);
try
    OceanAgents.Embodied.Dualism.form(OceanAgents.Minds.QSignalMind(q, 1:4), ...
        OceanAgents.Body.EmptyBody(labels, 'Sensors', {OceanAgents.Body.Sensors.MovesObservation(labels)}));
catch err
    fprintf('%s\n', err.identifier);
end
```
Output: `the Laws permit only move 2 here; the mind chose 2`, then `OceanAgents:Embodied:Incompatible`.

## 20. Cliff walking made of devices (DV1)

No environment class holds the rules: the terrain device moves the position, the reward device pays.

```matlab
env = OceanAgents.Devices.DeviceWorld.cliffWalking();
s = env.reset('Seed', 7);
[s2, r] = env.step(2);                                   % right from the start: into the cliff
fprintf('start %d, step right: back to %d with reward %d (the terrain device moved the position to %d)\n', ...
    s, s2, r, env.Position.Cell);
vd = OceanStats.RL.DP.ValueIterationSolver(env, 'DiscountFactor', 0.9);
vn = OceanStats.RL.DP.ValueIterationSolver(OceanStats.RL.Env.Classic.CliffWalkingEnv(), 'DiscountFactor', 0.9);
fprintf('value iteration on devices equals the native result: %d\n', isequal(vd.run(), vn.run()));
```
Output: `start 37, step right: back to 37 with reward -100 (the terrain device moved the position to 37)`, then `value iteration on devices equals the native result: 1`.

## 21. A player that learns from the other exploits it (PG1)

Player 1 puts even 90 % of the time and guesses at random; player 2 models player 1 by fictitious play and best-responds.

```matlab
env  = OceanAgents.Env.Reference.ParityGuessEnv('Roster', ["p1" "p2"]);
F    = env.Factors;                                                     % put x guess, each even / odd
coin = @(pEven) OceanAgents.Policies.TablePolicy(OceanAgents.Artifacts.PolicyArtifact('Kind', "ProbabilityTable", ...
    'Representation', "ColumnIndex", 'Table', repmat([pEven 1 - pEven], 3, 1), 'ActionLabels', 1:2, 'StateStart', 0));
sees = @() {OceanAgents.Minds.Perception.MovesPerception(string(1:4)), OceanAgents.Minds.Perception.DigitPerception()};
m1 = OceanAgents.Minds.PolicySignalMind(OceanAgents.Policies.FactoredPolicy(F, ...
        {coin(0.9), OceanAgents.Policies.RandomLegalPolicy()}), 1:4, 'Perceptions', sees());
fp = OceanStats.GameTheory.Learning.FictitiousPlay(eye(2), 1);           % my guess vs the other's put
m2 = OceanAgents.Minds.PolicySignalMind(OceanAgents.Policies.FactoredPolicy(F, ...
        {coin(0.7), OceanAgents.Policies.FictitiousPlayPolicy(fp)}), 1:4, 'Perceptions', sees());
m2.addMemory(OceanAgents.Memory.BeliefMemory("belief", fp));
m2.addLearner(OceanAgents.Learners.FictitiousPlayLearner(fp), "fp");
p1 = OceanAgents.Player("p1", m1);
p2 = OceanAgents.Player("p2", m2);
rec = OceanAgents.Runtime.JointRunner(env, {p1, p2}, 'Seed', 5).run(1000);
b = fp.beliefs();
fprintf('P2 wins %.3f of 1000 games (theory 0.474); it believes P1 puts even with probability %.3f\n', ...
    mean(rec.EpisodeReturns(:, 2) == 1), b{1}(1));
```
Output: `P2 wins 0.497 of 1000 games (theory 0.474); it believes P1 puts even with probability 0.891`.
