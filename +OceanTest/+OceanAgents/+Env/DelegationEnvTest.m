classdef DelegationEnvTest < OceanTest.BaseTestCase
    % DELEGATIONENVTEST  OceanAgents.Env.Reference.DelegationEnv (A2 contract section 5): documented draw order replayed
    % independently on a twin RandStream, slot-order resolution, own vs public visibility, obstruction -> "unknown",
    % scripted reports at the start of their round, reliability schedule, horizon, hidden truth (paired perturbation),
    % rewards = declared payoffs, seeded reproducibility, batch validated before mutation, describe and pillars.
    % Expected values: Planning/System/Game Theory/A2_Beliefs_Contract.md section 5.

    properties (Constant)
        Roster = ["alice" "bob" "carol"]
        Providers = ["p1" "p2"]
        Contexts = ["cheap" "expensive"]
        ContextProbs = [0.6 0.4]
        R = [0.9 0.3; 0.4 0.7]                          % providers x contexts (hidden truth)
        Obstructed = -3
    end

    methods (Static, Access = private)
        function pay = payoffs()
            pay = struct('cheap', [10 -20 0], 'expensive', [30 -50 5]);
        end

        function env = make(varargin)
            C = OceanTest.OceanAgents.Env.DelegationEnvTest;
            env = OceanAgents.Env.Reference.DelegationEnv('Roster', C.Roster, 'Providers', C.Providers, ...
                'Contexts', C.Contexts, 'ContextProbs', C.ContextProbs, 'Reliability', C.R, ...
                'Payoffs', C.payoffs(), 'ObstructedPayoff', C.Obstructed, 'Rounds', 6, 'RunTag', "T", varargin{:});
        end

        function b = batch(frame, actions, order)
            % Decisions for every requested slot, listed in 'order' (a permutation of the slots).
            if nargin < 3, order = find(frame.Requested); end
            ds = OceanAgents.Core.Decision.empty(1, 0);
            for k = order
                ds(end + 1) = OceanAgents.Core.Decision('AgentID', frame.Roster(k), 'EpisodeID', frame.EpisodeID, ...
                    'FrameID', frame.FrameID, 'Action', actions(k)); %#ok<AGROW>
            end
            b = OceanAgents.Core.ActionBatch('EpisodeID', frame.EpisodeID, 'FrameID', frame.FrameID, 'Decisions', ds);
        end

        function [frames, ctx] = play(env, seed, A, order)
            % Run one episode with the scripted action matrix A (T x N); returns every frame and each round's context.
            if nargin < 4, order = []; end
            f = env.reset(seed);
            frames = {f};
            ctx = zeros(size(A, 1), 1);
            for t = 1:size(A, 1)
                ctx(t) = f.Views{1}.Observation(1);
                if isempty(order)
                    f = env.step(OceanTest.OceanAgents.Env.DelegationEnvTest.batch(f, A(t, :)));
                else
                    f = env.step(OceanTest.OceanAgents.Env.DelegationEnvTest.batch(f, A(t, :), order));
                end
                frames{end + 1} = f; %#ok<AGROW>
            end
        end

        function [ctx, outcome, reward] = replay(seed, A, R, probs, pay, obstructionProb, obstructed, contexts)
            % Independent recomputation of the documented draw order on a twin RandStream (A2 section 5):
            % context(1) at reset; per round, delegating slots in slot order draw u1 (then u2 unless obstructed);
            % then the next round's context unless the round was the last.
            s = RandStream('mrg32k3a', 'Seed', seed);
            [T, n] = size(A);
            P = size(R, 1);
            ctx = zeros(T, 1);
            outcome = strings(T, n);
            reward = zeros(T, n);
            cdf = cumsum(probs);
            ctx(1) = find(rand(s) < cdf, 1);
            for t = 1:T
                c = ctx(t);
                u = pay.(char(contexts(c)));
                for k = 1:n
                    a = A(t, k);
                    if a == P + 1
                        outcome(t, k) = "outside";  reward(t, k) = u(3);
                    elseif rand(s) < obstructionProb
                        outcome(t, k) = "unknown";  reward(t, k) = obstructed;
                    elseif rand(s) < R(a, c)
                        outcome(t, k) = "success";  reward(t, k) = u(1);
                    else
                        outcome(t, k) = "failure";  reward(t, k) = u(2);
                    end
                end
                if t < T, ctx(t + 1) = find(rand(s) < cdf, 1); end
            end
        end

        function ids = idsOf(view)
            ids = view.evidenceIDs();
        end
    end

    methods (Test)
        function initialFrameAndActorInfo(tc)
            env = tc.make();
            f = env.reset(11);
            tc.verifyEqual(f.Requested, true(1, 3));
            tc.verifyEqual(f.Live, true(1, 3));
            tc.verifyEqual(f.RewardValid, false(1, 3));
            c = f.Views{1}.Observation(1);
            for k = 1:3
                v = f.Views{k};
                tc.verifyEqual(v.ActionLabels, 1:3);
                tc.verifyEqual(v.ActionMask, true(1, 3));
                tc.verifyEqual(v.Observation, [c 1]);            % common context, round 1
                tc.verifyEqual(v.Info.context, tc.Contexts(c));
                tc.verifyEqual(v.Info.action_subjects, ["p1" "p2" "outside"]);
                tc.verifyEqual(sort(string(fieldnames(v.Info)))', ["action_subjects" "context"]);   % nothing hidden
                tc.verifyEmpty(v.Evidence);
            end
        end

        function replayMatchesIndependentRecomputation(tc)
            % Mixed scripted actions (outside + both providers), obstruction 0.3, decisions listed in reverse order.
            T = 60;
            gen = RandStream('mrg32k3a', 'Seed', 99);             % test-owned stream for the action script
            A = randi(gen, 3, T, 3);
            pay = tc.payoffs();
            env = tc.make('Rounds', T, 'ObstructionProb', 0.3, 'Visibility', "public");
            [frames, ctx] = tc.play(env, 5, A, [3 2 1]);
            [ctxR, outR, rewR] = tc.replay(5, A, tc.R, tc.ContextProbs, pay, 0.3, tc.Obstructed, tc.Contexts);
            tc.verifyEqual(ctx, ctxR);
            rewards = cell2mat(cellfun(@(f) f.Rewards, frames(2:end)', 'UniformOutput', false));
            tc.verifyEqual(rewards, rewR);
            for t = 1:T
                ev = frames{t + 1}.Views{1}.Evidence;              % public: every delegation of round t
                delegating = find(A(t, :) <= 2);
                if isempty(delegating)
                    tc.verifyEmpty(ev);
                    continue
                end
                tc.verifyEqual([ev.Outcome], outR(t, delegating), sprintf('round %d', t));
                tc.verifyEqual([ev.SubjectID], tc.Providers(A(t, delegating)));
                tc.verifyEqual([ev.Context], repmat(tc.Contexts(ctxR(t)), 1, numel(delegating)));
            end
            % rewards are exactly the declared payoffs of the outcome (obstruction never becomes a failure)
            for t = 1:T
                u = pay.(char(tc.Contexts(ctxR(t))));
                byOutcome = containers.Map(["success" "failure" "outside" "unknown"], {u(1), u(2), u(3), tc.Obstructed});
                for k = 1:3
                    tc.verifyEqual(rewards(t, k), byOutcome(outR(t, k)));
                end
            end
            tc.verifyTrue(any(outR(:) == "unknown") && any(outR(:) == "success") && any(outR(:) == "failure"), ...
                'the script should exercise every outcome');
        end

        function resolutionIndependentOfDecideOrder(tc)
            A = [1 2 3; 2 2 1; 3 1 1; 1 1 1; 2 3 2; 1 2 1];
            [fa, ~] = tc.play(tc.make('ObstructionProb', 0.2), 3, A, [1 2 3]);
            [fb, ~] = tc.play(tc.make('ObstructionProb', 0.2), 3, A, [3 1 2]);
            tc.verifyEqual(fb, fa);
        end

        function ownVersusPublicVisibility(tc)
            A = [1 3 2; 3 3 3];                                    % round 1: alice->p1, bob outside, carol->p2
            fo = tc.play(tc.make('Visibility', "own", 'Rounds', 2), 4, A);
            fp = tc.play(tc.make('Visibility', "public", 'Rounds', 2), 4, A);
            f = fo{2};
            tc.verifyEqual(tc.idsOf(f.Views{1}), "T/E1-1-alice");
            tc.verifyEmpty(f.Views{2}.Evidence);                  % bob delegated nothing and sees nothing
            tc.verifyEqual(tc.idsOf(f.Views{3}), "T/E1-1-carol");
            e = f.Views{1}.Evidence;
            tc.verifyEqual([e.SubjectID e.Quantity e.SourceKind e.SourceID], ["p1" "completion" "direct" "alice"]);
            tc.verifyEqual([e.EpisodeID e.Tick], [1 1]);
            g = fp{2};
            for k = 1:3                                            % public: identical records for every live agent
                tc.verifyEqual(g.Views{k}.Evidence, [fo{2}.Views{1}.Evidence fo{2}.Views{3}.Evidence]);
            end
            tc.verifyEmpty(fp{3}.Views{1}.Evidence);               % round 2: nobody delegated
        end

        function obstructionIsUnknownNeverFailure(tc)
            A = repmat([1 2 3], 4, 1);
            [frames, ctx] = tc.play(tc.make('ObstructionProb', 1, 'Visibility', "public", 'Rounds', 4), 8, A);
            pay = tc.payoffs();
            for t = 1:4
                f = frames{t + 1};
                tc.verifyEqual([f.Views{1}.Evidence.Outcome], ["unknown" "unknown"]);
                tc.verifyEqual(f.Rewards, [tc.Obstructed tc.Obstructed pay.(char(tc.Contexts(ctx(t))))(3)]);
                tc.verifyEqual(f.ResearchInfo.Resolved.Drawn, [1 1 0]);   % u2 is never drawn when obstructed
            end
        end

        function reportsArriveAtStartOfTheirRound(tc)
            reports = struct('Round', {1, 3}, 'ReporterID', {"gossip", "critic"}, 'SubjectID', {"p2", "p1"}, ...
                'Context', {"cheap", "expensive"}, 'Outcome', {"success", "failure"}, 'EvidenceID', {"r1", "r3"});
            env = tc.make('Reports', reports, 'Rounds', 4, 'Visibility', "own");
            A = [3 3 3; 1 3 3; 3 3 3; 3 3 3];                     % alice delegates in round 2 only
            frames = tc.play(env, 2, A);
            for k = 1:3                                            % initial view carries round-1 reports
                ev = frames{1}.Views{k}.Evidence;
                tc.verifyEqual([ev.EvidenceID ev.SourceKind ev.SourceID ev.SubjectID ev.Outcome], ...
                    ["T/r1@1" "report" "gossip" "p2" "success"]);
                tc.verifyEqual([ev.EpisodeID ev.Tick], [1 1]);
            end
            tc.verifyEmpty(frames{2}.Views{1}.Evidence);           % round-2 view: no report, no delegation yet
            % round-3 view: alice's own round-2 direct evidence first, then the round-3 report; others: report only
            tc.verifyEqual(tc.idsOf(frames{3}.Views{1}), ["T/E1-2-alice" "T/r3@1"]);
            tc.verifyEqual(tc.idsOf(frames{3}.Views{2}), "T/r3@1");
            tc.verifyEqual(tc.idsOf(frames{3}.Views{3}), "T/r3@1");
            tc.verifyEmpty(frames{4}.Views{1}.Evidence);
            tc.verifyEmpty(frames{5}.Views{1}.Evidence);           % final view: no reports after the horizon
            % a second episode re-delivers the scripted claims under run-unique IDs
            f = env.reset(2);
            tc.verifyEqual(tc.idsOf(f.Views{2}), "T/r1@2");
        end

        function scheduleChangesReliabilityFromRound(tc)
            R0 = [1 1; 0 0];                                       % p1 always succeeds, p2 always fails ...
            sched = struct('FromRound', 3, 'Reliability', [0 0; 1 1]);   % ... until round 3, then the reverse
            env = tc.make('Reliability', R0, 'Schedule', sched, 'Rounds', 5, 'Visibility', "public");
            A = repmat([1 2 3], 5, 1);
            frames = tc.play(env, 1, A);
            for t = 1:5
                f = frames{t + 1};
                if t < 3, expected = ["success" "failure"]; else, expected = ["failure" "success"]; end
                tc.verifyEqual([f.Views{1}.Evidence.Outcome], expected, sprintf('round %d', t));
                tc.verifyEqual(f.ResearchInfo.Resolved.Reliability, R0 * (t < 3) + sched.Reliability * (t >= 3));
            end
        end

        function horizonTerminatesWithFinalEvidence(tc)
            env = tc.make('Rounds', 2, 'Visibility', "own");
            f = env.reset(0);
            f = env.step(tc.batch(f, [3 3 3]));
            prev = f.FrameID;
            f = env.step(tc.batch(f, [2 3 1]));
            tc.verifyTrue(f.EpisodeEnded);
            tc.verifyEqual(f.EndReason, "horizon");
            tc.verifyEqual(f.Terminated, true(1, 3));
            tc.verifyEqual(f.Truncated, false(1, 3));
            tc.verifyEqual([f.Live f.Requested], false(1, 6));
            tc.verifyEqual(tc.idsOf(f.Views{1}), "T/E1-2-alice");
            tc.verifyEqual(tc.idsOf(f.Views{3}), "T/E1-2-carol");
            for k = 1:3
                fb = f.feedbackFor(k, prev);
                tc.verifyTrue(fb.Terminated);
                tc.verifyEqual(fb.Reason, "horizon");
                tc.verifyEqual(fb.NextView.ActionMask, false(1, 3));
                tc.verifyEqual(fb.NextView.Observation, [0 3]);
                tc.verifyEqual(fb.NextView.Info.context, "");
            end
            tc.verifyError(@() env.step(tc.batch(f, [])), 'OceanAgents:Env:NotReady');
        end

        function hiddenTruthPairedPerturbation(tc)
            % Two worlds differing ONLY in hidden reliability: every actor-facing field is identical until an outcome
            % reveals the difference; the context sequence (and draw count) never depends on it.
            % B's p1 is far less reliable. Rounds 1-5: everyone takes the outside option; rounds 6-10: delegations go
            % to p2 only (identical in both worlds); from round 11 alice delegates to p1, which can reveal the difference.
            RB = tc.R;
            RB(1, :) = [0.1 0.1];
            T = 30;
            A = [repmat([3 3 3], 5, 1); repmat([2 2 3], 5, 1); repmat([1 2 3], T - 10, 1)];
            envA = tc.make('Rounds', T, 'Visibility', "public");
            envB = OceanAgents.Env.Reference.DelegationEnv('Roster', tc.Roster, 'Providers', tc.Providers, ...
                'Contexts', tc.Contexts, 'ContextProbs', tc.ContextProbs, 'Reliability', RB, ...
                'Payoffs', tc.payoffs(), 'ObstructedPayoff', tc.Obstructed, 'Rounds', T, 'Visibility', "public", 'RunTag', "T");
            [fa, ca] = tc.play(envA, 17, A);
            [fb, cb] = tc.play(envB, 17, A);
            tc.verifyEqual(ca, cb);                                % contexts never depend on hidden reliability
            revealed = false;
            compared = 0;
            for j = 1:numel(fa)
                if ~revealed
                    tc.verifyEqual(fb{j}.Views, fa{j}.Views, sprintf('frame %d', j));
                    compared = compared + 1;
                end
                for k = 1:3                                        % masks, observations and Info always agree
                    tc.verifyEqual(fb{j}.Views{k}.Observation, fa{j}.Views{k}.Observation);
                    tc.verifyEqual(fb{j}.Views{k}.ActionMask, fa{j}.Views{k}.ActionMask);
                    tc.verifyEqual(fb{j}.Views{k}.Info, fa{j}.Views{k}.Info);
                end
                if j < numel(fa) && ~isequal(fa{j + 1}.ResearchInfo.Resolved.Outcomes, fb{j + 1}.ResearchInfo.Resolved.Outcomes)
                    revealed = true;                               % the next views may now legitimately differ
                end
            end
            tc.verifyTrue(revealed, 'the perturbation should eventually be revealed by an outcome');
            tc.verifyGreaterThanOrEqual(compared, 11);  % initial view + 10 rounds with no revealing outcome
            tc.verifyNotEqual(fa{end}.ResearchInfo.Reliability, fb{end}.ResearchInfo.Reliability);
        end

        function seededReproducibility(tc)
            A = repmat([1 2 3; 2 1 3], 3, 1);
            env = tc.make('ObstructionProb', 0.25);
            f1 = tc.play(env, 42, A);
            f2 = tc.play(env, 42, A);                              % same env, reset with the same seed
            f3 = tc.play(tc.make('ObstructionProb', 0.25), 42, A); % fresh env
            strip = @(fs) cellfun(@(f) {f.Views, f.Rewards, f.ResearchInfo.Resolved}, fs, 'UniformOutput', false);
            % Episode/Frame IDs advance across episodes, so compare the world content per frame
            tc.verifyEqual(cellfun(@(f) f.Rewards, f2, 'UniformOutput', false), cellfun(@(f) f.Rewards, f1, 'UniformOutput', false));
            tc.verifyEqual(strip(f3), strip(f1));
            ctx = @(fs) cellfun(@(f) f.Views{1}.Observation(1), fs);
            tc.verifyEqual(ctx(f2), ctx(f1));
            [~, cA] = tc.play(tc.make('Rounds', 30), 1, repmat([3 3 3], 30, 1));
            [~, cB] = tc.play(tc.make('Rounds', 30), 2, repmat([3 3 3], 30, 1));
            tc.verifyNotEqual(cA, cB);                             % a different seed gives a different world
        end

        function batchValidatedBeforeMutation(tc)
            A = [1 2 3; 2 2 1; 3 1 2];
            clean = tc.play(tc.make('Rounds', 3, 'ObstructionProb', 0.3), 6, A);
            env = tc.make('Rounds', 3, 'ObstructionProb', 0.3);
            f = env.reset(6);
            d = @(id, a, fid) OceanAgents.Core.Decision('AgentID', id, 'EpisodeID', 1, 'FrameID', fid, 'Action', a);
            mk = @(ds, fid) OceanAgents.Core.ActionBatch('EpisodeID', 1, 'FrameID', fid, 'Decisions', ds);
            tc.verifyError(@() env.step(tc.batch(f, [1 4 3])), 'OceanAgents:Core:IllegalAction');
            tc.verifyError(@() env.step(mk([d("alice", 1, f.FrameID) d("bob", 1, f.FrameID)], f.FrameID)), ...
                'OceanAgents:Core:MissingDecision');
            tc.verifyError(@() env.step(mk([d("alice", 1, f.FrameID) d("bob", 1, f.FrameID) ...
                d("carol", 1, f.FrameID) d("dave", 1, f.FrameID)], f.FrameID)), 'OceanAgents:Core:UnexpectedDecision');
            tc.verifyError(@() env.step(tc.batch(clean{2}, [1 1 1])), 'OceanAgents:Core:StaleFrame');
            tc.verifyError(@() env.step("batch"), 'OceanAgents:Core:InvalidRecord');
            tc.verifyEqual([env.Round env.FrameID], [0 1]);
            % no draw was consumed by the rejected batches: the rest of the episode equals the clean run
            for t = 1:3
                f = env.step(tc.batch(f, A(t, :)));
                tc.verifyEqual(f, clean{t + 1});
            end
        end

        function idsMonotonicAndLifecycle(tc)
            env = tc.make('Rounds', 2);
            tc.verifyError(@() env.step(42), 'OceanAgents:Env:NotReady');
            frameIDs = [];
            for e = 1:3
                f = env.reset(e);
                tc.verifyEqual(f.EpisodeID, e);
                frameIDs(end + 1) = f.FrameID; %#ok<AGROW>
                while ~f.EpisodeEnded
                    f = env.step(tc.batch(f, [1 1 1]));
                    frameIDs(end + 1) = f.FrameID; %#ok<AGROW>
                    tc.verifyEqual(tc.idsOf(f.Views{1}), sprintf("T/E%d-%d-alice", e, f.Tick));
                end
            end
            tc.verifyEqual(frameIDs, 1:9);
            env.close();
            env.close();
            tc.verifyError(@() env.reset(0), 'OceanAgents:Env:NotReady');
        end

        function describeAndPillars(tc)
            env = tc.make('Rounds', 10);
            spec = env.describe();
            tc.verifyEqual(spec.Timing, "simultaneous");
            tc.verifyEqual(spec.Roster, tc.Roster);
            tc.verifyEqual(spec.Roles, repmat("delegator", 1, 3));
            tc.verifyEqual(spec.Horizon, 10);
            tc.verifyEmpty(spec.Capabilities.list());
            tc.verifyEqual(spec.ActionSpaces{1}.N, 3);
            tc.verifyTrue(spec.ObservationSpaces{2}.contains([2 10]));
            tc.verifyTrue(spec.ObservationSpaces{2}.contains([0 11]));   % final view
            tc.verifyEqual(env.EpisodeID, 0);                    % describe does not start an episode
            native = OceanStats.RL.Env.Classic.RandomWalkEnv();
            meta = env.getMetadata();
            tc.verifyEqual(sort(fieldnames(meta)), sort(fieldnames(native.getMetadata())));
            tc.verifyFalse(meta.IsSymbolic);
            ps = env.getPipelineSpec();
            tc.verifyEqual(sort(fieldnames(ps)), sort(fieldnames(native.getPipelineSpec())));
            tc.verifyEqual(ps.Outputs.Domain, "Agent-DecisionFrame");
            tc.verifyTrue(isa(env, 'OceanAgents.Core.IJointEnvironment'));
        end

        function constructorChecks(tc)
            base = {'Roster', tc.Roster, 'Providers', tc.Providers, 'Contexts', tc.Contexts, 'Reliability', tc.R, ...
                'Payoffs', tc.payoffs()};
            make = @(varargin) OceanAgents.Env.Reference.DelegationEnv(varargin{:});
            id = 'OceanAgents:Core:InvalidRecord';
            tc.verifyError(@() make(base{1:8}), id);                                         % no Payoffs
            tc.verifyError(@() make(base{:}, 'Reliability', [0.9 1.2; 0 0]), id);
            tc.verifyError(@() make(base{:}, 'Reliability', [0.9; 0.1]), id);
            tc.verifyError(@() make(base{:}, 'ContextProbs', [0.5 0.6]), id);
            tc.verifyError(@() make(base{:}, 'Payoffs', struct('cheap', [1 2 3])), id);
            tc.verifyError(@() make(base{:}, 'Providers', ["p1" "outside"]), id);
            tc.verifyError(@() make(base{:}, 'Roster', ["a" "a"]), id);
            tc.verifyError(@() make(base{:}, 'Schedule', struct('FromRound', 0, 'Reliability', tc.R)), id);
            rep = struct('Round', 9, 'ReporterID', "x", 'SubjectID', "p1", 'Context', "cheap", ...
                'Outcome', "success", 'EvidenceID', "r");
            tc.verifyError(@() make(base{:}, 'Rounds', 3, 'Reports', rep), id);             % round beyond horizon
            rep.Round = 1;
            rep.EvidenceID = "E1-1-alice";
            tc.verifyError(@() make(base{:}, 'Reports', rep), id);                           % collides with direct IDs
            rep.EvidenceID = "r";
            rep.Outcome = "great";
            tc.verifyError(@() make(base{:}, 'Reports', rep), id);
        end

        function runsUnderJointRunner(tc)
            % Real runtime: three PolicyMind(RandomLegalPolicy) hosts; every round is paid within the declared payoffs.
            T = 8;
            env = tc.make('Rounds', T, 'ObstructionProb', 0.1, 'Visibility', "public");
            hosts = arrayfun(@(id) OceanAgents.Runtime.AgentHost(id, ...
                OceanAgents.Minds.PolicyMind(OceanAgents.Policies.RandomLegalPolicy())), tc.Roster, 'UniformOutput', false);
            r = OceanAgents.Runtime.JointRunner(env, hosts, 'Seed', 3).run(5);
            tc.assertEqual(r.Status, "Completed");
            tc.verifyEqual(r.EpisodeLengths, repmat(T, 5, 1));
            tc.verifyEqual(r.StepCalls, 5 * T);
            tc.verifyLessThanOrEqual(r.EpisodeReturns(:), T * 30);
            tc.verifyGreaterThanOrEqual(r.EpisodeReturns(:), T * -50);
        end
    end
end
