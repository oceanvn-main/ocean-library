classdef CliffWalkingLearningTest < OceanTest.BaseTestCase
    % CLIFFWALKINGLEARNINGTEST  End to end, ordinary single-agent RL through the agent stack: the existing
    % CliffWalking environment (+ TimeLimit) -> NativeSingleAgentBridge -> SingleAgentRunner -> AgentHost ->
    % PolicyMind(EpsilonGreedyQPolicy) with a OneStepQLearner sharing the agent's own TabularQ.
    %
    % Evidence, not self-consistency: the learned Q must equal an independent plain-arithmetic Q-learning replay
    % (Fixtures.Learning.referenceQLearning, no OceanStats.RL.TD code) of exactly the transitions the agent consumed.
    % Gates: C31, C33, AP49, AP50, IG05, IG06, C12, C01 (A0_Contract_Freeze.md section 10).

    properties (Constant)
        Alpha = 0.5
        Gamma = 1
        Epsilon = 0.1
        Horizon = 200
        Episodes = 60
    end

    methods (Static)
        function [host, runner, q, rec] = buildAgent(seed, learningEnabled)
            if nargin < 2, learningEnabled = true; end
            C = OceanTest.OceanAgents.Integration.CliffWalkingLearningTest;
            env = OceanStats.RL.Env.Classic.CliffWalkingEnv();
            bridge = OceanAgents.Env.NativeSingleAgentBridge(OceanStats.RL.Env.Wrappers.TimeLimit(env, C.Horizon));
            codec = OceanAgents.Artifacts.DiscreteStateCodec(48);
            q = OceanStats.RL.TD.Values.TabularQ(env.ValidMask, 0);
            pol = OceanAgents.Policies.EpsilonGreedyQPolicy(q, 1:4, 'Epsilon', C.Epsilon, 'StateCodec', codec);
            lrn = OceanAgents.Learners.OneStepQLearner(q, 1:4, 'Alpha', C.Alpha, 'Gamma', C.Gamma, 'StateCodec', codec);
            rec = OceanTest.OceanAgents.Fixtures.Integration.RecordingLearner(lrn);
            host = OceanAgents.Runtime.AgentHost("agent1", OceanAgents.Minds.PolicyMind(pol), 'Learner', rec, ...
                'Gamma', C.Gamma, 'LearningEnabled', learningEnabled);
            runner = OceanAgents.Runtime.SingleAgentRunner(bridge, host, 'Seed', seed);
        end

        function steps = toReferenceSteps(ts)
            % Transitions (external labels, observations) -> table indices for the reference replay.
            % CliffWalking: observation = state index 1..48; labels 1..4 = columns 1..4 (Discrete(4), Start 1).
            steps = repmat(struct('s', 0, 'a', 0, 'r', 0, 's2', 1, 'term', false, 'discount', 1, ...
                'valid2', true(1, 4)), 1, numel(ts));
            for k = 1:numel(ts)
                t = ts{k};
                steps(k).s = t.Observation;
                steps(k).a = t.Action;
                steps(k).r = t.Reward;
                steps(k).s2 = t.NextObservation;
                steps(k).term = t.Terminated;
                steps(k).discount = t.Discount;
                steps(k).valid2 = t.NextMask;
            end
        end
    end

    methods (Test)
        function learnedQEqualsIndependentReplay(tc)
            [host, runner, q, rec] = tc.buildAgent(7);
            r = runner.run(tc.Episodes);
            tc.assertEqual(r.Status, "Completed", 'run failed');
            tc.verifyEqual(r.NumEpisodes, tc.Episodes);
            % one rollout owner: exactly one environment step and one transition per decision (C33, AP50, IG05)
            tc.verifyEqual(r.StepCalls, sum(r.EpisodeLengths));
            tc.verifyEqual(numel(rec.Seen), r.StepCalls);
            tc.verifyEqual(r.UpdateCounts, r.StepCalls);
            tc.verifyEqual(sum(q.Counts(:)), r.StepCalls);
            % every episode ends exactly once: goal reached (terminated) or time limit (truncated + successor mask)
            ends = cellfun(@(t) t.Terminated || t.Truncated, rec.Seen);
            tc.verifyEqual(sum(ends), tc.Episodes);
            for t = rec.Seen(ends)
                if t{1}.Truncated
                    tc.verifyTrue(any(t{1}.NextMask), 'truncation must carry successor legality');
                end
            end
            % independent replay of the exact experience (IG06); tolerance 1e-12 per A0 section 1
            steps = tc.toReferenceSteps(rec.Seen);
            [Qref, Nref] = OceanTest.OceanAgents.Fixtures.Learning.referenceQLearning(zeros(48, 4), steps, tc.Alpha);
            tc.verifyEqual(q.Q, Qref, 'AbsTol', 1e-12);
            tc.verifyEqual(q.Counts, Nref);
            % the runner's episode returns equal the rewards the agent actually received
            tc.verifyEqual(sum(r.EpisodeReturns), sum(cellfun(@(t) t.Reward, rec.Seen)), 'AbsTol', 1e-9);
            tc.verifyEqual(host.Status, "Ready");
            tc.log(matlab.unittest.Verbosity.Terse, sprintf('mean return episodes 1-10: %.1f; last 10: %.1f', ...
                mean(r.EpisodeReturns(1:10)), mean(r.EpisodeReturns(end-9:end))));
        end

        function noSocialMachineryConstructed(tc)
            % C31/AP49: the ordinary loop declares no channel, ledger or privileged state; the mind is a bare policy
            [host, runner] = tc.buildAgent(1);
            spec = runner.Env.describe();
            social = intersect(spec.Capabilities.list(), ["Communication" "Commitments" "PrivilegedState"]);
            tc.verifyEmpty(social);
            tc.verifyEqual(spec.Timing, "single");
            d = host.Mind.describe();
            tc.verifyEqual(d.Profile, "reactive-policy");
            tc.verifyEqual(fieldnames(d.Components), {'Policy'});
        end

        function sameSeedSameRunFreshObjects(tc)
            [~, runA, qA] = tc.buildAgent(11);
            [~, runB, qB] = tc.buildAgent(11);
            rA = runA.run(15);
            rB = runB.run(15);
            tc.verifyEqual(rB.EpisodeReturns, rA.EpisodeReturns);
            tc.verifyEqual(rB.EpisodeLengths, rA.EpisodeLengths);
            tc.verifyEqual(qB.Q, qA.Q);
            [~, runC] = tc.buildAgent(12);
            rC = runC.run(15);
            tc.verifyNotEqual(rC.EpisodeLengths, rA.EpisodeLengths, 'a different run seed should change the run');
        end

        function inferenceOnlyChangesNoParameters(tc)
            % C12: acting and sampling still happen; learned parameters do not move
            [~, runner, q, rec] = tc.buildAgent(3, false);
            r = runner.run(5);
            tc.assertEqual(r.Status, "Completed");
            tc.verifyGreaterThan(r.StepCalls, 0);
            tc.verifyEqual(q.Q, zeros(48, 4));
            tc.verifyEqual(q.Counts, zeros(48, 4));
            tc.verifyEmpty(rec.Seen);
            tc.verifyEqual(r.UpdateCounts, 0);
        end

        function twoAgentsShareNoMutableState(tc)
            % C01/IG04 on real components: two agents built the same way; one trains, the other is untouched
            [hA, runA, qA] = tc.buildAgent(5);
            [hB, ~, qB] = tc.buildAgent(5);
            tc.verifyEmpty(OceanAgents.Runtime.IsolationChecker.sharedHandles(hA, hB));
            runA.run(3);
            tc.verifyNotEqual(qA.Q, zeros(48, 4));
            tc.verifyEqual(qB.Q, zeros(48, 4));
            tc.verifyEqual(qB.Counts, zeros(48, 4));
        end
    end
end
