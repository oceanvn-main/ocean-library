classdef SingleAgentRunnerTest < OceanTest.BaseTestCase
    % SINGLEAGENTRUNNERTEST  OceanAgents.Runtime.SingleAgentRunner (A0 section 4.4; C31, C33, AP49, AP50, IG05):
    % the A0 section 9.1 RandomWalk trace through the runner, time-limit truncation, sole ownership of reset/step,
    % per-episode seeds from the "env" stream, the host's own action stream, determinism, RunRecord fields.

    methods (Test)
        function randomWalkHandTrace(tc)
            % A0 section 9.1: start 4, actions [2 2 2] -> 5, 6, 7; rewards 0 0 1; terminal on the 3rd step.
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            env = SpySingleEnv();
            learner = CountingLearner();
            host = OceanAgents.Runtime.AgentHost("agent1", ScriptedMind([2 2 2]), 'Learner', learner);
            rec = OceanAgents.Runtime.SingleAgentRunner(env, host, 'Seed', 1).run(1);
            tc.verifyEqual(rec.Status, "Completed");
            tc.verifyEqual([rec.NumEpisodes rec.StepCalls rec.EpisodeLengths rec.EpisodeReturns], [1 3 3 1]);
            tc.verifyEqual(rec.AgentIDs, "agent1");
            tc.verifyEqual(rec.UpdateCounts, 3);
            tc.verifyEqual(rec.FailureStage, "");
            tc.verifyEmpty(rec.Failure);
            t = [learner.Consumed{:}];
            tc.verifyEqual([t.Observation], [4 5 6]);
            tc.verifyEqual([t.Action], [2 2 2]);
            tc.verifyEqual([t.NextObservation], [5 6 7]);
            tc.verifyEqual([t.Reward], [0 0 1]);
            tc.verifyEqual([t.Terminated], [false false true]);
            tc.verifyEqual(t(3).NextMask, [false false]);                  % final view: all-false mask
            tc.verifyEqual(host.Status, "Ready");
        end

        function timeLimitTruncates(tc)
            % A0 section 9.1 with TimeLimit 2: truncated after 2 env steps, successor mask [true true].
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            env = SpySingleEnv('TimeLimit', 2);
            learner = CountingLearner();
            host = OceanAgents.Runtime.AgentHost("agent1", ScriptedMind([2 2 2]), 'Learner', learner);
            rec = OceanAgents.Runtime.SingleAgentRunner(env, host).run(1);
            tc.verifyEqual([rec.StepCalls env.StepCalls rec.EpisodeReturns], [2 2 0]);
            t = learner.Consumed{end};
            tc.verifyEqual([t.NextObservation t.Reward], [6 0]);
            tc.verifyFalse(t.Terminated);
            tc.verifyTrue(t.Truncated);
            tc.verifyEqual(t.NextMask, [true true]);
        end

        function runnerIsSoleOwnerOfResetAndStep(tc)
            % C31/C33: the environment is reset exactly once per episode and stepped exactly StepCalls times,
            % each episode seeded with the next draw of the "env" purpose stream.
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            env = SpySingleEnv();
            host = OceanAgents.Runtime.AgentHost("agent1", FixedProbabilityMind([0.5 0.5]));
            runner = OceanAgents.Runtime.SingleAgentRunner(env, host, 'Seed', 21);
            rec = runner.run(5);
            tc.verifyEqual(env.ResetCalls, 5);
            tc.verifyEqual(env.StepCalls, rec.StepCalls);
            tc.verifyEqual(sum(rec.EpisodeLengths), rec.StepCalls);
            ref = OceanAgents.Runtime.StreamRegistry(21);
            tc.verifyEqual(env.Seeds, arrayfun(@(~) ref.nextSeed("env"), 1:5));
            % the host draws from its own purpose stream
            tc.verifyEqual(double(host.Stream.StreamIndex), OceanAgents.Runtime.StreamRegistry.indexOf("agent:agent1:action"));
            tc.verifyTrue(host.Stream == runner.Registry.stream("agent:agent1:action"));
        end

        function deterministicGivenSeed(tc)
            % AP49/AP50: same seed -> identical run; different seed -> different draws. Global stream untouched.
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            g = RandStream.getGlobalStream();
            before = g.State;
            [rec1, m1] = runOnce(21);
            [rec2, m2] = runOnce(21);
            [~, m3] = runOnce(22);
            tc.verifyEqual(rec2.EpisodeLengths, rec1.EpisodeLengths);
            tc.verifyEqual(rec2.EpisodeReturns, rec1.EpisodeReturns);
            tc.verifyEqual(m2.Draws, m1.Draws);
            tc.verifyNotEqual(m3.Draws(1:5), m1.Draws(1:5));
            tc.verifyEqual(RandStream.getGlobalStream().State, before);

            function [rec, mind] = runOnce(seed)
                mind = FixedProbabilityMind([0.3 0.7]);
                h = OceanAgents.Runtime.AgentHost("agent1", mind);
                rec = OceanAgents.Runtime.SingleAgentRunner(SpySingleEnv(), h, 'Seed', seed).run(10);
            end
        end

        function repeatedRunsContinueStreams(tc)
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            env = SpySingleEnv();
            host = OceanAgents.Runtime.AgentHost("agent1", FixedProbabilityMind([0.5 0.5]));
            runner = OceanAgents.Runtime.SingleAgentRunner(env, host, 'Seed', 4);
            runner.run(2);
            runner.run(3);
            ref = OceanAgents.Runtime.StreamRegistry(4);
            tc.verifyEqual(env.Seeds, arrayfun(@(~) ref.nextSeed("env"), 1:5));
            tc.verifyEqual(runner.EpisodesRun, 5);
            rec0 = runner.run(0);
            tc.verifyEqual([rec0.NumEpisodes rec0.StepCalls], [0 0]);
            tc.verifyEqual(size(rec0.EpisodeReturns), [0 1]);
        end

        function learningDisabledThroughRunner(tc)
            % C12 through the loop: the learner is never called.
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            learner = CountingLearner();
            host = OceanAgents.Runtime.AgentHost("agent1", ScriptedMind(2), 'Learner', learner, 'LearningEnabled', false);
            rec = OceanAgents.Runtime.SingleAgentRunner(SpySingleEnv(), host).run(3);
            tc.verifyEqual(rec.StepCalls, 9);
            tc.verifyEqual(rec.UpdateCounts, 0);
            tc.verifyEmpty(learner.Consumed);
            tc.verifyEmpty(learner.UpdateCalls);
        end

        function rosterMismatch(tc)
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            host = OceanAgents.Runtime.AgentHost("someoneElse", ScriptedMind(2));
            tc.verifyError(@() OceanAgents.Runtime.SingleAgentRunner(SpySingleEnv(), host), ...
                'OceanAgents:Runtime:RosterMismatch');
        end

        function pillars(tc)
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            host = OceanAgents.Runtime.AgentHost("agent1", ScriptedMind(2));
            r = OceanAgents.Runtime.SingleAgentRunner(SpySingleEnv(), host);
            tc.verifyFalse(r.getMetadata().IsSymbolic);
            tc.verifyEqual(r.getPipelineSpec().Kind, "NONVIRTUAL");
        end
    end
end
