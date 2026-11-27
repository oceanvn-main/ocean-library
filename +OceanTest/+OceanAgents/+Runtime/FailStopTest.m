classdef FailStopTest < OceanTest.BaseTestCase
    % FAILSTOPTEST  Fail-stop semantics of both runners (A0 sections 4.4-4.5; C35 A1 slice, AP51, C17):
    % a fault at any stage returns a Failed RunRecord with that stage and the MException, nothing runs after the
    % fault (no further step, no learning from the failed frame), every host and the runner become Invalid,
    % and a second run() errors.

    methods (Test)
        function singleAgentFaultStages(tc)
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            % {env options, mind options, expected stage, expected identifier, env steps made, transitions learned}
            cases = {
                {{'ThrowOnReset', 1}, {}, "reset", 'Fixture:Injected', 0, 0}
                {{}, {'ThrowOnDecide', 2}, "decide", 'Fixture:Injected', 1, 1}
                {{}, {'IllegalOnDecide', 2}, "decide", 'OceanAgents:Core:IllegalAction', 1, 1}
                {{'ThrowOnStep', 2}, {}, "step", 'Fixture:Injected', 2, 1}
                {{}, {'ThrowOnObserve', 2}, "observe", 'Fixture:Injected', 2, 2}
                };
            for c = 1:numel(cases)
                [envOpts, mindOpts, stage, id, steps, learned] = cases{c}{:};
                env = SpySingleEnv(envOpts{:});
                learner = CountingLearner();
                host = OceanAgents.Runtime.AgentHost("agent1", ScriptedMind([2 2 2], mindOpts{:}), 'Learner', learner);
                runner = OceanAgents.Runtime.SingleAgentRunner(env, host);
                rec = runner.run(3);
                label = sprintf('case %d (%s)', c, stage);
                tc.verifyEqual(rec.Status, "Failed", label);
                tc.verifyEqual(rec.FailureStage, stage, label);
                tc.verifyEqual(rec.Failure.identifier, id, label);
                tc.verifyEqual([env.StepCalls rec.StepCalls], [steps steps], label);
                tc.verifyEqual(numel(learner.Consumed), learned, label);
                tc.verifyEqual(rec.NumEpisodes, 0, label);
                tc.verifyEqual(host.Status, "Invalid", label);
                tc.verifyEqual(runner.Status, "Invalid", label);
                tc.verifyError(@() runner.run(1), 'OceanAgents:Runtime:InvalidState');
                tc.verifyEqual(env.StepCalls, steps, 'a refused second run must not touch the environment');
            end
        end

        function completedEpisodesAreKept(tc)
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            env = SpySingleEnv('ThrowOnReset', 3);
            host = OceanAgents.Runtime.AgentHost("agent1", ScriptedMind(2));
            rec = OceanAgents.Runtime.SingleAgentRunner(env, host).run(5);
            tc.verifyEqual(rec.FailureStage, "reset");
            tc.verifyEqual(rec.NumEpisodes, 2);
            tc.verifyEqual(rec.EpisodeLengths, [3; 3]);
            tc.verifyEqual(rec.EpisodeReturns, [1; 1]);
            tc.verifyEqual(rec.StepCalls, 6);
        end

        function stepBudgetIsAFaultNotATruncation(tc)
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            env = SpySingleEnv('Terminals', false);
            learner = CountingLearner();
            host = OceanAgents.Runtime.AgentHost("agent1", ScriptedMind([1 2]), 'Learner', learner);
            rec = OceanAgents.Runtime.SingleAgentRunner(env, host, 'MaxStepsPerEpisode', 5).run(1);
            tc.verifyEqual(rec.Status, "Failed");
            tc.verifyEqual(rec.FailureStage, "step");
            tc.verifyEqual(rec.Failure.identifier, 'OceanAgents:Runtime:StepBudgetExceeded');
            tc.verifyEqual(env.StepCalls, 5);
            t = [learner.Consumed{:}];
            tc.verifyFalse(any([t.Truncated]), 'the budget must not be disguised as a truncation');
        end

        function invalidHostCannotBeReused(tc)
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            host = OceanAgents.Runtime.AgentHost("agent1", ScriptedMind(2, 'ThrowOnDecide', 1));
            OceanAgents.Runtime.SingleAgentRunner(SpySingleEnv(), host).run(1);
            tc.verifyError(@() OceanAgents.Runtime.SingleAgentRunner(SpySingleEnv(), host), ...
                'OceanAgents:Runtime:InvalidState');
        end

        function jointFaultStages(tc)
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            % {env options, p2 mind options, stage, identifier, env steps, transitions learned by p1}
            cases = {
                {{}, {'ThrowOnDecide', 2}, "decide", 'Fixture:Injected', 1, 1}
                {{}, {'IllegalOnDecide', 1}, "decide", 'OceanAgents:Core:IllegalAction', 0, 0}
                {{'ThrowOnStep', 2}, {}, "step", 'Fixture:Injected', 2, 1}
                {{}, {'ThrowOnObserve', 1}, "observe", 'Fixture:Injected', 1, 1}
                {{'SkipEndingFor', "p2"}, {}, "observe", 'OceanAgents:Runtime:InvalidState', 3, 3}
                };
            for c = 1:numel(cases)
                [envOpts, mindOpts, stage, id, steps, learned] = cases{c}{:};
                env = ToyJointEnv('Rounds', 3, envOpts{:});
                learner = CountingLearner();
                h1 = OceanAgents.Runtime.AgentHost("p1", ScriptedMind(1), 'Learner', learner);
                h2 = OceanAgents.Runtime.AgentHost("p2", ScriptedMind(1, mindOpts{:}));
                runner = OceanAgents.Runtime.JointRunner(env, {h1, h2});
                rec = runner.run(2);
                label = sprintf('joint case %d (%s)', c, stage);
                tc.verifyEqual(rec.Status, "Failed", label);
                tc.verifyEqual(rec.FailureStage, stage, label);
                tc.verifyEqual(rec.Failure.identifier, id, label);
                tc.verifyEqual([env.StepCalls rec.StepCalls], [steps steps], label);
                tc.verifyEqual(numel(learner.Consumed), learned, label);
                tc.verifyEqual([h1.Status h2.Status], ["Invalid" "Invalid"], label);
                tc.verifyEqual(runner.Status, "Invalid", label);
                tc.verifyError(@() runner.run(1), 'OceanAgents:Runtime:InvalidState');
            end
        end

        function jointFaultBeforeStepMeansNoStep(tc)
            % The first slot decided, the second threw: the environment is never stepped with a partial batch.
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            env = ToyJointEnv('Rounds', 3);
            m1 = ScriptedMind(1);
            h1 = OceanAgents.Runtime.AgentHost("p1", m1);
            h2 = OceanAgents.Runtime.AgentHost("p2", ScriptedMind(1, 'ThrowOnDecide', 1));
            rec = OceanAgents.Runtime.JointRunner(env, {h1, h2}).run(1);
            tc.verifyEqual(rec.FailureStage, "decide");
            tc.verifyEqual(m1.DecideCalls, 1);
            tc.verifyEqual(env.StepCalls, 0);
            tc.verifyEmpty(env.Batches);
        end

        function jointFrameBudget(tc)
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            env = ToyJointEnv('Rounds', 10);
            runner = OceanAgents.Runtime.JointRunner(env, {OceanAgents.Runtime.AgentHost("p1", ScriptedMind(1)), ...
                OceanAgents.Runtime.AgentHost("p2", ScriptedMind(2))}, 'MaxFramesPerEpisode', 3);
            rec = runner.run(1);
            tc.verifyEqual(rec.FailureStage, "step");
            tc.verifyEqual(rec.Failure.identifier, 'OceanAgents:Runtime:StepBudgetExceeded');
            tc.verifyEqual(env.StepCalls, 3);
        end

        function runRecordValidation(tc)
            tc.verifyError(@() OceanAgents.Runtime.RunRecord('Status', "Failed", 'AgentIDs', "a"), ...
                'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() OceanAgents.Runtime.RunRecord('Status', "Failed", 'AgentIDs', "a", ...
                'FailureStage', "nowhere", 'Failure', MException('a:b', 'c')), 'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() OceanAgents.Runtime.RunRecord('AgentIDs', ["a" "b"], 'EpisodeLengths', 3, ...
                'EpisodeReturns', [1 2 3]), 'OceanAgents:Core:InvalidRecord');
            r = OceanAgents.Runtime.RunRecord('AgentIDs', ["a" "b"], 'EpisodeLengths', [3; 4]);
            tc.verifyEqual(r.EpisodeReturns, zeros(2, 2));
            tc.verifyEqual(r.UpdateCounts, [0 0]);
            tc.verifyFalse(r.failed());
        end
    end
end
