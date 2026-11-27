classdef EmbodiedRegressionTest < OceanTest.BaseTestCase
    % EMBODIEDREGRESSIONTEST  AB1 success tests S1, S2, S6: the A1 Q-learning agent on cliff walking rebuilt as
    % mind -> body -> world (learner inside the mind, values in a TableMemory, actions through the body) gives a run
    % IDENTICAL to today's PolicyMind agent, with an empty body and with a grid robot body; the cost is reported.

    methods (Static)
        function [rec, q, seconds] = runAgent(kind, episodes)
            env = OceanStats.RL.Env.Classic.CliffWalkingEnv();
            bridge = OceanAgents.Env.NativeSingleAgentBridge(OceanStats.RL.Env.Wrappers.TimeLimit(env, 200));
            codec = OceanAgents.Artifacts.DiscreteStateCodec(48);
            q = OceanStats.RL.TD.Values.TabularQ(env.ValidMask, 0);
            learner = OceanAgents.Learners.OneStepQLearner(q, 1:4, 'Alpha', 0.5, 'Gamma', 1, 'StateCodec', codec);
            if kind == "policy"
                policy = OceanAgents.Policies.EpsilonGreedyQPolicy(q, 1:4, 'Epsilon', 0.1, 'StateCodec', codec);
                host = OceanAgents.Runtime.AgentHost("agent1", OceanAgents.Minds.PolicyMind(policy), ...
                    'Learner', learner, 'Gamma', 1);
            else
                mind = OceanAgents.Minds.QSignalMind(q, 1:4, 'Epsilon', 0.1, 'StateCodec', codec);
                if kind == "empty", body = OceanAgents.Body.EmptyBody(string(1:4));
                else, body = OceanAgents.Body.GridRobotBody(string(1:4)); end
                OceanAgents.Embodied.Dualism.form(mind, body);
                mind.addLearner(learner, "q");
                host = OceanAgents.Runtime.AgentHost("agent1", mind, 'Gamma', 1);
            end
            runner = OceanAgents.Runtime.SingleAgentRunner(bridge, host, 'Seed', 7);
            tic;
            rec = runner.run(episodes);
            seconds = toc;
        end
    end

    methods (Test)
        function emptyAndRobotBodiesReproduceTheA1Run(tc)
            [ref, qRef, tRef] = tc.runAgent("policy", 60);
            tc.verifyEqual(mean(ref.EpisodeReturns(1:10)), -290.8, 'AbsTol', 1e-9);      % recipe 1
            tc.verifyEqual(mean(ref.EpisodeReturns(end-9:end)), -124.0, 'AbsTol', 1e-9);
            for kind = ["empty" "robot"]
                [rec, q, t] = tc.runAgent(kind, 60);
                tc.verifyEqual(rec.Status, "Completed", kind);
                tc.verifyEqual(rec.EpisodeReturns, ref.EpisodeReturns, kind);
                tc.verifyEqual(rec.EpisodeLengths, ref.EpisodeLengths, kind);
                tc.verifyEqual(rec.StepCalls, ref.StepCalls, kind);
                tc.verifyEqual(q.Q, qRef.Q, kind);
                tc.verifyEqual(rec.UpdateCounts, ref.UpdateCounts, kind);
                tc.log(matlab.unittest.Verbosity.Terse, sprintf(['S6 %s body: %.3f ms/step vs PolicyMind %.3f ms/step ' ...
                    '(x%.2f), %d steps'], kind, 1e3 * t / rec.StepCalls, 1e3 * tRef / ref.StepCalls, t / tRef, rec.StepCalls));
                tc.verifyLessThan(t / tRef, 10, 'the body layer must not cost an order of magnitude');
            end
        end

        function learningDisabledMeansNoUpdates(tc)
            env = OceanStats.RL.Env.Classic.CliffWalkingEnv();
            codec = OceanAgents.Artifacts.DiscreteStateCodec(48);
            q = OceanStats.RL.TD.Values.TabularQ(env.ValidMask, 0);
            mind = OceanAgents.Minds.QSignalMind(q, 1:4, 'Epsilon', 0.1, 'StateCodec', codec);
            OceanAgents.Embodied.Dualism.form(mind, OceanAgents.Body.EmptyBody(string(1:4)));
            learner = OceanAgents.Learners.OneStepQLearner(q, 1:4, 'Alpha', 0.5, 'Gamma', 1, 'StateCodec', codec);
            mind.addLearner(learner, "q");
            host = OceanAgents.Runtime.AgentHost("agent1", mind, 'Gamma', 1, 'LearningEnabled', false);
            tc.verifyEmpty(host.Learner);                                  % the learner lives in the mind
            tc.verifyTrue(mind.hasLearners());
            runner = OceanAgents.Runtime.SingleAgentRunner( ...
                OceanAgents.Env.NativeSingleAgentBridge(OceanStats.RL.Env.Wrappers.TimeLimit(env, 50)), host, 'Seed', 3);
            rec = runner.run(3);
            tc.verifyEqual(rec.Status, "Completed");
            tc.verifyEqual(learner.NumUpdates, 0);
            tc.verifyEqual(q.Q, OceanStats.RL.TD.Values.TabularQ(env.ValidMask, 0).Q);
            q2 = OceanStats.RL.TD.Values.TabularQ(env.ValidMask, 0);
            mind2 = OceanAgents.Minds.QSignalMind(q2, 1:4);
            mind2.addLearner(OceanAgents.Learners.OneStepQLearner(q2, 1:4, 'Alpha', 0.5, 'Gamma', 1), "q");
            tc.verifyError(@() OceanAgents.Runtime.AgentHost("agent2", mind2, 'Gamma', 0.9), 'OceanAgents:Runtime:GammaMismatch');
            tc.verifyError(@() OceanAgents.Runtime.AgentHost("agent3", OceanAgents.Minds.QSignalMind(q2, 1:4), ...
                'Learner', learner), 'OceanAgents:Runtime:LearnerInMind');   % learners belong to the mind (D27)
        end
    end
end
