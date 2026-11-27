classdef JointRunnerTest < OceanTest.BaseTestCase
    % JOINTRUNNERTEST  OceanAgents.Runtime.JointRunner (A0 section 4.5; C03, AP37, IG05): roster matching by
    % AgentID, one-frame decision collection before any step, feedback and learning only after the joint step
    % (slot order), evaluation order cannot change any committed action/reward/stream state, matching pennies
    % (A0 section 9.4) through the runner.

    methods (Test)
        function rosterMatchingByAgentID(tc)
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            env = ToyJointEnv('Roster', ["p1" "p2" "p3"], 'Rounds', 2);
            h1 = OceanAgents.Runtime.AgentHost("p1", ScriptedMind(1));
            h2 = OceanAgents.Runtime.AgentHost("p2", ScriptedMind(1));
            h3 = OceanAgents.Runtime.AgentHost("p3", ScriptedMind(2));
            runner = OceanAgents.Runtime.JointRunner(env, {h3, h1, h2});       % any order
            tc.verifyEqual(runner.AgentIDs, ["p1" "p2" "p3"]);
            tc.verifyTrue(runner.Hosts{3} == h3);
            rec = runner.run(1);
            tc.verifyEqual(rec.AgentIDs, ["p1" "p2" "p3"]);
            % coordination payoff: p1, p2 agree (+1 -1 = 0 each), p3 disagrees with both (-2) -> per round [0 0 -2]
            tc.verifyEqual(rec.EpisodeReturns, [0 0 -4]);
            tc.verifyEqual([rec.EpisodeLengths rec.StepCalls], [2 2]);
            % mismatches
            env2 = ToyJointEnv('Roster', ["p1" "p2"]);
            a = OceanAgents.Runtime.AgentHost("p1", ScriptedMind(1));
            b = OceanAgents.Runtime.AgentHost("x", ScriptedMind(1));
            c = OceanAgents.Runtime.AgentHost("p2", ScriptedMind(1));
            tc.verifyError(@() OceanAgents.Runtime.JointRunner(env2, {a, b}), 'OceanAgents:Runtime:RosterMismatch');
            tc.verifyError(@() OceanAgents.Runtime.JointRunner(env2, {a}), 'OceanAgents:Runtime:RosterMismatch');
            tc.verifyError(@() OceanAgents.Runtime.JointRunner(env2, {a, c, b}), 'OceanAgents:Runtime:RosterMismatch');
            tc.verifyError(@() OceanAgents.Runtime.JointRunner(env2, {a, c}, 'Order', [1 1]), ...
                'OceanAgents:Runtime:RosterMismatch');
        end

        function decideAllThenStepThenObserve(tc)
            % AP37/IG05: every decision of a frame is collected before the step; feedback and learning happen
            % only after the step, in slot order; learners never update before the joint transition.
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            log = EventLog();
            env = ToyJointEnv('Roster', ["p1" "p2" "p3"], 'Rounds', 2, 'Log', log);
            hosts = cell(1, 3);
            for k = 1:3
                hosts{k} = OceanAgents.Runtime.AgentHost("p" + k, FixedProbabilityMind([0.5 0.5], 'Log', log), ...
                    'Learner', CountingLearner('Log', log));
            end
            % the EventLog is shared ON PURPOSE between the fixtures of this test, so it is allowlisted
            rec = OceanAgents.Runtime.JointRunner(env, hosts, 'Seed', 2, 'Order', [2 3 1], ...
                'IsolationAllow', "OceanTest.OceanAgents.Fixtures.Runtime.EventLog").run(1);
            tc.assertEqual(rec.Status, "Completed");
            frame = ["decide:p2" "decide:p3" "decide:p1" "env:step" ...
                "consume:p1" "update:step" "observe:p1" ...
                "consume:p2" "update:step" "observe:p2" ...
                "consume:p3" "update:step" "observe:p3"];
            last = ["decide:p2" "decide:p3" "decide:p1" "env:step" ...
                "consume:p1" "update:step" "observe:p1" "update:episode" ...
                "consume:p2" "update:step" "observe:p2" "update:episode" ...
                "consume:p3" "update:step" "observe:p3" "update:episode"];
            tc.verifyEqual(log.Events, [frame last]);
            tc.verifyEqual(rec.UpdateCounts, [2 2 2]);
            % every batch answers exactly one frame, decisions in slot order
            for b = 1:numel(env.Batches)
                batch = env.Batches{b};
                tc.verifyEqual(batch.agentIDs(), ["p1" "p2" "p3"]);
                tc.verifyEqual(unique([batch.Decisions.FrameID]), batch.FrameID);
            end
        end

        function evaluationOrderChangesNothing(tc)
            % C03: permuting the decide order leaves every committed action, reward and stream state unchanged.
            orders = {[1 2 3], [3 1 2], [2 3 1]};
            results = cell(1, numel(orders));
            for o = 1:numel(orders)
                results{o} = runWithOrder(orders{o});
            end
            for o = 2:numel(orders)
                tc.verifyEqual(results{o}.Actions, results{1}.Actions);
                tc.verifyEqual(results{o}.Returns, results{1}.Returns);
                tc.verifyEqual(results{o}.StreamStates, results{1}.StreamStates);
                tc.verifyEqual(results{o}.Draws, results{1}.Draws);
                tc.verifyEqual(results{o}.Seeds, results{1}.Seeds);
            end
            tc.verifyEqual(results{2}.FirstDecider, "decide:p3", 'the permuted order must really be used');
            tc.verifyGreaterThan(numel(unique(results{1}.Actions)), 1, 'the mixed strategies must actually mix');

            function r = runWithOrder(order)
                import OceanTest.OceanAgents.Fixtures.Runtime.*
                log = EventLog();
                env = ToyJointEnv('Roster', ["p1" "p2" "p3"], 'Rounds', 5);
                probs = {[0.2 0.8], [0.5 0.5], [0.9 0.1]};
                hosts = cell(1, 3);
                minds = cell(1, 3);
                for k = 1:3
                    minds{k} = FixedProbabilityMind(probs{k}, 'Log', log);
                    hosts{k} = OceanAgents.Runtime.AgentHost("p" + k, minds{k});
                end
                runner = OceanAgents.Runtime.JointRunner(env, hosts, 'Seed', 99, 'Order', order, ...
                    'IsolationAllow', "OceanTest.OceanAgents.Fixtures.Runtime.EventLog");   % holds the streams
                rec = runner.run(4);
                actions = zeros(numel(env.Batches), 3);
                for b = 1:numel(env.Batches)
                    actions(b, :) = [env.Batches{b}.Decisions.Action];
                end
                r = struct('Actions', actions, 'Returns', rec.EpisodeReturns, ...
                    'StreamStates', {cellfun(@(h) h.Stream.State, hosts, 'UniformOutput', false)}, ...
                    'Draws', {cellfun(@(m) m.Draws, minds, 'UniformOutput', false)}, ...
                    'Seeds', env.Seeds, 'FirstDecider', log.Events(1));
            end
        end

        function matchingPenniesThroughRunner(tc)
            % A0 section 9.4: p1 plays label 1 w.p. 0.7, p2 always 1. E[u1] = 0.4, SE = sqrt(0.84/T);
            % band +-4 SE = +-0.0367 for T = 10000. u2 = -u1 exactly every round.
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            T = 10000;
            env = ToyJointEnv('Roster', ["p1" "p2"], 'Rounds', T, 'Payoff', "pennies");
            h1 = OceanAgents.Runtime.AgentHost("p1", FixedProbabilityMind([0.7 0.3]));
            h2 = OceanAgents.Runtime.AgentHost("p2", FixedProbabilityMind([1 0]));
            rec = OceanAgents.Runtime.JointRunner(env, {h1, h2}, 'Seed', 2026).run(1);
            tc.assertEqual(rec.Status, "Completed");
            tc.verifyEqual(rec.EpisodeLengths, T);
            meanU1 = rec.EpisodeReturns(1) / T;
            tc.verifyLessThanOrEqual(abs(meanU1 - 0.4), 4 * sqrt(0.84 / T));
            tc.verifyEqual(rec.EpisodeReturns(2), -rec.EpisodeReturns(1));
            a1 = cellfun(@(b) b.decisionFor("p1").Action, env.Batches);
            tc.verifyEqual(rec.EpisodeReturns(1), sum(2 * (a1 == 1) - 1), 'returns must be the sum of the payoffs played');
        end

        function pillars(tc)
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            env = ToyJointEnv();
            r = OceanAgents.Runtime.JointRunner(env, {OceanAgents.Runtime.AgentHost("p1", ScriptedMind(1)), ...
                OceanAgents.Runtime.AgentHost("p2", ScriptedMind(1))});
            tc.verifyFalse(r.getMetadata().IsSymbolic);
            tc.verifyEqual(numel(r.getPipelineSpec().ComposesFrom), 3);
        end
    end
end
