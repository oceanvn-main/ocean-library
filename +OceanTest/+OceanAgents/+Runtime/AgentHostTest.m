classdef AgentHostTest < OceanTest.BaseTestCase
    % AGENTHOSTTEST  OceanAgents.Runtime.AgentHost (A0 section 4.1): status machine, decision validation,
    % learning only inside observe(), LearningEnabled = false never calls the learner (C12), SharedComponent,
    % GammaMismatch, invalidate/release, export/import.

    methods (Static, Access = private)
        function v = view(frameID, obs, mask)
            if nargin < 3, mask = [true true]; end
            v = OceanAgents.Core.ActorView('AgentID', "alice", 'EpisodeID', 1, 'FrameID', frameID, ...
                'Observation', obs, 'ActionLabels', [1 2], 'ActionMask', mask);
        end
        function fb = feedback(frameID, reward, nextObs, terminated)
            mask = [true true];
            if terminated, mask = [false false]; end
            nv = OceanAgents.Core.ActorView('AgentID', "alice", 'EpisodeID', 1, 'FrameID', frameID + 1, ...
                'Observation', nextObs, 'ActionLabels', [1 2], 'ActionMask', mask);
            fb = OceanAgents.Core.Feedback('AgentID', "alice", 'EpisodeID', 1, 'FrameID', frameID, ...
                'Reward', reward, 'Terminated', terminated, 'NextView', nv);
        end
        function rs = stream()
            rs = RandStream.create('mrg32k3a', 'Seed', 1, 'NumStreams', 2, 'StreamIndices', 1);
        end
    end

    methods (Test)
        function statusMachineAndLearningOnlyInObserve(tc)
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            mind = ScriptedMind([2 2]);
            learner = CountingLearner();
            h = OceanAgents.Runtime.AgentHost("alice", mind, 'Learner', learner, 'Gamma', 1);
            tc.verifyEqual(h.Status, "Ready");
            tc.verifyError(@() h.decide(tc.view(1, 4)), 'OceanAgents:Runtime:InvalidState');   % not in episode
            h.beginEpisode(tc.view(1, 4));
            tc.verifyEqual(h.Status, "InEpisode");
            tc.verifyEqual(mind.BeginCalls, 1);
            tc.verifyError(@() h.decide(tc.view(1, 4)), 'OceanAgents:Runtime:InvalidState');   % no stream
            tc.verifyError(@() h.attachStream(tc.stream()), 'OceanAgents:Runtime:InvalidState');  % between episodes only
            tc.verifyError(@() h.beginEpisode(tc.view(1, 4)), 'OceanAgents:Runtime:InvalidState');
        end

        function fullEpisode(tc)
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            mind = ScriptedMind([2 2]);
            learner = CountingLearner('Gamma', 0.9);
            h = OceanAgents.Runtime.AgentHost("alice", mind, 'Learner', learner, 'Gamma', 0.9);
            h.attachStream(tc.stream());
            h.beginEpisode(tc.view(1, 5));
            d = h.decide(tc.view(1, 5));
            tc.verifyEqual(d.Action, 2);
            tc.verifyTrue(h.hasPendingDecision());
            tc.verifyEmpty(learner.Consumed, 'nothing is learned before the world transition');
            tc.verifyError(@() h.decide(tc.view(1, 5)), 'OceanAgents:Runtime:TransitionOrder');
            h.observe(tc.feedback(1, 0, 6, false));
            tc.verifyEqual(numel(learner.Consumed), 1);
            tc.verifyEqual(learner.UpdateCalls, "step");
            tc.verifyEqual(h.UpdateCount, 1);
            t = learner.Consumed{1};
            tc.verifyEqual([t.Observation t.Action t.NextObservation t.Discount], [5 2 6 0.9]);
            h.decide(tc.view(2, 6));
            h.observe(tc.feedback(2, 1, 7, true));
            tc.verifyEqual(h.Status, "Ready");
            tc.verifyEqual(learner.UpdateCalls, ["step" "step" "episode"]);
            tc.verifyEqual(h.UpdateCount, 2);
            tc.verifyEqual([mind.ObserveCalls mind.EndCalls], [2 1]);
        end

        function learningDisabledNeverCallsLearner(tc)
            % C12
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            learner = CountingLearner();
            h = OceanAgents.Runtime.AgentHost("alice", ScriptedMind(2), 'Learner', learner, ...
                'LearningEnabled', false);
            h.attachStream(tc.stream());
            h.beginEpisode(tc.view(1, 5));
            h.decide(tc.view(1, 5));
            h.observe(tc.feedback(1, 0, 6, false));
            h.decide(tc.view(2, 6));
            h.observe(tc.feedback(2, 1, 7, true));
            tc.verifyEmpty(learner.Consumed);
            tc.verifyEmpty(learner.UpdateCalls);
            tc.verifyEqual(h.UpdateCount, 0);
            h.setLearningEnabled(true);
            tc.verifyTrue(h.LearningEnabled);
        end

        function episodeBoundaryLearner(tc)
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            learner = CountingLearner('Boundary', "episode");
            h = OceanAgents.Runtime.AgentHost("alice", ScriptedMind(2), 'Learner', learner);
            h.attachStream(tc.stream());
            h.beginEpisode(tc.view(1, 5));
            h.decide(tc.view(1, 5));
            h.observe(tc.feedback(1, 0, 6, false));
            tc.verifyEmpty(learner.UpdateCalls, 'an episode-boundary learner is not asked at step boundaries');
            h.decide(tc.view(2, 6));
            h.observe(tc.feedback(2, 1, 7, true));
            tc.verifyEqual(learner.UpdateCalls, "episode");
            tc.verifyEqual(h.UpdateCount, 2);
        end

        function illegalDecisionRejected(tc)
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            h = OceanAgents.Runtime.AgentHost("alice", ScriptedMind(2, 'IllegalOnDecide', 1));
            h.attachStream(tc.stream());
            h.beginEpisode(tc.view(1, 5));
            tc.verifyError(@() h.decide(tc.view(1, 5)), 'OceanAgents:Core:IllegalAction');
            tc.verifyFalse(h.hasPendingDecision());
            % a masked-out label is illegal too
            h2 = OceanAgents.Runtime.AgentHost("alice", ScriptedMind(2));
            h2.attachStream(tc.stream());
            h2.beginEpisode(tc.view(1, 5, [true false]));
            tc.verifyError(@() h2.decide(tc.view(1, 5, [true false])), 'OceanAgents:Core:IllegalAction');
            % a view for another agent
            other = OceanAgents.Core.ActorView('AgentID', "bob", 'EpisodeID', 1, 'FrameID', 1, ...
                'Observation', 1, 'ActionLabels', [1 2], 'ActionMask', [true true]);
            h3 = OceanAgents.Runtime.AgentHost("alice", ScriptedMind(2));
            tc.verifyError(@() h3.beginEpisode(other), 'OceanAgents:Core:InvalidRecord');
        end

        function sharedComponentRejected(tc)
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            mind = ScriptedMind(1);
            learner = CountingLearner();
            h1 = OceanAgents.Runtime.AgentHost("a", mind, 'Learner', learner);
            tc.verifyError(@() OceanAgents.Runtime.AgentHost("b", mind), 'OceanAgents:Runtime:SharedComponent');
            tc.verifyError(@() OceanAgents.Runtime.AgentHost("b", ScriptedMind(1), 'Learner', learner), ...
                'OceanAgents:Runtime:SharedComponent');
            h1.release();
            tc.verifyEqual(h1.Status, "Invalid");
            h2 = OceanAgents.Runtime.AgentHost("b", mind, 'Learner', learner);   % allowed after release
            tc.verifyEqual(h2.Status, "Ready");
            % a destroyed host releases its components too
            delete(h2);
            h3 = OceanAgents.Runtime.AgentHost("c", mind);
            tc.verifyEqual(h3.AgentID, "c");
        end

        function gammaMismatch(tc)
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            tc.verifyError(@() OceanAgents.Runtime.AgentHost("a", ScriptedMind(1), ...
                'Learner', CountingLearner('Gamma', 0.9), 'Gamma', 1), 'OceanAgents:Runtime:GammaMismatch');
            h = OceanAgents.Runtime.AgentHost("a", ScriptedMind(1), ...
                'Learner', CountingLearner('Gamma', []), 'Gamma', 0.5);   % no Gamma reported: accepted
            tc.verifyEqual(h.Gamma, 0.5);
            tc.verifyError(@() OceanAgents.Runtime.AgentHost("a", ScriptedMind(1), 'Learner', 7), ...
                'OceanAgents:Core:InvalidRecord');
        end

        function invalidateIsTerminal(tc)
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            learner = CountingLearner();
            h = OceanAgents.Runtime.AgentHost("alice", ScriptedMind(2), 'Learner', learner);
            h.attachStream(tc.stream());
            h.beginEpisode(tc.view(1, 5));
            h.decide(tc.view(1, 5));
            h.invalidate("test");
            tc.verifyEqual(h.Status, "Invalid");
            tc.verifyFalse(h.hasPendingDecision(), 'the pending decision is dropped, never learned from');
            ID = 'OceanAgents:Runtime:InvalidState';
            tc.verifyError(@() h.observe(tc.feedback(1, 0, 6, false)), ID);
            tc.verifyError(@() h.decide(tc.view(1, 5)), ID);
            tc.verifyError(@() h.beginEpisode(tc.view(1, 5)), ID);
            tc.verifyError(@() h.setLearningEnabled(false), ID);
            tc.verifyError(@() h.attachStream(tc.stream()), ID);
            tc.verifyError(@() h.exportState(), ID);
            tc.verifyError(@() h.importState(struct()), ID);
            tc.verifyError(@() h.invalidate("again"), ID);
            tc.verifyEmpty(learner.Consumed);
        end

        function exportImportRoundTrip(tc)
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            mind = ScriptedMind(2);
            h = OceanAgents.Runtime.AgentHost("alice", mind, 'Learner', CountingLearner('Gamma', 0.8), 'Gamma', 0.8);
            h.attachStream(tc.stream());
            h.beginEpisode(tc.view(1, 5));
            tc.verifyError(@() h.exportState(), 'OceanAgents:Runtime:InvalidState');   % only between episodes
            h.decide(tc.view(1, 5));
            h.observe(tc.feedback(1, 1, 7, true));
            s = h.exportState();
            tc.verifyEqual([s.UpdateCount s.Learner.Applied s.Mind.DecideCalls], [1 1 1]);
            h2 = OceanAgents.Runtime.AgentHost("alice", ScriptedMind(2), ...
                'Learner', CountingLearner('Gamma', 0.8), 'Gamma', 0.8);
            h2.importState(s);
            tc.verifyEqual(h2.UpdateCount, 1);
            tc.verifyEqual(h2.Mind.DecideCalls, 1);
            h3 = OceanAgents.Runtime.AgentHost("bob", ScriptedMind(2), 'Gamma', 0.8);
            tc.verifyError(@() h3.importState(s), 'OceanAgents:Runtime:InvalidState');
        end

        function pillars(tc)
            h = OceanAgents.Runtime.AgentHost("alice", OceanTest.OceanAgents.Fixtures.Runtime.ScriptedMind(1));
            meta = h.getMetadata();
            tc.verifyFalse(meta.IsSymbolic);
            spec = h.getPipelineSpec();
            tc.verifyEqual([spec.Inputs.Domain], ["Agent-ActorView" "Agent-Feedback"]);
            tc.verifyEqual([spec.Outputs.Domain], ["Agent-Decision" "Agent-Transition"]);
        end
    end
end
