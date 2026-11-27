classdef PolicySelectionTest < OceanTest.BaseTestCase
    % POLICYSELECTIONTEST  Design D29: a mind holds several policies; settings (FixedSelector / usePolicy) choose which
    % acts today, and any IPolicySelector (later a learned one) can choose from the situation. Learners are given to
    % the mind, never to the host (D27).

    methods (Static)
        function [mind, view] = setup()
            labels = string(1:4);
            q = OceanStats.RL.TD.Values.TabularQ(true(10, 4), 0);
            T = q.Q;  T(:, 4) = 1;  q.setTables(T, zeros(size(T)));          % greedy = move 4 everywhere
            mind = OceanAgents.Minds.QSignalMind(q, 1:4, 'Epsilon', 0);
            mind.addPolicy("fixed", OceanAgents.Policies.TablePolicy(OceanAgents.Artifacts.PolicyArtifact( ...
                'Kind', "ActionTable", 'Representation', "Label", 'Table', repmat(2, 10, 1), 'ActionLabels', 1:4)));
            OceanAgents.Embodied.Dualism.form(mind, OceanAgents.Body.EmptyBody(labels));
            view = @(tick) OceanAgents.Core.ActorView('AgentID', "a", 'EpisodeID', 1, 'FrameID', tick + 1, ...
                'Tick', tick, 'Observation', 3, 'ActionLabels', 1:4, 'ActionMask', true(1, 4));
        end
    end

    methods (Test)
        function settingsChooseThePolicy(tc)
            [mind, view] = tc.setup();
            tc.verifyEqual(mind.policyNames(), ["egreedy" "fixed"]);
            mind.beginEpisode(view(0));
            d = mind.decide(view(0), RandStream('mrg32k3a', 'Seed', 1));
            tc.verifyEqual([d.Action, string(mind.actedPolicy())], ["4" "egreedy"]);   % default: the first policy
            tc.verifyError(@() mind.usePolicy("fixed"), 'OceanAgents:Embodied:MidEpisode');
            mind.endEpisode();
            mind.usePolicy("fixed");
            mind.beginEpisode(view(0));
            d = mind.decide(view(0), RandStream('mrg32k3a', 'Seed', 1));
            tc.verifyEqual([d.Action, string(mind.actedPolicy())], ["2" "fixed"]);
            mind.endEpisode();
            mind.usePolicy("nonexistent");
            mind.beginEpisode(view(0));
            tc.verifyError(@() mind.decide(view(0), RandStream('mrg32k3a', 'Seed', 1)), 'OceanAgents:Minds:UnknownPolicy');
        end

        function aSelectorCanChooseFromTheSituation(tc)
            [mind, view] = tc.setup();
            mind.useSelector(OceanTest.OceanAgents.Fixtures.Signals.TickSelector());
            mind.beginEpisode(view(0));
            acts = zeros(1, 4);  who = strings(1, 4);
            for t = 0:3
                d = mind.decide(view(t), RandStream('mrg32k3a', 'Seed', t + 1));
                acts(t + 1) = d.Action;  who(t + 1) = mind.actedPolicy();
            end
            tc.verifyEqual(acts, [4 2 4 2]);
            tc.verifyEqual(who, ["egreedy" "fixed" "egreedy" "fixed"]);
        end

        function learnersBelongToTheMind(tc)
            q = OceanStats.RL.TD.Values.TabularQ(true(4, 2), 0);
            learner = OceanAgents.Learners.OneStepQLearner(q, 1:2, 'Alpha', 0.5);
            tc.verifyError(@() OceanAgents.Runtime.AgentHost("a", OceanAgents.Minds.QSignalMind(q, 1:2), ...
                'Learner', learner), 'OceanAgents:Runtime:LearnerInMind');
            mind = OceanAgents.Minds.QSignalMind(q, 1:2);
            mind.addLearner(learner, "q");
            host = OceanAgents.Runtime.AgentHost("a", mind);
            tc.verifyEmpty(host.Learner);
            tc.verifyEqual(mind.numLearners(), 1);
        end
    end
end
