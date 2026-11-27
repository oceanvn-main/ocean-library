classdef PolicyMindTest < OceanTest.BaseTestCase
    % POLICYMINDTEST  OceanAgents.Minds.PolicyMind (A0 section 6.6): a thin IDecisionProcess that delegates every
    % call to exactly one IPolicy and owns nothing else.

    methods (Static, Access = private)
        function v = view(frameID)
            v = OceanAgents.Core.ActorView('AgentID', "alice", 'EpisodeID', 1, 'FrameID', frameID, ...
                'Observation', 37, 'ActionLabels', 1:4, 'ActionMask', [true true true false]);
        end
        function pol = policy()
            q = OceanStats.RL.TD.Values.TabularQ(true(48, 4));
            q.applyIncrement(full(sparse(37, 1:4, [-0.5 -50 0 0], 48, 4)));
            pol = OceanAgents.Policies.EpsilonGreedyQPolicy(q, 1:4, 'Epsilon', 0.1);
        end
    end

    methods (Test)
        function decideIsThePolicysAct(tc)
            % Same stream seed -> the mind's decisions are exactly the policy's.
            pol = tc.policy();
            mind = OceanAgents.Minds.PolicyMind(pol);
            rsA = RandStream('mrg32k3a', 'Seed', 5);
            rsB = RandStream('mrg32k3a', 'Seed', 5);
            for k = 1:40
                v = tc.view(k);
                dm = mind.decide(v, rsA);
                dp = pol.act(v, rsB);
                tc.verifyEqual(dm, dp);
                dm.validateAgainst(v);
            end
        end

        function describeProfile(tc)
            mind = OceanAgents.Minds.PolicyMind(tc.policy());
            d = mind.describe();
            tc.verifyEqual(d.Profile, "reactive-policy");
            tc.verifyEqual(d.Components.Policy.Name, "EpsilonGreedyQPolicy");
        end

        function ownsOnlyItsPolicy(tc)
            % No beliefs, plans or peers: the only property is the policy.
            mind = OceanAgents.Minds.PolicyMind(OceanAgents.Policies.RandomLegalPolicy());
            tc.verifyEqual(string(properties(mind)), "Policy");
            mc = metaclass(mind);
            tc.verifyEqual(numel(mc.PropertyList), 1);
            tc.verifyClass(mind.Policy, 'OceanAgents.Policies.RandomLegalPolicy');
        end

        function delegatesLifecycleAndState(tc)
            pol = tc.policy();
            mind = OceanAgents.Minds.PolicyMind(pol);
            v = tc.view(1);
            mind.beginEpisode(v);
            fb = OceanAgents.Core.Feedback('AgentID', "alice", 'EpisodeID', 1, 'FrameID', 1, 'Reward', -1, ...
                'NextView', tc.view(2));
            mind.observe(fb);
            mind.endEpisode();
            s = mind.exportState();
            tc.verifyEqual(s.Policy, pol.exportState());
            other = OceanAgents.Minds.PolicyMind(OceanAgents.Policies.EpsilonGreedyQPolicy( ...
                OceanStats.RL.TD.Values.TabularQ(true(48, 4)), 1:4, 'Epsilon', 0.7));
            other.importState(s);
            tc.verifyEqual(other.Policy.Epsilon, 0.1);
            tc.verifyError(@() other.importState(struct('Name', "x")), 'OceanAgents:Policies:Incompatible');
            obsSp = OceanStats.RL.Env.Spaces.Discrete(48);
            mind.checkCompatible(obsSp, OceanStats.RL.Env.Spaces.Discrete(4), "default");
            tc.verifyError(@() mind.checkCompatible(obsSp, OceanStats.RL.Env.Spaces.Discrete(3), "default"), ...
                'OceanAgents:Policies:Incompatible');
            tc.verifyError(@() OceanAgents.Minds.PolicyMind(42), 'MATLAB:validation:UnableToConvert');
        end

        function pillars(tc)
            mind = OceanAgents.Minds.PolicyMind(OceanAgents.Policies.RandomLegalPolicy());
            m = mind.getMetadata();
            tc.verifyFalse(m.IsSymbolic);
            s = mind.getPipelineSpec();
            tc.verifyEqual([s.Inputs.Domain], ["Agent-ActorView" "Agent-Feedback"]);
            tc.verifyEqual(s.Outputs.Domain, "Agent-Decision");
            tc.verifyEqual(s.ComposesFrom, {"OceanAgents.Policies.RandomLegalPolicy"});
        end
    end
end
