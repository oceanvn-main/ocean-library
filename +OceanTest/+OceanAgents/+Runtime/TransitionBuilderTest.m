classdef TransitionBuilderTest < OceanTest.BaseTestCase
    % TRANSITIONBUILDERTEST  OceanAgents.Runtime.TransitionBuilder (A0 section 4.2): field mapping, Discount =
    % gamma^Duration, successor masks for terminal/truncated feedback, and strict open/close ordering.

    methods (Static, Access = private)
        function v = view(frameID, obs, mask, info)
            if nargin < 4, info = struct(); end
            v = OceanAgents.Core.ActorView('AgentID', "a", 'EpisodeID', 1, 'FrameID', frameID, ...
                'Observation', obs, 'ActionLabels', [0 1 2], 'ActionMask', mask, 'Info', info);
        end
        function d = dec(frameID, action)
            p = [0.25 0 0.75];                         % over labels [0 1 2]; used with actions 0 and 2 only
            d = OceanAgents.Core.Decision('AgentID', "a", 'EpisodeID', 1, 'FrameID', frameID, 'Action', action, ...
                'Probabilities', p, 'Likelihood', p(action + 1), 'PolicyVersion', 3);
        end
    end

    methods (Test)
        function mapsDecisionAndFeedback(tc)
            tb = OceanAgents.Runtime.TransitionBuilder();
            v1 = tc.view(1, 37, [true false true]);
            tb.open(v1, tc.dec(1, 2));
            tc.verifyTrue(tb.IsOpen);
            v2 = tc.view(2, 25, [true true false]);
            fb = OceanAgents.Core.Feedback('AgentID', "a", 'EpisodeID', 1, 'FrameID', 1, 'Reward', -1, 'NextView', v2);
            t = tb.close(fb, 0.9);
            tc.verifyFalse(tb.IsOpen);
            tc.verifyEqual([t.Observation t.Action t.Reward t.NextObservation], [37 2 -1 25]);
            tc.verifyEqual(t.ActionLabels, [0 1 2]);
            tc.verifyEqual(t.NextActionLabels, [0 1 2]);
            tc.verifyEqual(t.NextMask, [true true false]);
            tc.verifyEqual([t.Duration t.Discount t.PolicyVersion t.Likelihood], [1 0.9 3 0.75]);
            tc.verifyFalse(t.Terminated || t.Truncated);
        end

        function terminalAndTruncatedSuccessorMasks(tc)
            tb = OceanAgents.Runtime.TransitionBuilder();
            % terminated: no successor legality
            tb.open(tc.view(1, 6, [true true true]), tc.dec(1, 2));
            ended = tc.view(2, 7, false(1, 3));
            t = tb.close(OceanAgents.Core.Feedback('AgentID', "a", 'EpisodeID', 1, 'FrameID', 1, 'Reward', 1, ...
                'Terminated', true, 'NextView', ended), 1);
            tc.verifyTrue(t.Terminated);
            tc.verifyEqual(t.NextMask, false(1, 3));
            % truncated with a declared successor mask: bootstrapping legality is carried
            tb.open(tc.view(3, 5, [true true true]), tc.dec(3, 0));
            trunc = tc.view(4, 6, false(1, 3), struct('successor_action_mask', [true false true]));
            t = tb.close(OceanAgents.Core.Feedback('AgentID', "a", 'EpisodeID', 1, 'FrameID', 3, ...
                'Truncated', true, 'NextView', trunc), 0.5);
            tc.verifyTrue(t.Truncated);
            tc.verifyEqual(t.NextMask, [true false true]);
            tc.verifyEqual(t.Discount, 0.5);
            % truncated without one: all false (a learner that must bootstrap then rejects it)
            tb.open(tc.view(5, 5, [true true true]), tc.dec(5, 0));
            t = tb.close(OceanAgents.Core.Feedback('AgentID', "a", 'EpisodeID', 1, 'FrameID', 5, ...
                'Truncated', true, 'NextView', tc.view(6, 6, false(1, 3))), 1);
            tc.verifyEqual(t.NextMask, false(1, 3));
        end

        function orderIsStrict(tc)
            tb = OceanAgents.Runtime.TransitionBuilder();
            v2 = tc.view(2, 25, [true true true]);
            fb = OceanAgents.Core.Feedback('AgentID', "a", 'EpisodeID', 1, 'FrameID', 1, 'NextView', v2);
            tc.verifyError(@() tb.close(fb, 1), 'OceanAgents:Runtime:TransitionOrder');
            tb.open(tc.view(1, 37, [true true true]), tc.dec(1, 2));
            tc.verifyError(@() tb.open(tc.view(1, 37, [true true true]), tc.dec(1, 2)), 'OceanAgents:Runtime:TransitionOrder');
            stale = OceanAgents.Core.Feedback('AgentID', "a", 'EpisodeID', 1, 'FrameID', 7, ...
                'NextView', tc.view(8, 25, [true true true]));                 % answers frame 7, not frame 1
            tc.verifyError(@() tb.close(stale, 1), 'OceanAgents:Core:StaleFrame');
            tc.verifyTrue(tb.IsOpen, 'a rejected close must leave the decision open');
            tb.close(fb, 1);
            tc.verifyError(@() tb.close(fb, 1), 'OceanAgents:Runtime:TransitionOrder');
        end

        function pillars(tc)
            tb = OceanAgents.Runtime.TransitionBuilder();
            tc.verifyFalse(tb.getMetadata().IsSymbolic);
            tc.verifyEqual(tb.getPipelineSpec().Outputs.Domain, "Agent-Transition");
        end
    end
end
