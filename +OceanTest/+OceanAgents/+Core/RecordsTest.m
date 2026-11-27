classdef RecordsTest < OceanTest.BaseTestCase
    % RECORDSTEST  OceanAgents.Core value records: validation, decision legality, batch validation, feedback rules.
    % Expected values come from Planning/System/Game Theory/A0_Contract_Freeze.md sections 2 and 9.5.

    methods (Static, Access = private)
        function v = view3(frameID)
            % Discrete(3, Start = 0) labels, middle action illegal (A0 section 9.5).
            if nargin < 1, frameID = 1; end
            v = OceanAgents.Core.ActorView('AgentID', "alice", 'EpisodeID', 1, 'FrameID', frameID, ...
                'Observation', 5, 'ActionLabels', [0 1 2], 'ActionMask', [true false true]);
        end
        function d = dec(action, varargin)
            d = OceanAgents.Core.Decision('AgentID', "alice", 'EpisodeID', 1, 'FrameID', 1, 'Action', action, varargin{:});
        end
        function f = frame2(requested)
            views = cell(1, 2);
            ids = ["alice" "bob"];
            for k = find(requested)
                views{k} = OceanAgents.Core.ActorView('AgentID', ids(k), 'EpisodeID', 1, 'FrameID', 4, ...
                    'Observation', [0 0], 'ActionLabels', [1 2], 'ActionMask', [true true]);
            end
            f = OceanAgents.Core.DecisionFrame('EpisodeID', 1, 'FrameID', 4, 'Roster', ids, ...
                'Live', [true true], 'Requested', requested, 'Views', views);
        end
        function d = decFor(id, action, frameID)
            d = OceanAgents.Core.Decision('AgentID', id, 'EpisodeID', 1, 'FrameID', frameID, 'Action', action);
        end
    end

    methods (Test)
        function actorViewBasics(tc)
            v = tc.view3();
            tc.verifyEqual(v.legalLabels(), [0 2]);
            tc.verifyTrue(v.isLegal(2));
            tc.verifyFalse(v.isLegal(1));
            tc.verifyFalse(v.isLegal(7));
            tc.verifyEqual(v.columnOf(2), 3);
            tc.verifyFalse(v.isTerminalView());
            tc.verifyEqual(v.Phase, "act");
        end

        function actorViewRejectsBadInput(tc)
            base = {'AgentID', "a", 'EpisodeID', 1, 'FrameID', 1, 'ActionLabels', [1 2], 'ActionMask', [true true]};
            tc.verifyError(@() OceanAgents.Core.ActorView(base{:}, 'Observation', NaN), 'OceanAgents:Core:NonFinite');
            tc.verifyError(@() OceanAgents.Core.ActorView('AgentID', "a", 'EpisodeID', 1, 'FrameID', 1, ...
                'Observation', 1, 'ActionLabels', [1 1], 'ActionMask', [true true]), 'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() OceanAgents.Core.ActorView('AgentID', "a", 'EpisodeID', 1, 'FrameID', 1, ...
                'Observation', 1, 'ActionLabels', [1 2], 'ActionMask', true), 'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() OceanAgents.Core.ActorView(base{:}), 'OceanAgents:Core:InvalidRecord');   % no Observation
            tc.verifyError(@() OceanAgents.Core.ActorView(base{1:4}, 'FrameID', -1, base{7:end}, 'Observation', 1), ...
                'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() OceanAgents.Core.ActorView(base{:}, 'Observation', 1, 'Info', struct('h', containers.Map())), ...
                'OceanAgents:Core:InvalidRecord');                                                        % handle in Info
        end

        function decisionLegalityHandCases(tc)
            % A0 section 9.5
            v = tc.view3();
            tc.verifyError(@() tc.dec(1).validateAgainst(v), 'OceanAgents:Core:IllegalAction');
            tc.dec(2, 'Probabilities', [0.5 0 0.5], 'Likelihood', 0.5).validateAgainst(v);   % valid: no error
            tc.verifyError(@() tc.dec(2, 'Probabilities', [0.5 0.1 0.4]).validateAgainst(v), 'OceanAgents:Core:IllegalAction');
            tc.verifyError(@() tc.dec(2, 'Probabilities', [0.5 0 0.5], 'Likelihood', 0.4).validateAgainst(v), ...
                'OceanAgents:Core:InvalidRecord');
        end

        function decisionRejectsBadProbabilities(tc)
            tc.verifyError(@() tc.dec(2, 'Probabilities', [0.5 NaN 0.5]), 'OceanAgents:Core:NonFinite');
            tc.verifyError(@() tc.dec(2, 'Probabilities', [0.6 0 0.5]), 'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() tc.dec(2, 'Likelihood', 0), 'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() tc.dec(NaN), 'OceanAgents:Core:NonFinite');
            tc.verifyFalse(tc.dec(2).hasLikelihood());
        end

        function decisionStaleOrWrongAgent(tc)
            v = tc.view3(2);
            tc.verifyError(@() tc.dec(2).validateAgainst(v), 'OceanAgents:Core:StaleFrame');
            other = OceanAgents.Core.Decision('AgentID', "bob", 'EpisodeID', 1, 'FrameID', 2, 'Action', 2);
            tc.verifyError(@() other.validateAgainst(v), 'OceanAgents:Core:InvalidRecord');
        end

        function feedbackRules(tc)
            ended = OceanAgents.Core.ActorView('AgentID', "alice", 'EpisodeID', 1, 'FrameID', 2, ...
                'Observation', 7, 'ActionLabels', [1 2], 'ActionMask', [false false]);
            live = OceanAgents.Core.ActorView('AgentID', "alice", 'EpisodeID', 1, 'FrameID', 2, ...
                'Observation', 6, 'ActionLabels', [1 2], 'ActionMask', [true true]);
            fb = OceanAgents.Core.Feedback('AgentID', "alice", 'EpisodeID', 1, 'FrameID', 1, 'Reward', 1, ...
                'Terminated', true, 'NextView', ended);
            tc.verifyTrue(fb.ended());
            tc.verifyEqual(fb.successorMask(), [false false]);
            tc.verifyError(@() OceanAgents.Core.Feedback('AgentID', "alice", 'EpisodeID', 1, 'FrameID', 1, ...
                'Terminated', true, 'Truncated', true, 'NextView', ended), 'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() OceanAgents.Core.Feedback('AgentID', "alice", 'EpisodeID', 1, 'FrameID', 1, ...
                'Terminated', true, 'NextView', live), 'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() OceanAgents.Core.Feedback('AgentID', "alice", 'EpisodeID', 1, 'FrameID', 1, ...
                'Reward', 3, 'RewardValid', false, 'NextView', live), 'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() OceanAgents.Core.Feedback('AgentID', "alice", 'EpisodeID', 1, 'FrameID', 1, ...
                'Reward', Inf, 'NextView', live), 'OceanAgents:Core:NonFinite');
            % truncation with declared successor legality (A0 section 9.1, TimeLimit case)
            trunc = OceanAgents.Core.ActorView('AgentID', "alice", 'EpisodeID', 1, 'FrameID', 3, 'Observation', 6, ...
                'ActionLabels', [1 2], 'ActionMask', [false false], 'Info', struct('successor_action_mask', [true true]));
            fb2 = OceanAgents.Core.Feedback('AgentID', "alice", 'EpisodeID', 1, 'FrameID', 2, 'Truncated', true, 'NextView', trunc);
            tc.verifyEqual(fb2.successorMask(), [true true]);
            tc.verifyError(@() OceanAgents.Core.Feedback('AgentID', "alice", 'EpisodeID', 1, 'FrameID', 2, ...
                'Terminated', true, 'NextView', trunc), 'OceanAgents:Core:InvalidRecord');
        end

        function transitionRules(tc)
            args = {'AgentID', "a", 'EpisodeID', 1, 'Observation', 37, 'ActionLabels', 1:4, 'Reward', -1, ...
                'NextObservation', 25, 'NextActionLabels', 1:4, 'NextMask', true(1, 4), 'Discount', 1};
            t = OceanAgents.Core.Transition(args{:}, 'Action', 1);
            tc.verifyEqual(t.Duration, 1);
            tc.verifyError(@() OceanAgents.Core.Transition(args{:}, 'Action', 9), 'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() OceanAgents.Core.Transition(args{1:end-1}, 1.5, 'Action', 1), 'OceanAgents:Core:InvalidRecord');
        end

        function batchValidation(tc)
            f = tc.frame2([true true]);
            ok = OceanAgents.Core.ActionBatch('EpisodeID', 1, 'FrameID', 4, 'Decisions', [tc.decFor("alice", 1, 4), tc.decFor("bob", 2, 4)]);
            ok.validate(f);                                                    % no error
            tc.verifyEqual(ok.decisionFor("bob").Action, 2);
            missing = OceanAgents.Core.ActionBatch('EpisodeID', 1, 'FrameID', 4, 'Decisions', tc.decFor("alice", 1, 4));
            tc.verifyError(@() missing.validate(f), 'OceanAgents:Core:MissingDecision');
            dup = OceanAgents.Core.ActionBatch('EpisodeID', 1, 'FrameID', 4, 'Decisions', ...
                [tc.decFor("alice", 1, 4), tc.decFor("alice", 2, 4), tc.decFor("bob", 1, 4)]);
            tc.verifyError(@() dup.validate(f), 'OceanAgents:Core:DuplicateDecision');
            stale = OceanAgents.Core.ActionBatch('EpisodeID', 1, 'FrameID', 3, 'Decisions', [tc.decFor("alice", 1, 3), tc.decFor("bob", 1, 3)]);
            tc.verifyError(@() stale.validate(f), 'OceanAgents:Core:StaleFrame');
            illegal = OceanAgents.Core.ActionBatch('EpisodeID', 1, 'FrameID', 4, 'Decisions', [tc.decFor("alice", 3, 4), tc.decFor("bob", 1, 4)]);
            tc.verifyError(@() illegal.validate(f), 'OceanAgents:Core:IllegalAction');
            onlyAlice = tc.frame2([true false]);
            extra = OceanAgents.Core.ActionBatch('EpisodeID', 1, 'FrameID', 4, 'Decisions', [tc.decFor("alice", 1, 4), tc.decFor("bob", 1, 4)]);
            tc.verifyError(@() extra.validate(onlyAlice), 'OceanAgents:Core:UnexpectedDecision');
        end

        function frameRules(tc)
            tc.verifyError(@() OceanAgents.Core.DecisionFrame('EpisodeID', 1, 'FrameID', 1, 'Roster', ["a" "b"], ...
                'Live', [true false], 'Requested', [true true], 'Views', {[], []}), 'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() OceanAgents.Core.DecisionFrame('EpisodeID', 1, 'FrameID', 1, 'Roster', ["a" "b"], ...
                'Live', [true true], 'Requested', [true false], 'Views', {[], []}), 'OceanAgents:Core:InvalidRecord');
            f = tc.frame2([true true]);
            fb = f.feedbackFor(2, 3);
            tc.verifyEqual(fb.AgentID, "bob");
            tc.verifyEqual(fb.FrameID, 3);
            tc.verifyFalse(fb.RewardValid);
        end

        function recordsRejectNestedHandles(tc)
            % A0 section 1: records hold value data only, at any depth (critic round 1, F2)
            base = {'AgentID', "a", 'EpisodeID', 1, 'FrameID', 1, 'Observation', 1, 'ActionLabels', [1 2], 'ActionMask', [true true]};
            m = containers.Map();
            bad = {struct('debug', {{m}}), struct('a', struct('b', {{1, m}})), struct('f', {{@() 1}}), ...
                   struct('s', struct('x', {1, m})), struct('j', {{java.util.HashMap()}})};
            for k = 1:numel(bad)
                tc.verifyError(@() OceanAgents.Core.ActorView(base{:}, 'Info', bad{k}), 'OceanAgents:Core:InvalidRecord', ...
                    sprintf('nested handle case %d accepted', k));
            end
            tc.verifyError(@() OceanAgents.Core.Decision('AgentID', "a", 'EpisodeID', 1, 'FrameID', 1, 'Action', 1, ...
                'Provenance', struct('p', {{m}})), 'OceanAgents:Core:InvalidRecord');
            ok = OceanAgents.Core.ActorView(base{:}, 'Info', struct('a', {{1, "x", struct('b', [1 2])}}, 't', "text"));
            tc.verifyEqual(ok.Info.t, "text");
        end

        function feedbackMustFollowItsDecision(tc)
            % the successor view must be from the same episode and a later frame (critic round 1, N2)
            mk = @(ep, fr) OceanAgents.Core.ActorView('AgentID', "alice", 'EpisodeID', ep, 'FrameID', fr, ...
                'Observation', 1, 'ActionLabels', [1 2], 'ActionMask', [true true]);
            fb = @(nv) OceanAgents.Core.Feedback('AgentID', "alice", 'EpisodeID', 1, 'FrameID', 5, 'NextView', nv);
            fb(mk(1, 6));                                                    % valid
            tc.verifyError(@() fb(mk(9, 6)), 'OceanAgents:Core:StaleFrame');  % other episode
            tc.verifyError(@() fb(mk(1, 5)), 'OceanAgents:Core:StaleFrame');  % same frame
            tc.verifyError(@() fb(mk(1, 4)), 'OceanAgents:Core:StaleFrame');  % earlier frame
        end

        function capabilities(tc)
            caps = OceanAgents.Core.CapabilitySet('Enumerable', true);
            tc.verifyEqual(caps.list(), "Enumerable");
            caps.require("Enumerable");
            tc.verifyError(@() caps.require(["Enumerable" "ExactModel"]), 'OceanAgents:Core:MissingCapability');
            tc.verifyError(@() caps.require("Teleport"), 'OceanAgents:Core:InvalidRecord');
        end

        function envSpec(tc)
            sp = OceanStats.RL.Env.Spaces.Discrete(4);
            s = OceanAgents.Core.EnvSpec('Name', "x", 'Timing', "single", 'Roster', "agent1", ...
                'ObservationSpaces', {sp}, 'ActionSpaces', {sp});
            tc.verifyEqual(s.Roles, "default");
            tc.verifyEqual(s.slotOf("agent1"), 1);
            tc.verifyError(@() OceanAgents.Core.EnvSpec('Name', "x", 'Timing', "single", 'Roster', ["a" "b"], ...
                'ObservationSpaces', {sp, sp}, 'ActionSpaces', {sp, sp}), 'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() OceanAgents.Core.EnvSpec('Name', "x", 'Timing', "async", 'Roster', "a", ...
                'ObservationSpaces', {sp}, 'ActionSpaces', {sp}), 'OceanAgents:Core:InvalidRecord');
        end
    end
end
