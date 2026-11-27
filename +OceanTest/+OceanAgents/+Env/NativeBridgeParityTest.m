classdef NativeBridgeParityTest < OceanTest.BaseTestCase
    % NATIVEBRIDGEPARITYTEST  OceanAgents.Env.NativeSingleAgentBridge: a native loop and the bridged loop with the same
    % seed and the same prescribed actions see identical trajectories (C32 / IG01), hand table A0 section 9.1,
    % truncation semantics, label-not-column passing, validation before env.step, NotReady, InfoAllow, capabilities,
    % and monotonic Episode/Frame IDs. Expected values: Planning/System/Game Theory/A0_Contract_Freeze.md 5.1 and 9.1.

    methods (Static, Access = private)
        function env = randomWalk()
            env = OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner', 5);
        end

        function env = cliff()
            env = OceanStats.RL.Env.Classic.CliffWalkingEnv();
        end

        function d = decide(view, label)
            d = OceanAgents.Core.Decision('AgentID', view.AgentID, 'EpisodeID', view.EpisodeID, ...
                'FrameID', view.FrameID, 'Action', label);
        end

        function T = nativeRun(env, seed, actions)
            % Plain OceanStats loop: env.reset('Seed', s); env.step(a) ... (stops when the episode ends).
            [obs, info] = env.reset('Seed', seed);
            T = struct('Obs0', obs, 'Mask0', logical(info.action_mask), 'Obs', [], 'Reward', [], ...
                'Terminated', false(1, 0), 'Truncated', false(1, 0), 'Mask', {{}});
            for a = actions
                [obs, r, term, trunc, info] = env.step(a);
                T.Obs(end + 1) = obs;
                T.Reward(end + 1) = r;
                T.Terminated(end + 1) = term;
                T.Truncated(end + 1) = trunc;
                T.Mask{end + 1} = logical(info.action_mask);
                if term || trunc, break; end
            end
        end

        function [T, fbs, b] = bridgeRun(env, seed, actions)
            % The same loop through the bridge; Mask is the native-equivalent successor legality.
            b = OceanAgents.Env.NativeSingleAgentBridge(env);
            view = b.reset(seed);
            T = struct('Obs0', view.Observation, 'Mask0', view.ActionMask, 'Obs', [], 'Reward', [], ...
                'Terminated', false(1, 0), 'Truncated', false(1, 0), 'Mask', {{}});
            fbs = {};
            for a = actions
                fb = b.step(OceanTest.OceanAgents.Env.NativeBridgeParityTest.decide(view, a));
                fbs{end + 1} = fb; %#ok<AGROW>
                T.Obs(end + 1) = fb.NextView.Observation;
                T.Reward(end + 1) = fb.Reward;
                T.Terminated(end + 1) = fb.Terminated;
                T.Truncated(end + 1) = fb.Truncated;
                if fb.Truncated     % the native mask lives in successor_action_mask; the view's own mask is all false
                    tc_assertAllFalse_(fb.NextView.ActionMask);
                    T.Mask{end + 1} = fb.NextView.Info.successor_action_mask;
                else
                    T.Mask{end + 1} = fb.NextView.ActionMask;
                end
                view = fb.NextView;
                if fb.ended(), break; end
            end
        end

        function f = fixture(varargin)
            f = OceanTest.OceanAgents.Fixtures.Env.LabelledCounterEnv(varargin{:});
        end
    end

    methods (Test)
        function randomWalkHandTable(tc)
            % A0 section 9.1: start 4, actions [2 2 2] -> 5, 6, 7; rewards 0, 0, 1; terminal only at 7.
            b = OceanAgents.Env.NativeSingleAgentBridge(tc.randomWalk(), 'AgentID', "walker");
            view = b.reset(3);
            tc.verifyEqual(view.ActionLabels, [1 2]);
            tc.verifyEqual(view.Observation, 4);
            tc.verifyEqual(view.ActionMask, [true true]);
            obsBefore = [4 5 6];
            expectedObs = [5 6 7];
            expectedReward = [0 0 1];
            expectedTerm = [false false true];
            for t = 1:3
                tc.verifyEqual(view.Observation, obsBefore(t));
                fb = b.step(tc.decide(view, 2));
                tc.verifyEqual(fb.NextView.Observation, expectedObs(t));
                tc.verifyEqual(fb.Reward, expectedReward(t));
                tc.verifyEqual(fb.Terminated, expectedTerm(t));
                tc.verifyFalse(fb.Truncated);
                tc.verifyEqual(fb.FrameID, view.FrameID);            % feedback names the decision's frame
                view = fb.NextView;
            end
            tc.verifyEqual(view.Observation, 7);
            tc.verifyEqual(view.ActionMask, [false false]);
            tc.verifyTrue(view.isTerminalView());
            tc.verifyEqual(fb.Reason, "terminal");
            tc.verifyFalse(isfield(view.Info, 'successor_action_mask'));
            tc.verifyEqual(b.NativeStepCalls, 3);
        end

        function randomWalkParityWithNativeLoop(tc)
            actionSets = {[2 2 2], [1 2 1 1 1], [2 1 2 1 2 2 2]};
            for s = [0 5 123]
                for k = 1:numel(actionSets)
                    native = tc.nativeRun(tc.randomWalk(), s, actionSets{k});
                    bridged = tc.bridgeRun(tc.randomWalk(), s, actionSets{k});
                    tc.verifyEqual(bridged, native, sprintf('seed %d, action set %d', s, k));
                end
            end
        end

        function cliffWalkingParityWithNativeLoop(tc)
            % Includes a cliff fall (right from the start: -100 and back to 37), wall bumps and a left move.
            actions = [2 1 1 4 2 2 3 3 2 1 1 1 1 3];
            for s = [1 42]
                native = tc.nativeRun(tc.cliff(), s, actions);
                bridged = tc.bridgeRun(tc.cliff(), s, actions);
                tc.verifyEqual(bridged, native);
            end
            % hand check of the first two steps: 37 -right-> cliff -> 37 (-100); 37 -up-> 25 (-1)
            native = tc.nativeRun(tc.cliff(), 1, actions);
            tc.verifyEqual(native.Obs(1:2), [37 25]);
            tc.verifyEqual(native.Reward(1:2), [-100 -1]);
        end

        function cliffWalkingEpisodeToGoal(tc)
            % up, 11 x right, down: 37 -> 25 -> 26 ... 36 -> 48 (goal). 13 steps of -1 -> return -13.
            actions = [1, repmat(2, 1, 11), 3];
            [T, fbs] = tc.bridgeRun(tc.cliff(), 9, actions);
            tc.verifyEqual(T.Obs0, 37);
            tc.verifyEqual(T.Obs, [25:36, 48]);
            tc.verifyEqual(T.Reward, -ones(1, 13));
            tc.verifyEqual(sum(T.Reward), -13);
            tc.verifyEqual(T.Terminated, [false(1, 12) true]);
            tc.verifyFalse(any(T.Truncated));
            tc.verifyEqual(fbs{end}.NextView.ActionMask, false(1, 4));
            tc.verifyEqual(T, tc.nativeRun(tc.cliff(), 9, actions));
        end

        function timeLimitTruncation(tc)
            % A0 section 9.1: TimeLimit(env, 2) with [2 2 2] -> step 2 obs 6, reward 0, truncated, successor [true true].
            actions = [2 2 2];
            [T, fbs, b] = tc.bridgeRun(OceanStats.RL.Env.Wrappers.TimeLimit(tc.randomWalk(), 2), 4, actions);
            tc.verifyEqual(numel(fbs), 2);
            fb = fbs{2};
            tc.verifyEqual(fb.NextView.Observation, 6);
            tc.verifyEqual(fb.Reward, 0);
            tc.verifyFalse(fb.Terminated);
            tc.verifyTrue(fb.Truncated);
            tc.verifyEqual(fb.Reason, "time-limit");
            tc.verifyEqual(fb.NextView.ActionMask, [false false]);
            tc.verifyEqual(fb.NextView.Info.successor_action_mask, [true true]);
            tc.verifyEqual(fb.successorMask(), [true true]);
            tc.verifyEqual(b.NativeStepCalls, 2);
            tc.verifyError(@() b.step(tc.decide(fb.NextView, 2)), 'OceanAgents:Env:NotReady');
            tc.verifyEqual(b.NativeStepCalls, 2);
            tc.verifyEqual(T, tc.nativeRun(OceanStats.RL.Env.Wrappers.TimeLimit(tc.randomWalk(), 2), 4, actions));
            % step 1 is an ordinary transition (no successor mask field)
            tc.verifyFalse(isfield(fbs{1}.NextView.Info, 'successor_action_mask'));
        end

        function nonDefaultLabelsPassLabelNotColumn(tc)
            env = tc.fixture('Horizon', 3);
            b = OceanAgents.Env.NativeSingleAgentBridge(env);
            view = b.reset(0);
            tc.verifyEqual(view.ActionLabels, [0 1 2]);
            tc.verifyEqual(view.ActionMask, env.maskAt(0));
            fb = b.step(tc.decide(view, 2));                  % label 2 = column 3
            tc.verifyEqual(env.LastAction, 2);
            tc.verifyEqual(fb.Reward, 2);
            tc.verifyEqual(fb.NextView.ActionMask, env.maskAt(1));   % the mask changes: [false true true]
            fb = b.step(tc.decide(fb.NextView, 1));           % label 1 (illegal at obs 0, legal now)
            tc.verifyEqual(env.LastAction, 1);
            tc.verifyEqual(fb.NextView.ActionMask, env.maskAt(2));
            fb = b.step(tc.decide(fb.NextView, 0));           % label 0 = column 1
            tc.verifyEqual(env.LastAction, 0);
            tc.verifyTrue(fb.Terminated);
            tc.verifyEqual(fb.NextView.Observation, 3);
            tc.verifyEqual(fb.NextView.ActionMask, false(1, 3));
        end

        function decisionValidatedBeforeEnvStep(tc)
            env = tc.fixture('Horizon', 5);
            b = OceanAgents.Env.NativeSingleAgentBridge(env);
            view = b.reset(0);                                % mask [true false true]
            frame0 = b.FrameID;
            tc.verifyError(@() b.step(tc.decide(view, 1)), 'OceanAgents:Core:IllegalAction');
            tc.verifyError(@() b.step(tc.decide(view, 3)), 'OceanAgents:Core:IllegalAction');   % not a label at all
            stale = OceanAgents.Core.Decision('AgentID', view.AgentID, 'EpisodeID', view.EpisodeID, ...
                'FrameID', view.FrameID - 1, 'Action', 0);
            tc.verifyError(@() b.step(stale), 'OceanAgents:Core:StaleFrame');
            other = OceanAgents.Core.Decision('AgentID', "intruder", 'EpisodeID', view.EpisodeID, ...
                'FrameID', view.FrameID, 'Action', 0);
            tc.verifyError(@() b.step(other), 'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() b.step(0), 'OceanAgents:Core:InvalidRecord');                  % not a Decision
            % nothing reached the world, and the bridge still answers the same view
            tc.verifyEqual(env.StepCalls, 0);
            tc.verifyEqual(env.Count, 0);
            tc.verifyEqual(b.NativeStepCalls, 0);
            tc.verifyEqual(b.FrameID, frame0);
            tc.verifyEqual(b.Status, "Running");
            fb = b.step(tc.decide(view, 0));
            tc.verifyEqual(env.StepCalls, 1);
            tc.verifyEqual(fb.NextView.Observation, 1);
            % a decision on the previous (now stale) view after a successful step
            tc.verifyError(@() b.step(tc.decide(view, 0)), 'OceanAgents:Core:StaleFrame');
            tc.verifyEqual(env.StepCalls, 1);
        end

        function notReadyBeforeResetAndAfterEnd(tc)
            env = tc.fixture('Horizon', 1);
            b = OceanAgents.Env.NativeSingleAgentBridge(env);
            d = OceanAgents.Core.Decision('AgentID', "agent1", 'EpisodeID', 1, 'FrameID', 1, 'Action', 0);
            tc.verifyError(@() b.step(d), 'OceanAgents:Env:NotReady');
            tc.verifyEqual(env.StepCalls, 0);
            view = b.reset(7);
            tc.verifyEqual(env.SeedSeen, 7);                  % seed forwarded as env.reset('Seed', seed)
            fb = b.step(tc.decide(view, 0));
            tc.verifyTrue(fb.Terminated);
            tc.verifyEqual(b.Status, "Ended");
            tc.verifyError(@() b.step(tc.decide(fb.NextView, 0)), 'OceanAgents:Env:NotReady');
            tc.verifyEqual(env.StepCalls, 1);
            view = b.reset(8);                                % a new episode makes it ready again
            tc.verifyEqual(b.Status, "Running");
            tc.verifyEqual(view.Observation, 0);
            tc.verifyError(@() b.reset(-1), 'MATLAB:validators:mustBeNonnegative');
        end

        function infoAllowlist(tc)
            b = OceanAgents.Env.NativeSingleAgentBridge(tc.fixture());
            view = b.reset(0);
            tc.verifyEmpty(fieldnames(view.Info));            % default: action_mask consumed, nothing else leaks
            fb = b.step(tc.decide(view, 0));
            tc.verifyEmpty(fieldnames(fb.NextView.Info));

            b2 = OceanAgents.Env.NativeSingleAgentBridge(tc.fixture(), 'InfoAllow', ["action_mask" "secret"]);
            view = b2.reset(0);
            tc.verifyEqual(sort(string(fieldnames(view.Info))), "secret");
            tc.verifyEqual(view.Info.secret, 42);
            tc.verifyFalse(isfield(view.Info, 'debug'));
            tc.verifyFalse(isfield(view.Info, 'action_mask'));
        end

        function missingMaskDefaults(tc)
            env = tc.fixture('Horizon', 2, 'ProvideMask', false);
            b = OceanAgents.Env.NativeSingleAgentBridge(env);
            view = b.reset(0);
            tc.verifyEqual(view.ActionMask, true(1, 3));      % non-ended, no mask -> all legal
            fb = b.step(tc.decide(view, 0));
            tc.verifyEqual(fb.NextView.ActionMask, true(1, 3));
            fb = b.step(tc.decide(fb.NextView, 1));
            tc.verifyTrue(fb.Terminated);
            tc.verifyEqual(fb.NextView.ActionMask, false(1, 3));
        end

        function capabilityDerivation(tc)
            social = ["Snapshot" "Communication" "Commitments" "PrivilegedState" "TurnBased"];

            spec = OceanAgents.Env.NativeSingleAgentBridge(tc.randomWalk(), 'AgentID', "a", 'Role', "walker").describe();
            tc.verifyEqual(spec.Timing, "single");
            tc.verifyEqual(spec.Roster, "a");
            tc.verifyEqual(spec.Roles, "walker");
            tc.verifyTrue(spec.Capabilities.Enumerable);
            tc.verifyTrue(spec.Capabilities.ExactModel);
            tc.verifyEqual(spec.Horizon, Inf);
            tc.verifyClass(spec.ActionSpaces{1}, 'OceanStats.RL.Env.Spaces.Discrete');

            wrapped = OceanStats.RL.Env.Wrappers.TimeLimit(tc.randomWalk(), 2);
            spec = OceanAgents.Env.NativeSingleAgentBridge(wrapped).describe();
            tc.verifyTrue(spec.Capabilities.Enumerable);      % from unwrapped()
            tc.verifyFalse(spec.Capabilities.ExactModel);     % the horizon changes the model
            tc.verifyEqual(spec.Horizon, 2);
            tc.verifyError(@() spec.Capabilities.require("ExactModel"), 'OceanAgents:Core:MissingCapability');

            spec = OceanAgents.Env.NativeSingleAgentBridge(tc.fixture()).describe();
            tc.verifyFalse(spec.Capabilities.Enumerable);
            tc.verifyFalse(spec.Capabilities.ExactModel);
            for c = social
                tc.verifyFalse(spec.Capabilities.(char(c)), c);
            end
            tc.verifyEmpty(spec.Capabilities.list());

            % describe never mutates an episode
            b = OceanAgents.Env.NativeSingleAgentBridge(tc.randomWalk());
            b.describe();
            tc.verifyEqual([b.EpisodeID b.FrameID], [0 0]);
        end

        function episodeAndFrameIDsMonotonic(tc)
            b = OceanAgents.Env.NativeSingleAgentBridge(tc.randomWalk());
            frames = [];
            episodes = [];
            for e = 1:3
                view = b.reset(e);
                frames(end + 1) = view.FrameID; %#ok<AGROW>
                episodes(end + 1) = view.EpisodeID; %#ok<AGROW>
                fb = b.step(tc.decide(view, 2));
                while ~fb.ended()
                    frames(end + 1) = fb.NextView.FrameID; %#ok<AGROW>
                    episodes(end + 1) = fb.NextView.EpisodeID; %#ok<AGROW>
                    fb = b.step(tc.decide(fb.NextView, 2));
                end
                frames(end + 1) = fb.NextView.FrameID; %#ok<AGROW>
                episodes(end + 1) = fb.NextView.EpisodeID; %#ok<AGROW>
            end
            tc.verifyEqual(frames, 1:numel(frames));          % never reset within the run
            tc.verifyEqual(unique(episodes), 1:3);
            tc.verifyTrue(all(diff(episodes) >= 0));
            tc.verifyEqual(numel(frames), 3 * 4);             % 4 views per episode (start 4 -> 7 in 3 steps)
        end

        function unsupportedSpaceAndInput(tc)
            tc.verifyError(@() OceanAgents.Env.NativeSingleAgentBridge(OceanTest.OceanAgents.Fixtures.Env.BoxActionEnv()), ...
                'OceanAgents:Env:UnsupportedSpace');
            tc.verifyError(@() OceanAgents.Env.NativeSingleAgentBridge(42), 'OceanAgents:Env:UnsupportedSpace');
            tc.verifyError(@() OceanAgents.Env.NativeSingleAgentBridge(tc.randomWalk(), 'AgentID', ""), ...
                'OceanAgents:Core:InvalidRecord');
        end

        function pillarsAndClose(tc)
            env = tc.randomWalk();
            b = OceanAgents.Env.NativeSingleAgentBridge(env);
            tc.verifyTrue(isa(b, 'OceanAgents.Core.ISingleAgentEnvironment'));
            meta = b.getMetadata();
            tc.verifyEqual(sort(fieldnames(meta)), sort(fieldnames(env.getMetadata())));
            tc.verifyFalse(meta.IsSymbolic);
            spec = b.getPipelineSpec();
            tc.verifyEqual(sort(fieldnames(spec)), sort(fieldnames(env.getPipelineSpec())));
            tc.verifyEqual(spec.Kind, "NONVIRTUAL");
            tc.verifyEqual([spec.Outputs.Domain], ["Agent-ActorView" "Agent-Feedback"]);
            tc.verifyEqual(spec.Inputs.Domain, "Agent-Decision");
            b.close();
            b.close();                                        % documented no-op, idempotent
            view = b.reset(0);
            tc.verifyEqual(view.Observation, 4);
        end
    end
end

function tc_assertAllFalse_(mask)
% A truncated step's NextView must carry an all-false mask (A0 5.1); fail loudly inside the helper loop otherwise.
assert(~any(mask), 'OceanTest:NativeBridgeParity:MaskNotCleared', 'An ended NextView must have an all-false mask.');
end
