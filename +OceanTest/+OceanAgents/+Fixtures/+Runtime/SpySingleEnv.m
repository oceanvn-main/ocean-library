classdef SpySingleEnv < OceanAgents.Core.ISingleAgentEnvironment
    % SPYSINGLEENV  Test fixture: the RandomWalk of A0 section 9.1 as an ISingleAgentEnvironment, counting every
    % reset/step call. States 1..NumInner+2, start in the middle (4 for NumInner = 5), labels [1 2] = left/right,
    % reward 1 on reaching the right terminal. Options inject faults and a time limit:
    %   'TimeLimit', T        truncate after T steps (successor_action_mask supplied, as the bridge does)
    %   'Terminals', false    reflecting ends: the episode never ends (for StepBudgetExceeded)
    %   'ThrowOnReset', k / 'ThrowOnStep', k   the k-th call throws Fixture:Injected
    properties
        Log = []                          % optional EventLog
    end
    properties (SetAccess = private)
        AgentID string = "agent1"
        NumInner double = 5
        Terminals logical = true
        TimeLimit double = Inf
        ThrowOnReset double = 0
        ThrowOnStep double = 0
        ResetCalls double = 0
        StepCalls double = 0
        Seeds double = zeros(1, 0)
        Position double = 0
        EpisodeID double = 0
        FrameID double = 0
        EpisodeSteps double = 0
        CurrentView = []
        Ended logical = true
    end
    methods
        function obj = SpySingleEnv(options)
            arguments
                options.AgentID (1,1) string = "agent1"
                options.NumInner (1,1) double = 5
                options.Terminals (1,1) logical = true
                options.TimeLimit (1,1) double = Inf
                options.ThrowOnReset (1,1) double = 0
                options.ThrowOnStep (1,1) double = 0
                options.Log = []
            end
            obj.AgentID = options.AgentID;
            obj.NumInner = options.NumInner;
            obj.Terminals = options.Terminals;
            obj.TimeLimit = options.TimeLimit;
            obj.ThrowOnReset = options.ThrowOnReset;
            obj.ThrowOnStep = options.ThrowOnStep;
            obj.Log = options.Log;
        end

        function spec = describe(obj)
            spec = OceanAgents.Core.EnvSpec('Name', "SpyWalk", 'Timing', "single", 'Roster', obj.AgentID, ...
                'ObservationSpaces', {OceanStats.RL.Env.Spaces.Discrete(obj.NumInner + 2)}, ...
                'ActionSpaces', {OceanStats.RL.Env.Spaces.Discrete(2)});
        end

        function view = reset(obj, seed)
            obj.ResetCalls = obj.ResetCalls + 1;
            obj.log_("env:reset");
            if obj.ResetCalls == obj.ThrowOnReset
                error('Fixture:Injected', 'Injected reset fault on call %d.', obj.ResetCalls);
            end
            obj.Seeds(end + 1) = seed;
            obj.Position = (obj.NumInner + 1) / 2 + 1;
            obj.EpisodeID = obj.EpisodeID + 1;
            obj.FrameID = obj.FrameID + 1;
            obj.EpisodeSteps = 0;
            obj.Ended = false;
            view = obj.view_(true(1, 2), struct());
        end

        function fb = step(obj, decision)
            obj.StepCalls = obj.StepCalls + 1;
            obj.log_("env:step");
            if obj.StepCalls == obj.ThrowOnStep
                error('Fixture:Injected', 'Injected step fault on call %d.', obj.StepCalls);
            end
            if obj.Ended
                error('OceanAgents:Env:NotReady', 'step() before reset() or after the episode ended.');
            end
            decision.validateAgainst(obj.CurrentView);               % before the world moves
            last = obj.NumInner + 2;
            obj.Position = obj.Position + 2 * decision.Action - 3;   % label 1 -> -1, label 2 -> +1
            terminated = false;
            reward = 0;
            if obj.Terminals
                terminated = obj.Position == 1 || obj.Position == last;
                reward = double(obj.Position == last);
            else
                obj.Position = min(max(obj.Position, 1), last);
            end
            obj.EpisodeSteps = obj.EpisodeSteps + 1;
            truncated = ~terminated && obj.EpisodeSteps >= obj.TimeLimit;
            obj.FrameID = obj.FrameID + 1;
            info = struct();
            mask = true(1, 2);
            if terminated || truncated
                mask = false(1, 2);
                if truncated, info.successor_action_mask = true(1, 2); end
                obj.Ended = true;
            end
            next = obj.view_(mask, info);
            fb = OceanAgents.Core.Feedback('AgentID', obj.AgentID, 'EpisodeID', obj.EpisodeID, ...
                'FrameID', decision.FrameID, 'Reward', reward, 'Terminated', terminated, 'Truncated', truncated, ...
                'NextView', next);
        end
    end

    methods (Access = private)
        function v = view_(obj, mask, info)
            v = OceanAgents.Core.ActorView('AgentID', obj.AgentID, 'EpisodeID', obj.EpisodeID, ...
                'FrameID', obj.FrameID, 'Observation', obj.Position, 'ActionLabels', [1 2], ...
                'ActionMask', mask, 'Info', info);
            obj.CurrentView = v;
        end

        function log_(obj, event)
            if ~isempty(obj.Log), obj.Log.add(event); end
        end
    end
end
