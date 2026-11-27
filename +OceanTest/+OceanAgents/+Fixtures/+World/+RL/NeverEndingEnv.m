classdef NeverEndingEnv < OceanStats.RL.Env.IEnvironment
    % NEVERENDINGENV  A correct sample-only environment that never terminates: a ring of 5 states, action 1 steps
    % forward, action 2 stays, reward 0. It declares no model, so no probe can reach a terminal state and the
    % post-termination half of RL-5 can never be exercised on it (certificate coverage fixture).
    properties (Access = private)
        State = []
    end
    methods
        function obj = NeverEndingEnv()
            obj.ObservationSpace = OceanStats.RL.Env.Spaces.Discrete(5);
            obj.ActionSpace = OceanStats.RL.Env.Spaces.Discrete(2);
        end
        function setSeed(~, ~)
        end
        function [obs, reward, terminated, truncated, info] = step(obj, action)
            if isempty(obj.State)
                error('OceanTest:NeverEndingEnv:NotReset', 'step called before reset.');
            end
            if ~obj.ActionSpace.contains(action)
                error('OceanTest:NeverEndingEnv:InvalidAction', 'The action is not in the action space.');
            end
            if action == 1, obj.State = mod(obj.State, 5) + 1; end
            obs = obj.State;  reward = 0;  terminated = false;  truncated = false;
            info = struct('action_mask', true(1, 2));
        end
    end
    methods (Access = protected)
        function [obs, info] = doReset_(obj)
            obj.State = 1;
            obs = 1;
            info = struct('action_mask', true(1, 2));
        end
    end
end
