classdef ContinuingSelfLoopEnv < OceanStats.RL.Env.IEnvironment & OceanStats.RL.Env.IEnumerableEnvironment
    %CONTINUINGSELFLOOPENV One-state continuing fixture with action label -3.
    methods
        function obj = ContinuingSelfLoopEnv()
            obj.ObservationSpace = OceanStats.RL.Env.Spaces.Discrete(1);
            obj.ActionSpace = OceanStats.RL.Env.Spaces.Discrete(1, 'Start', -3);
        end
        function setSeed(~, ~), end
        function [next, reward, terminated, truncated, info] = step(obj, action)
            arguments
                obj
                action (1,1) double
            end
            if ~obj.ActionSpace.contains(action)
                error('OceanTest:RL:Dyna:WrongActionLabel', 'Expected external action label -3.');
            end
            next = 1;
            reward = 2;
            terminated = false;
            truncated = false;
            info = struct('action_mask', true);
        end
        function states = get_all_states(~), states = 1; end
        function idx = state_to_index(~, state), idx = double(state == 1); end
        function idx = action_to_index(~, action), idx = double(action == -3); end
        function actions = get_valid_actions(~, ~), actions = -3; end
        function tf = is_terminal(~, ~), tf = false; end
    end
    methods (Access = protected)
        function [obs, info] = doReset_(~)
            obs = 1;
            info = struct('action_mask', true);
        end
    end
end
