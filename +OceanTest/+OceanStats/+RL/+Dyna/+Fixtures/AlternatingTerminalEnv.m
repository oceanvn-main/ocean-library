classdef AlternatingTerminalEnv < OceanStats.RL.Env.IEnvironment & OceanStats.RL.Env.IEnumerableEnvironment
    %ALTERNATINGTERMINALENV One-state stochastic fixture with action label 7.
    properties (Access = private)
        Seed (1,1) double = 1
        EpisodeDone (1,1) logical = true
    end
    methods
        function obj = AlternatingTerminalEnv()
            obj.ObservationSpace = OceanStats.RL.Env.Spaces.Discrete(1);
            obj.ActionSpace = OceanStats.RL.Env.Spaces.Discrete(1, 'Start', 7);
        end
        function setSeed(obj, seed)
            arguments
                obj
                seed (1,1) double {mustBeInteger,mustBeNonnegative}
            end
            obj.Seed = seed;
        end
        function [next, reward, terminated, truncated, info] = step(obj, action)
            arguments
                obj
                action (1,1) double
            end
            if obj.EpisodeDone
                error('OceanTest:RL:Dyna:EpisodeEnded', 'Call reset before step.');
            end
            if action ~= 7
                error('OceanTest:RL:Dyna:WrongActionLabel', 'Expected external action label 7.');
            end
            next = 1;
            reward = 2 + 4 * mod(obj.Seed + 1, 2);
            terminated = true;
            truncated = false;
            info = struct('action_mask', false);
            obj.EpisodeDone = true;
        end
        function states = get_all_states(~), states = 1; end
        function idx = state_to_index(~, state), idx = double(state == 1); end
        function idx = action_to_index(~, action), idx = double(action == 7); end
        function actions = get_valid_actions(~, ~), actions = 7; end
        function tf = is_terminal(~, ~), tf = false; end
    end
    methods (Access = protected)
        function [obs, info] = doReset_(obj)
            obj.EpisodeDone = false;
            obs = 1;
            info = struct('action_mask', true);
        end
    end
end
