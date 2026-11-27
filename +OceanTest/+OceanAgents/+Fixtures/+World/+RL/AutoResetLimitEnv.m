classdef AutoResetLimitEnv < OceanStats.RL.Env.EnvWrapper
    % AUTORESETLIMITENV  Violates exactly RL-6: a time-limit wrapper that AUTO-RESETS on truncation and returns the
    % new episode's first observation instead of the pre-reset final one (Gymnasium AutoReset-style defect).
    %   env = AutoResetLimitEnv(RandomWalkEnv(), 3)
    properties (SetAccess = private)
        Limit double = 3
    end
    properties (Access = private)
        Steps double = 0
    end
    methods
        function obj = AutoResetLimitEnv(inner, limit)
            obj@OceanStats.RL.Env.EnvWrapper(inner);
            obj.Limit = limit;
        end
        function [obs, reward, terminated, truncated, info] = step(obj, action)
            [obs, reward, terminated, truncated, info] = obj.Inner.step(action);
            obj.Steps = obj.Steps + 1;
            if ~terminated && obj.Steps >= obj.Limit
                truncated = true;
                [obs, info] = obj.Inner.reset();           % the defect: the final observation is lost
                obj.Steps = 0;
            end
        end
    end
    methods (Access = protected)
        function [obs, info] = doReset_(obj)
            obj.Steps = 0;
            [obs, info] = doReset_@OceanStats.RL.Env.EnvWrapper(obj);
        end
    end
end
