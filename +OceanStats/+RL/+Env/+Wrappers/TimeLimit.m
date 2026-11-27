classdef TimeLimit < OceanStats.RL.Env.EnvWrapper
    % TIMELIMIT  Truncates every episode after MaxSteps steps (Gymnasium TimeLimit). The episode ends with
    % truncated = true, terminated = false, unless the inner environment terminated on that very step
    % (a true terminal is never reported as a truncation).
    %
    %   env = OceanStats.RL.Env.Wrappers.TimeLimit(inner, 200);
    properties (SetAccess = private)
        MaxSteps (1,1) double {mustBeInteger, mustBePositive} = 1
    end
    properties (Access = private)
        Steps (1,1) double = 0
    end

    methods
        function obj = TimeLimit(inner, maxSteps)
            arguments
                inner (1,1) OceanStats.RL.Env.IEnvironment
                maxSteps (1,1) double {mustBeInteger, mustBePositive}
            end
            obj@OceanStats.RL.Env.EnvWrapper(inner);
            obj.MaxSteps = maxSteps;
        end

        function [obs, reward, terminated, truncated, info] = step(obj, action)
            [obs, reward, terminated, truncated, info] = obj.Inner.step(action);
            obj.Steps = obj.Steps + 1;
            if ~terminated && obj.Steps >= obj.MaxSteps
                truncated = true;
            end
        end

        function meta = getMetadata(obj)
            meta = getMetadata@OceanStats.RL.Env.EnvWrapper(obj);
            meta.Assumptions = [{sprintf('episodes truncated after %d steps', obj.MaxSteps)}, meta.Assumptions];
        end
    end

    methods (Access = protected)
        function [obs, info] = doReset_(obj)
            obj.Steps = 0;
            [obs, info] = doReset_@OceanStats.RL.Env.EnvWrapper(obj);
        end
    end
end
