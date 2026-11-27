classdef BrokenStepTypesEnv < OceanStats.RL.Env.Classic.RandomWalkEnv
    % BROKENSTEPTYPESENV  Violates exactly RL-2: terminated is returned as a double (0/1), not a logical.
    methods
        function [obs, reward, terminated, truncated, info] = step(obj, action)
            [obs, reward, terminated, truncated, info] = step@OceanStats.RL.Env.Classic.RandomWalkEnv(obj, action);
            terminated = double(terminated);
        end
    end
end
