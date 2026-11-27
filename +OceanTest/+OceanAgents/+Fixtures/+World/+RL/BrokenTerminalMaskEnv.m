classdef BrokenTerminalMaskEnv < OceanStats.RL.Env.Classic.RandomWalkEnv
    % BROKENTERMINALMASKENV  Violates exactly RL-3: a terminal step reports every action as legal.
    methods
        function [obs, reward, terminated, truncated, info] = step(obj, action)
            [obs, reward, terminated, truncated, info] = step@OceanStats.RL.Env.Classic.RandomWalkEnv(obj, action);
            if terminated, info.action_mask = true(1, 2); end
        end
    end
end
