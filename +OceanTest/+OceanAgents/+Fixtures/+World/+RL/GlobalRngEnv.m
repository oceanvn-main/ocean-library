classdef GlobalRngEnv < OceanStats.RL.Env.Classic.RandomWalkEnv
    % GLOBALRNGENV  Violates exactly RL-7: every step also draws once from MATLAB's GLOBAL stream (the draw is
    % discarded, so the dynamics stay reproducible).
    methods
        function [obs, reward, terminated, truncated, info] = step(obj, action)
            rand();                                                   % deliberate defect for RL-7
            [obs, reward, terminated, truncated, info] = step@OceanStats.RL.Env.Classic.RandomWalkEnv(obj, action);
        end
    end
end
