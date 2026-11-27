classdef BrokenIllegalActionEnv < OceanStats.RL.Env.Classic.RandomWalkEnv
    % BROKENILLEGALACTIONENV  Violates exactly RL-5: an action outside the ActionSpace is silently replaced by action 1
    % (and the world moves) instead of being refused.
    methods
        function [obs, reward, terminated, truncated, info] = step(obj, action)
            if ~obj.ActionSpace.contains(action), action = 1; end
            [obs, reward, terminated, truncated, info] = step@OceanStats.RL.Env.Classic.RandomWalkEnv(obj, action);
        end
    end
end
