classdef BrokenRewardSeedEnv < OceanStats.RL.Env.Classic.RandomWalkEnv
    % BROKENREWARDSEEDENV  Violates exactly RL-4: a small reward noise is drawn from a stream seeded by a reset
    % COUNTER instead of the episode seed, so the same seed and actions give different rewards (the observations,
    % masks and flags stay reproducible). The noise stream is private: the global stream is never used.
    properties (Access = private)
        Resets double = 0
        Noise = []
    end
    methods
        function [obs, reward, terminated, truncated, info] = step(obj, action)
            [obs, reward, terminated, truncated, info] = step@OceanStats.RL.Env.Classic.RandomWalkEnv(obj, action);
            reward = reward + 1e-3 * rand(obj.Noise);
        end
    end
    methods (Access = protected)
        function [obs, info] = doReset_(obj)
            obj.Resets = obj.Resets + 1;
            obj.Noise = RandStream('mrg32k3a', 'Seed', obj.Resets);
            [obs, info] = doReset_@OceanStats.RL.Env.Classic.RandomWalkEnv(obj);
        end
    end
end
