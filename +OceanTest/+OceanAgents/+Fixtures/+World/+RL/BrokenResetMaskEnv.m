classdef BrokenResetMaskEnv < OceanStats.RL.Env.Classic.RandomWalkEnv
    % BROKENRESETMASKENV  Violates exactly RL-1: reset returns an action mask of length |A| + 1.
    methods (Access = protected)
        function [obs, info] = doReset_(obj)
            [obs, info] = doReset_@OceanStats.RL.Env.Classic.RandomWalkEnv(obj);
            info.action_mask = [info.action_mask false];
        end
    end
end
