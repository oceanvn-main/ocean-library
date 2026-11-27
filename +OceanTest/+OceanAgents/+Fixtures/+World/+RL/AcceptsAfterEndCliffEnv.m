classdef AcceptsAfterEndCliffEnv < OceanStats.RL.Env.Classic.CliffWalkingEnv
    % ACCEPTSAFTERENDCLIFFENV  Violates exactly RL-5 (post-termination half): once the goal is reached, a further step
    % is ACCEPTED and returns the terminal observation again (raw Gymnasium behaviour). A random probe episode almost
    % never reaches the cliff-walking goal in 50 steps, so only a probe DIRECTED to a terminal exposes the defect.
    properties (Access = private)
        Ended (1,1) logical = false
        LastObs = []
    end
    methods
        function [obs, reward, terminated, truncated, info] = step(obj, action)
            if obj.Ended
                obs = obj.LastObs;  reward = 0;  terminated = true;  truncated = false;
                info = struct('action_mask', false(1, obj.ActionSpace.N));
                return
            end
            [obs, reward, terminated, truncated, info] = step@OceanStats.RL.Env.Classic.CliffWalkingEnv(obj, action);
            obj.LastObs = obs;
            obj.Ended = terminated;
        end
    end
    methods (Access = protected)
        function [obs, info] = doReset_(obj)
            obj.Ended = false;
            [obs, info] = doReset_@OceanStats.RL.Env.Classic.CliffWalkingEnv(obj);
        end
    end
end
