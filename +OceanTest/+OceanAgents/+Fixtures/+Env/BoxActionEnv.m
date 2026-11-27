classdef BoxActionEnv < OceanStats.RL.Env.IEnvironment
    % BOXACTIONENV  Test fixture: a native environment with a continuous (Box) action space, which the single-agent
    % bridge must refuse with OceanAgents:Env:UnsupportedSpace. It is never stepped.
    methods
        function obj = BoxActionEnv()
            obj.ActionSpace = OceanStats.RL.Env.Spaces.Box(-1, 1);
        end
        function [obs, reward, terminated, truncated, info] = step(~, ~)
            obs = 0; reward = 0; terminated = true; truncated = false; info = struct();
        end
        function setSeed(~, ~), end
    end
    methods (Access = protected)
        function [obs, info] = doReset_(~)
            obs = 0; info = struct();
        end
    end
end
