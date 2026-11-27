classdef (Abstract) IJointEnvironment < handle
    % IJOINTENVIRONMENT  A world shared by several agents. It owns the authoritative state and applies the rules;
    % it stores no policy or learner. Kept separate from OceanStats.RL.Env.IEnvironment (single-agent contract).
    %
    %   spec  = env.describe()        EnvSpec (no episode mutation)
    %   frame = env.reset(seed)       first DecisionFrame of a new episode; seed is required
    %   frame = env.step(batch)       validates the whole ActionBatch, then exactly one transition
    %   env.close()                   release resources; safe to repeat
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 3.

    methods (Abstract)
        spec = describe(obj)
        frame = reset(obj, seed)
        frame = step(obj, batch)
    end

    methods
        function close(~), end
    end
end
