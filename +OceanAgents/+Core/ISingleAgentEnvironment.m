classdef (Abstract) ISingleAgentEnvironment < handle
    % ISINGLEAGENTENVIRONMENT  The native single-agent loop in agent records: one agent, one world, no social machinery.
    % Implemented by OceanAgents.Env.NativeSingleAgentBridge over an existing OceanStats.RL.Env.IEnvironment.
    %
    %   spec = env.describe()         EnvSpec with Timing "single" (no episode mutation)
    %   view = env.reset(seed)        first ActorView of a new episode; seed is required
    %   fb   = env.step(decision)     validates the Decision against the current view BEFORE the world moves,
    %                                 then exactly one transition; returns the agent's Feedback
    %   env.close()                   release resources; safe to repeat
    % See Planning/System/Game Theory/A0_Contract_Freeze.md sections 3 and 5.1.

    methods (Abstract)
        spec = describe(obj)
        view = reset(obj, seed)
        fb = step(obj, decision)
    end

    methods
        function close(~), end
    end
end
