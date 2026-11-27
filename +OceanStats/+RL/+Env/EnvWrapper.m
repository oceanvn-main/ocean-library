classdef EnvWrapper < OceanStats.RL.Env.IEnvironment
    % ENVWRAPPER  Base of every wrapper: adds behaviour to an environment without subclassing it and without touching
    % it (Gymnasium Wrapper / TorchRL TransformedEnv idea). By default it forwards everything to the wrapped environment;
    % a concrete wrapper overrides only what it changes.
    %
    %   env = OceanStats.RL.Env.Wrappers.TimeLimit(inner, 100);
    %   env.unwrapped()          the innermost environment (e.g. to reach its model: env.unwrapped().get_transitions(...))
    properties (SetAccess = protected)
        Inner = []                                  % the wrapped OceanStats.RL.Env.IEnvironment (checked in the constructor)
    end

    methods
        function obj = EnvWrapper(inner)
            arguments
                inner (1,1) OceanStats.RL.Env.IEnvironment
            end
            obj.Inner = inner;
            obj.ObservationSpace = inner.ObservationSpace;
            obj.ActionSpace = inner.ActionSpace;
        end

        function [obs, reward, terminated, truncated, info] = step(obj, action)
            [obs, reward, terminated, truncated, info] = obj.Inner.step(action);
        end

        function setSeed(obj, seed)
            obj.Inner.setSeed(seed);
        end

        function env = unwrapped(obj)
            env = obj.Inner.unwrapped();
        end

        function meta = getMetadata(obj)
            meta = obj.Inner.getMetadata();
            meta.Assumptions = [{sprintf('wrapped by %s', class(obj))}, meta.Assumptions];
        end

        function spec = getPipelineSpec(obj)
            spec = obj.Inner.getPipelineSpec();
            spec.Kind = "NONVIRTUAL";
            spec.ComposesFrom = [{class(obj.Inner)}, spec.ComposesFrom];
        end
    end

    methods (Access = protected)
        function [obs, info] = doReset_(obj)
            [obs, info] = obj.Inner.reset();
        end
    end
end
