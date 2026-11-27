classdef (Abstract) IEnvironment < handle
    % IENVIRONMENT  The algorithm-free contract of a reinforcement-learning environment (Gymnasium / dm_env style).
    % An environment knows nothing about the algorithm that uses it: nothing in this package refers to a learning method.
    %
    %   [obs, info] = env.reset()                       start an episode
    %   [obs, info] = env.reset('Seed', 7)              ... after (re)seeding the environment's own random stream
    %   [obs, r, terminated, truncated, info] = env.step(action)
    %
    %   terminated  true MDP terminal state reached (an estimator must NOT use a successor value past it)
    %   truncated   episode cut off from outside the MDP, e.g. a time limit (an estimator MUST still use the successor value)
    %   info        struct; info.action_mask (1 x A logical) is the set of admissible actions at the NEW observation
    %               (all false when that observation is terminal)
    %
    % ObservationSpace / ActionSpace (OceanStats.RL.Env.Spaces.*) describe what obs and action look like.
    % Randomness: after setSeed / reset('Seed') the environment draws from its own RandStream and never touches the global
    % one. Until it is seeded it draws from MATLAB's global stream, so rng(k) still makes an unseeded run reproducible.
    % Behaviour is added by wrappers (OceanStats.RL.Env.Wrappers.*), not by flags on a base class; unwrapped() returns
    % the innermost environment.
    %
    % Model access (exact p(s',r|s,a)) is NOT part of this contract; see IModelEnvironment.
    % 3 pillars: getMetadata() / getPipelineSpec() (NONVIRTUAL, IsSymbolic = false: an environment is data, not a formula).

    properties (SetAccess = protected)
        ObservationSpace OceanStats.RL.Env.Spaces.ISpace = OceanStats.RL.Env.Spaces.Discrete(1)
        ActionSpace OceanStats.RL.Env.Spaces.ISpace = OceanStats.RL.Env.Spaces.Discrete(1)
    end

    methods (Abstract)
        [obs, reward, terminated, truncated, info] = step(obj, action)
        setSeed(obj, seed)
    end

    methods (Abstract, Access = protected)
        [obs, info] = doReset_(obj)
    end

    methods
        function [obs, info] = reset(obj, options)
            % RESET  Begin a new episode. 'Seed' (optional) re-seeds the environment's random stream first.
            arguments
                obj
                options.Seed double {mustBeScalarOrEmpty} = []
            end
            if ~isempty(options.Seed), obj.setSeed(options.Seed); end
            [obs, info] = obj.doReset_();
        end

        function env = unwrapped(obj)
            env = obj;
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            meta.Assumptions = {sprintf('%s: observation %s, action %s', class(obj), ...
                obj.ObservationSpace.describe(), obj.ActionSpace.describe())};
            meta.IsSymbolic = false;
            meta.Reference = "reinforcement-learning environment (reset/step; Gymnasium-style contract)";
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.ComponentName = string(regexprep(class(obj), '^.*\.', ''));
            spec.Inputs = struct('Name', "action", 'Domain', "RL-Action", 'Dtype', "double", ...
                'Shape', string(mat2str(obj.ActionSpace.Shape)));
            spec.Outputs = struct('Name', "obs,reward,terminated,truncated,info", 'Domain', "RL-Transition", ...
                'Dtype', "double", 'Shape', string(mat2str(obj.ObservationSpace.Shape)));
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', lower(regexprep(class(obj), '^.*\.', '')));
            spec.ComposesFrom = {};
        end
    end
end
