classdef LegacyEnvironment < OceanStats.RL.MonteCarlo.IEpisodeEnvironment
    % LEGACYENVIRONMENT  Compatibility adapter (DEPRECATED, removed together with the learners' migration): presents an
    % algorithm-free OceanStats.RL.Env.IEnvironment that is also an IEnumerableEnvironment to the sampling solvers written
    % against MonteCarlo.IEpisodeEnvironment (three-output step + was_truncated, index helpers, generate_episode).
    %
    %   legacy = OceanStats.RL.Compat.LegacyEnvironment(env);   [s, r, done] = legacy.step(a);  legacy.was_truncated()
    %
    % done = terminated OR truncated; was_truncated() says which. Only the methods the solvers use are forwarded, explicitly
    % (no subsref overloading: that made every call about 10x slower). The deprecated class names (MonteCarlo.BlackjackEnv,
    % MonteCarlo.RacetrackEnv) subclass this adapter and forward their old public properties explicitly.
    %
    % Package note: this adapter is the ONLY place where the environment package meets a learner package. It depends on
    % MonteCarlo.IEpisodeEnvironment, and the deprecated MonteCarlo.* environment names depend on it (a cycle that disappears
    % when the deprecated names are removed); OceanStats.RL.Env itself references none of them.
    properties (SetAccess = private)
        Inner       % the adapted OceanStats.RL.Env.IEnvironment (possibly wrapped)
        Base        % Inner.unwrapped()
    end
    properties (Access = private)
        LastTruncated_ (1,1) logical = false
    end

    methods
        function obj = LegacyEnvironment(inner)
            arguments
                inner (1,1) OceanStats.RL.Env.IEnvironment
            end
            base = inner.unwrapped();
            if ~isa(base, 'OceanStats.RL.Env.IEnumerableEnvironment') || ~isa(inner.ActionSpace, 'OceanStats.RL.Env.Spaces.Discrete')
                error('OceanStats:RL:Compat:LegacyEnvironment:NotEnumerable', ...
                    'The legacy sampling interface needs an enumerable environment with a Discrete action space.');
            end
            obj.Inner = inner;
            obj.Base = base;
        end

        function setSeed(obj, seed), obj.Inner.setSeed(seed); end

        function s = reset(obj)
            obj.LastTruncated_ = false;
            s = obj.Inner.reset();
        end

        function [s_next, reward, done] = step(obj, action)
            [s_next, reward, terminated, truncated] = obj.Inner.step(action);
            obj.LastTruncated_ = truncated;
            done = terminated || truncated;
        end

        function tf = was_truncated(obj), tf = obj.LastTruncated_; end

        function set_state(obj, s)
            if ~ismethod(obj.Base, 'set_state')
                error('OceanStats:RL:Compat:LegacyEnvironment:NoSetState', '%s does not support set_state.', class(obj.Base));
            end
            obj.Base.set_state(s);
        end

        function actions = get_valid_actions(obj, state), actions = obj.Base.get_valid_actions(state); end
        function states = get_all_states(obj), states = obj.Base.get_all_states(); end
        function tf = is_terminal(obj, state), tf = obj.Base.is_terminal(state); end
        function idx = state_to_index(obj, state), idx = obj.Base.state_to_index(state); end
        function state = index_to_state(obj, idx), state = obj.Base.index_to_state(idx); end
        function k = action_to_index(obj, action), k = obj.Base.action_to_index(action); end
        function n = get_num_states(obj), n = obj.Base.get_num_states(); end
        function n = get_num_actions(obj), n = obj.Inner.ActionSpace.N; end
        function actions = get_all_actions(obj)
            a0 = obj.Inner.ActionSpace.Start;
            actions = a0:(a0 + obj.Inner.ActionSpace.N - 1);
        end

        function [n, p, r] = get_transitions(obj, state, action)
            if ~isa(obj.Base, 'OceanStats.RL.Env.IModelEnvironment')
                error('OceanStats:RL:Compat:LegacyEnvironment:NoModel', ...
                    '%s is sample-only: it does not expose get_transitions.', class(obj.Base));
            end
            [n, p, r] = obj.Base.get_transitions(state, action);
        end

        function env = unwrapped(obj), env = obj.Base; end
    end

    methods (Static)
        function env = adapt(env, requiredClass)
            % ADAPT  Entry point of the sampling solvers. An environment that already is a requiredClass is returned
            % untouched; an algorithm-free OceanStats.RL.Env.IEnvironment is wrapped in the adapter; anything else errors.
            OceanStats.RL.Compat.LegacyEnvironment.mustBeEnvironmentObject_(env);
            if isa(env, requiredClass), return; end
            if isa(env, 'OceanStats.RL.Env.IEnvironment')
                env = OceanStats.RL.Compat.LegacyEnvironment(env);
            end
            if ~isa(env, requiredClass)
                error('OceanStats:RL:Compat:LegacyEnvironment:BadEnvironment', ...
                    'Expected an environment implementing %s (or an OceanStats.RL.Env.IEnvironment), got %s.', requiredClass, class(env));
            end
        end

        function env = adaptModel(env)
            % ADAPTMODEL  Entry point of the exact (model-based) solvers: accepts a legacy DP.TabularEnvironment or any
            % OceanStats.RL.Env.IModelEnvironment. A WRAPPED environment is rejected (WrapperChangesModel): a wrapper may change
            % what an episode is (a time limit truncates it), so silently solving the unwrapped model would answer a different
            % question; pass env.unwrapped() to solve the underlying model on purpose. A sample-only environment is rejected too.
            OceanStats.RL.Compat.LegacyEnvironment.mustBeEnvironmentObject_(env);
            if isa(env, 'OceanStats.RL.DP.TabularEnvironment') || isa(env, 'OceanStats.RL.Env.IModelEnvironment'), return; end
            if isa(env, 'OceanStats.RL.Env.IEnvironment') && isa(env.unwrapped(), 'OceanStats.RL.Env.IModelEnvironment')
                error('OceanStats:RL:Compat:LegacyEnvironment:WrapperChangesModel', ...
                    'A wrapped environment (%s) cannot be solved exactly as it is; pass env.unwrapped() to solve the underlying model.', class(env));
            end
            error('OceanStats:RL:Compat:LegacyEnvironment:NoModel', ...
                '%s does not expose its dynamics (get_transitions): an exact solver needs an IModelEnvironment or a DP.TabularEnvironment.', class(env));
        end
    end

    methods (Static, Access = private)
        function mustBeEnvironmentObject_(env)
            if ~isobject(env) || isa(env, 'matlab.graphics.GraphicsPlaceholder')
                error('OceanStats:RL:Compat:LegacyEnvironment:BadEnvironment', ...
                    'Expected an environment object, got a value of class %s.', class(env));
            end
        end
    end
end
