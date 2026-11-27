classdef Sampler < handle
    % SAMPLER  Everything a learner needs to know about the environment, and nothing else: it owns the interaction
    % (reset / step), the index encoding (state index, action index <-> action value), the admissible-action table, the
    % per-episode step cap (MaxSteps => truncation) and the handling of a terminal state that lies outside the tabulated
    % state space (e.g. a Blackjack bust). The update rules never touch the environment.
    %
    %   sm = OceanStats.RL.TD.Sampler(env, maxSteps);     env: any object with the sampling interface (see Compat)
    %   sm.seed(s);  si = sm.start();  [s2i, r, term, trunc, done] = sm.step(ai);  sm.capped(done, L)
    %
    % term  = a true MDP terminal was reached (never bootstrap past it); trunc = cut off by the environment or by MaxSteps.
    % NumS / NumA / ValidMask (S x A logical) / ActionValues (action value of each column) describe the table shape.
    properties (Access = private)
        Env
    end
    properties (SetAccess = private)
        NumS (1,1) double = 0
        NumA (1,1) double = 0
        ValidMask logical = logical([])
        ActionValues double = []
        MaxSteps (1,1) double = Inf
        Steps (1,1) double = 0            % environment steps taken since construction
        OutsideLast (1,1) logical = false % the last successor was a terminal outside the table (its index is a placeholder)
    end

    methods
        function obj = Sampler(env, maxSteps)
            arguments
                env
                maxSteps (1,1) double = Inf
            end
            obj.Env = env;
            obj.MaxSteps = maxSteps;
            obj.NumS = env.get_num_states();
            acts = env.get_all_actions();
            obj.NumA = numel(acts);
            obj.ActionValues = zeros(1, obj.NumA);
            for k = 1:obj.NumA, obj.ActionValues(env.action_to_index(acts(k))) = acts(k); end
            states = env.get_all_states();
            obj.ValidMask = false(obj.NumS, obj.NumA);
            for s = 1:obj.NumS
                st = states(s, :);
                if env.is_terminal(st), continue; end
                for a = env.get_valid_actions(st)
                    obj.ValidMask(env.state_to_index(st), env.action_to_index(a)) = true;
                end
            end
        end

        function seed(obj, seed)
            % re-seed the environment's own stream; warn when the environment cannot be seeded
            if ismethod(obj.Env, 'setSeed')
                obj.Env.setSeed(seed);
            else
                warning('OceanStats:RL:TD:TDLearner:EnvNotSeedable', ...
                    'Environment %s has no setSeed(): its randomness is not controlled by Seed, so runs are not reproducible.', class(obj.Env));
            end
        end

        function si = start(obj)
            s = obj.Env.reset();
            si = obj.Env.state_to_index(s);
        end

        function [s2i, r, term, trunc, done] = step(obj, ai)
            [s2, r, done] = obj.Env.step(obj.ActionValues(ai));
            trunc = done && obj.Env.was_truncated();
            term = done && ~trunc;
            obj.Steps = obj.Steps + 1;
            obj.OutsideLast = false;
            try
                s2i = obj.Env.state_to_index(s2);
            catch ME
                if ~term, rethrow(ME); end
                s2i = [];
            end
            if isempty(s2i) || ~isfinite(s2i(1)) || s2i(1) < 1 || s2i(1) > obj.NumS
                % Some environments (e.g. Blackjack) end an episode in a state outside their tabulated state space
                % (a bust score). That is only legitimate for a true terminal, whose value is never read (the
                % bootstrap is dropped), so a placeholder row is used; anywhere else it is an error.
                if ~term
                    error('OceanStats:RL:TD:TDLearner:StateOutsideTable', ...
                        'A non-terminal successor state is outside the environment''s state space.');
                end
                s2i = 1;  obj.OutsideLast = true;
            end
            s2i = s2i(1);
        end

        function tf = capped(obj, done, nsteps)
            % true when the per-episode step cap is reached while the episode has not ended (=> truncation)
            tf = ~done && nsteps >= obj.MaxSteps;
        end
    end
end
