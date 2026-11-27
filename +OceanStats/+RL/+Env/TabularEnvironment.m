classdef (Abstract) TabularEnvironment < OceanStats.RL.Env.IEnvironment & OceanStats.RL.Env.IModelEnvironment
    % TABULARENVIRONMENT  A small finite MDP that is BOTH sampled (IEnvironment) and exactly modelled
    % (IModelEnvironment), built from ONE transition table so the two views cannot disagree.
    %
    % States and actions are integer indices: ObservationSpace = Discrete(S), ActionSpace = Discrete(A), obs is a
    % scalar state index. A subclass builds
    %   model{s,a}   struct('Next', K x 1, 'Prob', K x 1, 'Reward', K x 1)     ([] = action not admissible in s)
    % and calls setModel_(model, terminalMask, startProbs). Terminal states have no admissible actions.
    %
    % step() returns terminated = (next state is terminal) and truncated = false; a time limit is the
    % Wrappers.TimeLimit wrapper, not a property. info.action_mask is the admissible-action row of the NEW state.
    % Randomness: after setSeed / reset('Seed') an env-owned RandStream is used and the global stream is never touched;
    % until then the global stream is used (so rng(k) makes an unseeded run reproducible).
    % Optional RewardStd(s,a): zero-mean Gaussian noise added on top of the model reward when SAMPLING
    % (get_transitions returns the mean reward, so the exact model is the expected MDP).

    properties (SetAccess = protected)
        NumStates (1,1) double = 0
        NumActions (1,1) double = 0
        TerminalMask (:,1) logical
        StartProbs (:,1) double
        Model cell
        ValidMask logical = logical([])
        RewardStd double = []
    end
    properties (Access = protected)
        Current (1,1) double = 1
        Stream = []                 % empty: use the global stream
        Ready_ (1,1) logical = false
    end

    methods
        function setSeed(obj, seed)
            obj.Stream = RandStream('mt19937ar', 'Seed', seed);
        end

        function [obs, reward, terminated, truncated, info] = step(obj, action)
            if ~obj.Ready_
                error('OceanStats:RL:Env:TabularEnvironment:NotReset', 'Call reset() before step().');
            end
            s = obj.Current;
            if ~obj.ActionSpace.contains(action) || obj.TerminalMask(s) || isempty(obj.Model{s, action})
                error('OceanStats:RL:Env:TabularEnvironment:InvalidAction', ...
                    'Action %s is not admissible in state %d.', mat2str(action), s);
            end
            m = obj.Model{s, action};
            c = cumsum(m.Prob);
            rs = obj.stream_();
            k = find(rand(rs) * c(end) <= c, 1);
            reward = m.Reward(k);
            if obj.RewardStd(s, action) > 0
                reward = reward + obj.RewardStd(s, action) * randn(rs);
            end
            obs = m.Next(k);
            obj.Current = obs;
            terminated = obj.TerminalMask(obs);
            truncated = false;
            info = struct('action_mask', obj.ValidMask(obs, :));
        end

        function set_state(obj, s)
            % SET_STATE  Force the current state (used to start an episode from a chosen state).
            obj.Current = s(1);
            obj.Ready_ = true;
        end

        % ---- IModelEnvironment
        function [next_states, probabilities, immediate_rewards] = get_transitions(obj, state, action)
            if ~obj.ActionSpace.contains(action) || isempty(obj.Model{state(1), action})
                error('OceanStats:RL:Env:TabularEnvironment:InvalidAction', ...
                    'Action %s is not admissible in state %d.', mat2str(action), state(1));
            end
            m = obj.Model{state(1), action};
            next_states = m.Next(:);
            probabilities = m.Prob(:);
            immediate_rewards = m.Reward(:);
        end
        function actions = get_valid_actions(obj, state)
            actions = find(obj.ValidMask(state(1), :));
        end
        function tf = is_terminal(obj, state), tf = obj.TerminalMask(state(1)); end
        function states = get_all_states(obj), states = (1:obj.NumStates)'; end
        function idx = state_to_index(~, state), idx = state(1); end
        function k = action_to_index(~, action), k = action; end

        function V = exactPolicyValues(obj, policyProbs, gamma)
            % EXACTPOLICYVALUES  Exact V_pi for a stochastic policy by solving (I - gamma P_pi) V = r_pi.
            %   policyProbs (S x A) action probabilities (rows of terminal states ignored); gamma in [0,1].
            % Ground truth for prediction validation (independent of any sampling or estimation code).
            S = obj.NumStates;
            P = zeros(S, S);  r = zeros(S, 1);
            for s = 1:S
                if obj.TerminalMask(s), continue; end
                for a = obj.get_valid_actions(s)
                    m = obj.Model{s, a};
                    for k = 1:numel(m.Next)
                        P(s, m.Next(k)) = P(s, m.Next(k)) + policyProbs(s, a) * m.Prob(k);
                        r(s) = r(s) + policyProbs(s, a) * m.Prob(k) * m.Reward(k);
                    end
                end
            end
            V = (eye(S) - gamma * P) \ r;
        end

        function pi = uniformPolicy(obj)
            % UNIFORMPOLICY  Equiprobable random policy over the admissible actions (S x A).
            pi = zeros(obj.NumStates, obj.NumActions);
            for s = 1:obj.NumStates
                v = obj.get_valid_actions(s);
                pi(s, v) = 1 / numel(v);
            end
        end

        function meta = getMetadata(obj)
            meta = getMetadata@OceanStats.RL.Env.IEnvironment(obj);
            meta.Assumptions = [meta.Assumptions, {sprintf('tabular MDP: %d states, %d actions; exact model via get_transitions, sampled via step', ...
                obj.NumStates, obj.NumActions)}];
            meta.Reference = "tabular MDP (p(s',r|s,a) table)";
        end
    end

    methods (Access = protected)
        function [obs, info] = doReset_(obj)
            c = cumsum(obj.StartProbs);
            obj.Current = find(rand(obj.stream_()) * c(end) <= c, 1);
            obj.Ready_ = true;
            obs = obj.Current;
            info = struct('action_mask', obj.ValidMask(obs, :));
        end

        function setModel_(obj, model, terminalMask, startProbs)
            if ~iscell(model) || ndims(model) ~= 2
                error('OceanStats:RL:Env:TabularEnvironment:BadModel', 'The model must be a 2-D cell array (states x actions).');
            end
            [S, A] = size(model);
            terminalMask = logical(terminalMask(:));
            startProbs = startProbs(:);
            if numel(terminalMask) ~= S || numel(startProbs) ~= S
                error('OceanStats:RL:Env:TabularEnvironment:BadModel', ...
                    'terminalMask and startProbs must have one entry per state (%d).', S);
            end
            if any(~isfinite(startProbs)) || any(startProbs < 0) || sum(startProbs) <= 0
                error('OceanStats:RL:Env:TabularEnvironment:BadModel', 'startProbs must be finite and non-negative with a positive sum.');
            end
            if any(startProbs(terminalMask) > 0)
                error('OceanStats:RL:Env:TabularEnvironment:BadModel', 'A terminal state cannot be a start state.');
            end
            mask = false(S, A);
            for s = 1:S
                for a = 1:A
                    m = model{s, a};
                    if isempty(m), continue; end
                    if terminalMask(s)
                        error('OceanStats:RL:Env:TabularEnvironment:BadModel', 'Terminal state %d must have no admissible action.', s);
                    end
                    if ~(numel(m.Next) == numel(m.Prob) && numel(m.Prob) == numel(m.Reward))
                        error('OceanStats:RL:Env:TabularEnvironment:BadModel', ...
                            'Model{%d,%d}: Next, Prob and Reward must have the same length.', s, a);
                    end
                    if ~all(isfinite(m.Prob(:))) || ~all(isfinite(m.Reward(:)))
                        error('OceanStats:RL:Env:TabularEnvironment:BadModel', 'Model{%d,%d}: Prob and Reward must be finite.', s, a);
                    end
                    if any(m.Prob < 0) || abs(sum(m.Prob) - 1) > 1e-9
                        error('OceanStats:RL:Env:TabularEnvironment:BadModel', 'Model{%d,%d}: Prob must be non-negative and sum to 1.', s, a);
                    end
                    if any(m.Next < 1 | m.Next > S | m.Next ~= fix(m.Next))
                        error('OceanStats:RL:Env:TabularEnvironment:BadModel', 'Model{%d,%d}: Next must be state indices in 1..%d.', s, a, S);
                    end
                    mask(s, a) = true;
                end
                if ~terminalMask(s) && ~any(mask(s, :))
                    error('OceanStats:RL:Env:TabularEnvironment:BadModel', 'Non-terminal state %d has no admissible action.', s);
                end
            end
            obj.Model = model;
            obj.NumStates = S;  obj.NumActions = A;
            obj.TerminalMask = terminalMask;
            obj.StartProbs = startProbs / sum(startProbs);
            obj.ValidMask = mask;
            obj.RewardStd = zeros(S, A);
            obj.ObservationSpace = OceanStats.RL.Env.Spaces.Discrete(S);
            obj.ActionSpace = OceanStats.RL.Env.Spaces.Discrete(A);
        end

        function rs = stream_(obj)
            if isempty(obj.Stream), rs = RandStream.getGlobalStream(); else, rs = obj.Stream; end
        end
    end
end
