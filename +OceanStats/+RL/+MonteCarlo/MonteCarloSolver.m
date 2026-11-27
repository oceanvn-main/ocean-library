classdef MonteCarloSolver < handle
    % MONTECARLOSOLVER Generic Monte Carlo Solver Context Class for tabular MDPs.
    %
    % Refactored into a lightweight Context class that delegates algorithm execution
    % to modular strategy classes in +OceanStats/+RL/+MonteCarlo/+Strategies/
    % and importance weighting updates to +OceanStats/+RL/+MonteCarlo/+Weighting/.
    %
    % ALGORITHMS (Algorithm property):
    %   "FirstVisitPrediction"       → FirstVisitPredictionStrategy
    %   "EveryVisitPrediction"       → EveryVisitPredictionStrategy
    %   "ExploringStarts"            → ExploringStartsStrategy
    %   "OnPolicyControl"            → OnPolicyControlStrategy
    %   "OffPolicyPrediction"        → OffPolicyPredictionStrategy
    %   "OffPolicyControl"           → OffPolicyControlStrategy
    %   "DiscountingAwarePrediction" → DiscountingAwarePredictionStrategy (S&B §5.8)
    %
    % IMPORTANCE WEIGHTING SCHEMES (ISType & DiscountingAware properties):
    %   Ordinary IS                   → OrdinaryISWeighting (Eq 5.5)
    %   Weighted IS                   → WeightedISWeighting (Eq 5.6)
    %   Discounting-Aware Ordinary IS → DiscountingAwareOrdinaryISWeighting (Eq 5.9)
    %   Discounting-Aware Weighted IS → DiscountingAwareWeightedISWeighting (Eq 5.10)
    %
    % 3-PILLAR COMPONENT CONTRACT:
    %   - Pillar 1 (Symbolic φ): getMetadata() returning LaTeX formulas and symbols dict.
    %   - Pillar 2 (Computational f): run() delegating to active strategy.
    %   - Pillar 3 (Diagram D): getPipelineSpec() returning input/output port spec.
    %
    % Package: OceanStats.RL.MonteCarlo
    % Reference: Sutton & Barto (2018) Chapter 5 & Section 5.8

    % -----------------------------------------------------------------------
    properties
        Environment    % Injected instance of IEpisodeEnvironment
        Algorithm      (1,1) string {mustBeMember(Algorithm, [ ...
                           "FirstVisitPrediction", "EveryVisitPrediction", ...
                           "ExploringStarts", "OnPolicyControl",           ...
                           "OffPolicyPrediction", "OffPolicyControl",       ...
                           "DiscountingAwarePrediction",                   ...
                           "DiscountingAwareControl"                       ...
                       ])} = "FirstVisitPrediction"
        Gamma          (1,1) double {mustBeNonnegative, mustBeLessThanOrEqual(Gamma, 1)} = 1.0
        NumEpisodes    (1,1) double {mustBePositive, mustBeInteger} = 500000
        MaxEpisodeLen  (1,1) double {mustBePositive, mustBeInteger} = 500

        Epsilon        (1,1) double {mustBeNonnegative, mustBeLessThanOrEqual(Epsilon, 1)} = 0.1
        TargetPolicy   function_handle = function_handle.empty()
        BehaviorPolicy function_handle = function_handle.empty()
        ISType         (1,1) string {mustBeMember(ISType, ["Weighted", "Ordinary"])} = "Weighted"
        DiscountingAware (1,1) logical = false
        ForceStartState double = []
        InitialQValue  (1,1) double = 0.0
        TieBreak       (1,1) string {mustBeMember(TieBreak, ["first", "random"])} = "first"
                       % ties among greedy actions: "first" (legacy) or uniformly "random"
        TruncatedEpisodes (1,1) string {mustBeMember(TruncatedEpisodes, ["learn", "skip"])} = "learn"
                       % episodes cut off at MaxEpisodeLen: "learn" (legacy) or "skip" (MC needs
                       % complete episodes -- a cut-off return is not a return; S&B Ch. 5)

        % Output tables
        QTable         double
        VTable         double
        PolicyTable    double

        % Internal accumulators
        NQTable        double
        NVTable        double
        CTable         double
        VAccumulator   OceanEngine.Accumulators.IAccumulator = OceanEngine.Accumulators.WeightedMeanAccumulator.empty()
        QAccumulator   OceanEngine.Accumulators.IAccumulator = OceanEngine.Accumulators.WeightedMeanAccumulator.empty()

        CheckInterval  (1,1) double {mustBePositive, mustBeInteger} = 10000
        Verbose        (1,1) logical = true
        History        struct
    end

    % -----------------------------------------------------------------------
    methods
        function obj = MonteCarloSolver(env, opts)
            arguments
                env
                opts.Algorithm        (1,1) string  = "FirstVisitPrediction"
                opts.Gamma            (1,1) double  = 1.0
                opts.NumEpisodes      (1,1) double  = 500000
                opts.MaxEpisodeLen    (1,1) double  = 500
                opts.Epsilon          (1,1) double  = 0.1
                opts.TargetPolicy     = []
                opts.BehaviorPolicy   = []
                opts.ISType           (1,1) string  = "Weighted"
                opts.DiscountingAware (1,1) logical = false
                opts.ForceStartState  double        = []
                opts.InitialQValue    (1,1) double  = 0.0
                opts.TieBreak         (1,1) string  = "first"
                opts.TruncatedEpisodes (1,1) string = "learn"
                opts.CheckInterval    (1,1) double  = 10000
                opts.Verbose          (1,1) logical = true
            end

            env = OceanStats.RL.Compat.LegacyEnvironment.adapt(env, 'OceanStats.RL.MonteCarlo.IEpisodeEnvironment');
            obj.Environment     = env;
            obj.Algorithm       = opts.Algorithm;
            obj.Gamma           = opts.Gamma;
            obj.NumEpisodes     = opts.NumEpisodes;
            obj.MaxEpisodeLen   = opts.MaxEpisodeLen;
            obj.Epsilon         = opts.Epsilon;
            obj.ISType          = opts.ISType;
            obj.DiscountingAware= opts.DiscountingAware;
            obj.ForceStartState = opts.ForceStartState;
            obj.InitialQValue   = opts.InitialQValue;
            obj.TieBreak        = opts.TieBreak;
            obj.TruncatedEpisodes = opts.TruncatedEpisodes;
            obj.CheckInterval   = opts.CheckInterval;
            obj.Verbose         = opts.Verbose;

            if ~isempty(opts.TargetPolicy)
                obj.TargetPolicy = opts.TargetPolicy;
            end
            if ~isempty(opts.BehaviorPolicy)
                obj.BehaviorPolicy = opts.BehaviorPolicy;
            end

            obj.initialize_tables_();
        end

        % -------------------------------------------------------------------
        function [V, Q, pi, history] = run(obj)
            % RUN Executes configured algorithm via Strategy pattern delegation.
            obj.initialize_tables_();
            obj.History = struct( ...
                'Algorithm',    obj.Algorithm,              ...
                'NumEpisodes',  obj.NumEpisodes,            ...
                'Gamma',        obj.Gamma,                  ...
                'CheckPoints',  [],                         ...
                'AvgReturn',    [],                         ...
                'VSnapshot',    {{}},                       ...
                'QSnapshot',    {{}}                        ...
            );

            if obj.Verbose
                fprintf('\n[MonteCarloSolver] Algorithm: %s | Episodes: %d | gamma=%.2f\n', ...
                    obj.Algorithm, obj.NumEpisodes, obj.Gamma);
            end

            strategy = obj.get_strategy_();
            [V, Q, pi, history] = strategy.execute(obj, obj.Environment);

            if obj.Verbose
                fprintf('[MonteCarloSolver] Done.\n\n');
            end
        end

        % -------------------------------------------------------------------
        function strategy = get_strategy_(obj)
            % GET_STRATEGY_ Factory method for Monte Carlo strategy instances.
            if obj.DiscountingAware
                if obj.Algorithm == "OffPolicyControl" || obj.Algorithm == "DiscountingAwareControl"
                    strategy = OceanStats.RL.MonteCarlo.Strategies.DiscountingAwareControlStrategy();
                else
                    strategy = OceanStats.RL.MonteCarlo.Strategies.DiscountingAwarePredictionStrategy();
                end
                return;
            end

            switch obj.Algorithm
                case "FirstVisitPrediction"
                    strategy = OceanStats.RL.MonteCarlo.Strategies.FirstVisitPredictionStrategy();
                case "EveryVisitPrediction"
                    strategy = OceanStats.RL.MonteCarlo.Strategies.EveryVisitPredictionStrategy();
                case "ExploringStarts"
                    strategy = OceanStats.RL.MonteCarlo.Strategies.ExploringStartsStrategy();
                case "OnPolicyControl"
                    strategy = OceanStats.RL.MonteCarlo.Strategies.OnPolicyControlStrategy();
                case "OffPolicyPrediction"
                    strategy = OceanStats.RL.MonteCarlo.Strategies.OffPolicyPredictionStrategy();
                case "OffPolicyControl"
                    strategy = OceanStats.RL.MonteCarlo.Strategies.OffPolicyControlStrategy();
                case "DiscountingAwarePrediction"
                    strategy = OceanStats.RL.MonteCarlo.Strategies.DiscountingAwarePredictionStrategy();
                case "DiscountingAwareControl"
                    strategy = OceanStats.RL.MonteCarlo.Strategies.DiscountingAwareControlStrategy();
                otherwise
                    error('MonteCarloSolver:UnknownAlgorithm', ...
                        'Algorithm "%s" is not recognized.', obj.Algorithm);
            end
        end

        % -------------------------------------------------------------------
        function scheme = get_weighting_scheme_(obj)
            % GET_WEIGHTING_SCHEME_ Factory method for Importance Weighting schemes.
            if obj.DiscountingAware || obj.Algorithm == "DiscountingAwarePrediction"
                if obj.ISType == "Ordinary"
                    scheme = OceanStats.RL.MonteCarlo.Weighting.DiscountingAwareOrdinaryISWeighting();
                else
                    scheme = OceanStats.RL.MonteCarlo.Weighting.DiscountingAwareWeightedISWeighting();
                end
            else
                if obj.ISType == "Ordinary"
                    scheme = OceanStats.RL.MonteCarlo.Weighting.OrdinaryISWeighting();
                else
                    scheme = OceanStats.RL.MonteCarlo.Weighting.WeightedISWeighting();
                end
            end
        end

        % -------------------------------------------------------------------
        function ep = generate_episode_fast_(obj, policy_fn, varargin)
            force_state  = [];
            force_action = NaN;
            returns_prob = false;
            for k = 1:2:numel(varargin)
                switch varargin{k}
                    case 'ForceStartState',   force_state  = varargin{k+1};
                    case 'ForceStartAction',  force_action = varargin{k+1};
                    case 'PolicyReturnsProb', returns_prob = varargin{k+1};
                end
            end

            ep = obj.Environment.generate_episode(policy_fn, ...
                'MaxLen',            obj.MaxEpisodeLen, ...
                'ForceStartState',   force_state,       ...
                'ForceStartAction',  force_action,      ...
                'PolicyReturnsProb', returns_prob       ...
            );
        end

        function action = greedy_action_(obj, state, env)
            all_acts = env.get_valid_actions(state);
            if isempty(all_acts)
                action = 1;
                return;
            end

            s_idx = env.state_to_index(state);
            if s_idx == 0
                action = all_acts(randi(numel(all_acts)));
                return;
            end

            act_indices = env.action_to_index(all_acts);
            q_vals      = obj.QAccumulator.getValue();
            q_valid     = q_vals(s_idx, act_indices);
            [~, best_i] = max(q_valid);
            action      = all_acts(best_i);
        end

        function [action, prob] = epsilon_greedy_action_(obj, state, env)
            % Epsilon-greedy behaviour b. prob = b(action | state) at the moment of choice:
            %   epsilon / m  +  (1 - epsilon) / k  if action is one of the k tied greedy actions.
            % TieBreak "first" keeps the legacy choice (k = 1: the first maximum).
            all_acts = env.get_valid_actions(state);
            if isempty(all_acts)
                action = 1; prob = 1;
                return;
            end
            n_acts = numel(all_acts);

            explore = rand() < obj.Epsilon;
            s_idx = env.state_to_index(state);
            if s_idx == 0                                   % unknown state: uniform either way
                action = all_acts(randi(n_acts)); prob = 1 / n_acts;
                return;
            end
            act_indices = env.action_to_index(all_acts);
            q_vals      = obj.QAccumulator.getValue();
            q_valid     = q_vals(s_idx, act_indices);
            if obj.TieBreak == "random"
                best = find(q_valid == max(q_valid));
            else
                [~, best] = max(q_valid);
            end
            if explore
                pick = randi(n_acts);
            elseif isscalar(best)
                pick = best;
            else
                pick = best(randi(numel(best)));
            end
            action = all_acts(pick);
            prob   = obj.Epsilon / n_acts + (1 - obj.Epsilon) * ismember(pick, best) / numel(best);
        end

        function action = pick_random_action_(~, state, env)
            all_acts = env.get_valid_actions(state);
            if isempty(all_acts)
                action = 1;
            else
                action = all_acts(randi(numel(all_acts)));
            end
        end

        function policy_fn = get_target_policy_fn_(obj)
            if isempty(obj.TargetPolicy) || ...
               (isnumeric(obj.TargetPolicy) && isempty(obj.TargetPolicy))
                policy_fn = @(s, env) obj.greedy_action_(s, env);
            else
                policy_fn = obj.TargetPolicy;
            end
        end

        function policy_fn = get_behavior_policy_fn_(obj)
            if isempty(obj.BehaviorPolicy) || ...
               (isnumeric(obj.BehaviorPolicy) && isempty(obj.BehaviorPolicy))
                policy_fn = @(s, env) obj.epsilon_greedy_action_(s, env);
            else
                policy_fn = obj.BehaviorPolicy;
            end
        end

        function update_policy_at_state_(obj, s_idx, env)
            q_vals     = obj.QAccumulator.getValue();
            all_states = env.get_all_states();
            s_vec      = all_states(s_idx, :);
            valid_acts = env.get_valid_actions(s_vec);
            if isempty(valid_acts), return; end

            act_indices = env.action_to_index(valid_acts);
            q_valid     = q_vals(s_idx, act_indices);
            [~, best_i] = max(q_valid);
            obj.PolicyTable(s_idx) = act_indices(best_i);
        end

        function b_prob = get_behavior_prob_(obj, s_idx, a_idx, env)
            if isempty(obj.BehaviorPolicy) || ...
               (isnumeric(obj.BehaviorPolicy) && isempty(obj.BehaviorPolicy))
                all_states = env.get_all_states();
                s_vec      = all_states(s_idx, :);
                valid_acts = env.get_valid_actions(s_vec);
                m_valid    = numel(valid_acts);
                if m_valid == 0
                    b_prob = 1.0;
                    return;
                end
                target_a = obj.PolicyTable(s_idx);
                if a_idx == target_a
                    b_prob = (1.0 - obj.Epsilon) + (obj.Epsilon / m_valid);
                else
                    b_prob = obj.Epsilon / m_valid;
                end
            else
                b_prob = obj.BehaviorPolicy(s_idx, a_idx);
            end
        end

        function initialize_tables_(obj)
            n = obj.Environment.get_num_states();
            a = obj.Environment.get_num_actions();

            obj.VAccumulator = OceanEngine.Accumulators.WeightedMeanAccumulator([n 1]);
            obj.QAccumulator = OceanEngine.Accumulators.WeightedMeanAccumulator([n a], 'InitValue', obj.InitialQValue);

            obj.PolicyTable = ones(n, 1);
            obj.sync_tables_from_accumulators_();
        end

        function sync_tables_from_accumulators_(obj)
            if ~isempty(obj.VAccumulator)
                obj.VTable  = obj.VAccumulator.getValue();
                obj.NVTable = obj.VAccumulator.getWeight();
            end
            if ~isempty(obj.QAccumulator)
                obj.QTable  = obj.QAccumulator.getValue();
                obj.NQTable = obj.QAccumulator.getWeight();
                obj.CTable  = obj.QAccumulator.getWeight();
            end
        end

        function finalize_tables_(obj)
            obj.sync_tables_from_accumulators_();
            n_states = obj.Environment.get_num_states();

            % Update PolicyTable and VTable for control algorithms
            q_vals = obj.QAccumulator.getValue();
            for s_idx = 1:n_states
                all_states = obj.Environment.get_all_states();
                s_vec = all_states(s_idx, :);
                valid_acts = obj.Environment.get_valid_actions(s_vec);
                if ~isempty(valid_acts)
                    act_indices = obj.Environment.action_to_index(valid_acts);
                    q_valid = q_vals(s_idx, act_indices);
                    [max_v, best_i] = max(q_valid);
                    if obj.TieBreak == "random"                 % consistent with the behaviour / target ties
                        ties = find(q_valid == max_v);
                        best_i = ties(randi(numel(ties)));
                    end
                    obj.PolicyTable(s_idx) = act_indices(best_i);
                    if any(obj.NQTable(:) > 0)
                        obj.VTable(s_idx) = max_v;
                    end
                end
            end
        end

        function maybe_checkpoint_(obj, ep_num, return_acc)
            if mod(ep_num, obj.CheckInterval) ~= 0 && ep_num ~= obj.NumEpisodes
                return;
            end

            obj.sync_tables_from_accumulators_();

            k = numel(obj.History.CheckPoints) + 1;
            obj.History.CheckPoints(k)  = ep_num;
            obj.History.AvgReturn(k)    = return_acc / ep_num;
            obj.History.VSnapshot{k}    = obj.VTable;
            obj.History.QSnapshot{k}    = obj.QTable;

            if obj.Verbose
                fprintf('  Episode %7d / %d | Avg Return: %+.4f\n', ...
                    ep_num, obj.NumEpisodes, obj.History.AvgReturn(k));
            end
        end

        function validate_off_policy_handles_(obj, mode)
            if nargin < 2, mode = 'both'; end
            needs_target   = strcmp(mode, 'both');
            needs_behavior = true;

            if needs_target && (isempty(obj.TargetPolicy) || ...
               (isnumeric(obj.TargetPolicy) && isempty(obj.TargetPolicy)))
                error('MonteCarloSolver:MissingTargetPolicy', ...
                    'Off-policy prediction requires setting TargetPolicy handle.');
            end
            if needs_behavior && (isempty(obj.BehaviorPolicy) || ...
               (isnumeric(obj.BehaviorPolicy) && isempty(obj.BehaviorPolicy)))
                if obj.Algorithm ~= "OffPolicyControl" && obj.Algorithm ~= "DiscountingAwareControl"
                    error('MonteCarloSolver:MissingBehaviorPolicy', ...
                        'Off-policy algorithms require setting BehaviorPolicy handle.');
                end
            end
        end
    end

    % -----------------------------------------------------------------------
    % 3-PILLAR COMPONENT CONTRACT METADATA METHODS
    % -----------------------------------------------------------------------
    methods (Static)
        function meta = getMetadata()
            meta = struct( ...
                'ClassName', "OceanStats.RL.MonteCarlo.MonteCarloSolver", ...
                'Package', "+OceanStats/+RL/+MonteCarlo/MonteCarloSolver.m", ...
                'Category', "OceanStats.RL", ...
                'TextbookRef', "Sutton & Barto, Reinforcement Learning: An Introduction (2nd Ed.), Chapter 5 & Section 5.8", ...
                'TextbookEq', "V(s) = E_\pi[G_t | S_t=s], \quad Q(s,a) = E_\pi[G_t | S_t=s, A_t=a]", ...
                'SymbolicLatex', "V_\pi(s) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \, \middle| \, S_t = s \right]", ...
                'Symbols', [ ...
                    struct('Symbol', 'V_\pi(s)', 'Description', 'State-value function under policy pi'), ...
                    struct('Symbol', 'Q_\pi(s,a)', 'Description', 'Action-value function for state s and action a'), ...
                    struct('Symbol', '\gamma', 'Description', 'Discount factor in [0, 1]'), ...
                    struct('Symbol', 'G_t', 'Description', 'Discounted return from step t'), ...
                    struct('Symbol', '\rho_{t:T-1}', 'Description', 'Importance sampling ratio product for off-policy learning'), ...
                    struct('Symbol', 'G_t^{\text{disc}}', 'Description', 'Discounting-aware composite partial return for gamma < 1') ...
                ], ...
                'MetadataText', "Unified Monte Carlo Solver context class utilizing Strategy pattern algorithms and decoupled IImportanceWeighting schemes.", ...
                'Methods', [ ...
                    "MonteCarloSolver(env, opts) -> initialized solver context", ...
                    "run() -> [V, Q, pi, history] execution", ...
                    "get_strategy_() -> active IMonteCarloStrategy", ...
                    "get_weighting_scheme_() -> active IImportanceWeighting" ...
                ] ...
            );
        end

        function spec = getPipelineSpec()
            spec = struct( ...
                'ComponentName', "MonteCarloSolver", ...
                'Kind', "NONVIRTUAL", ...
                'Inputs', [ ...
                    struct('Name', "Environment", 'Domain', "RL-Environment", 'Type', "IEpisodeEnvironment"), ...
                    struct('Name', "Algorithm", 'Domain', "String", 'Type', "string"), ...
                    struct('Name', "TargetPolicy", 'Domain', "RL-Policy", 'Type', "function_handle"), ...
                    struct('Name', "BehaviorPolicy", 'Domain', "RL-Policy", 'Type', "function_handle") ...
                ], ...
                'Outputs', [ ...
                    struct('Name', "VTable", 'Domain', "Real", 'Type', "double"), ...
                    struct('Name', "QTable", 'Domain', "Real", 'Type', "double"), ...
                    struct('Name', "PolicyTable", 'Domain', "Integer", 'Type', "double"), ...
                    struct('Name', "History", 'Domain', "Struct", 'Type', "struct") ...
                ] ...
            );
        end
    end
end
