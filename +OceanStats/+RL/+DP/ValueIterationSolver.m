classdef ValueIterationSolver < handle
    % VALUEITERATIONSOLVER Decoupled Dynamic Programming Value Iteration Solver.
    %
    % Solves finite tabular Markov Decision Processes (MDPs) using exact
    % Bellman optimality backups and deterministic tie-breaking policy extraction.
    %
    % Interface Dependency: Interacts with OceanStats.RL.DP.TabularEnvironment.
    %
    % Package: OceanStats.RL.DP
    
    properties
        Environment    % Injected instance of TabularEnvironment
        DiscountFactor (1,1) double {mustBeNonnegative, mustBeLessThanOrEqual(DiscountFactor, 1)} = 1.0
        Theta          (1,1) double {mustBePositive} = 1e-9
        ValueVector    double % (N_states x 1) state-value array V(s)
        PolicyVector   double % (N_states x 1) policy action array pi(s)
        SweepHistory   struct % Cached V vectors at specified iteration sweeps
        TieBreakMode   (1,1) string {mustBeMember(TieBreakMode, ["min_stake", "max_stake", "first"])} = "min_stake"
        TieTolerance   (1,1) double {mustBeNonnegative} = 1e-9
        History        struct % Convergence delta history
    end
    
    methods
        function obj = ValueIterationSolver(env, opts)
            % VALUEITERATIONSOLVER Constructor injecting tabular environment and options.
            arguments
                env
                opts.DiscountFactor   (1,1) double = 1.0
                opts.Theta            (1,1) double = 1e-9
                opts.InitialValue     double = []
                opts.TieBreakMode     (1,1) string = "min_stake"
                opts.TieTolerance     (1,1) double = 1e-9
            end
            
            env = OceanStats.RL.Compat.LegacyEnvironment.adaptModel(env);
            obj.Environment    = env;
            obj.DiscountFactor = opts.DiscountFactor;
            obj.Theta          = opts.Theta;
            obj.TieBreakMode   = opts.TieBreakMode;
            obj.TieTolerance   = opts.TieTolerance;
            
            num_states = obj.Environment.get_num_states();
            if isempty(opts.InitialValue)
                obj.ValueVector = zeros(num_states, 1);
            else
                obj.ValueVector = opts.InitialValue(:);
            end
            obj.PolicyVector = zeros(num_states, 1);
            
            obj.SweepHistory = struct('Sweeps', [], 'Labels', {{}}, 'Values', {{}});
            obj.History = struct('Sweeps', 0, 'MaxDelta', []);
        end
        
        function q_val = compute_q_value(obj, state, action)
            % COMPUTE_Q_VALUE Computes Q(s, a) using Bellman optimality operator.
            %
            % Q(s, a) = sum_s' P(s' | s, a) * [ R(s, a, s') + gamma * V(s') ]
            
            [next_states, probabilities, immediate_rewards] = obj.Environment.get_transitions(state, action);
            
            num_next = size(next_states, 1);
            next_indices = zeros(num_next, 1);
            
            for k = 1:num_next
                next_indices(k) = obj.Environment.state_to_index(next_states(k, :));
            end
            
            v_next = obj.ValueVector(next_indices);
            
            if isscalar(immediate_rewards) || (length(immediate_rewards) > 1 && all(immediate_rewards == immediate_rewards(1)))
                q_val = immediate_rewards(1) + obj.DiscountFactor * sum(probabilities .* v_next);
            else
                q_val = sum(probabilities .* (immediate_rewards + obj.DiscountFactor * v_next));
            end
        end
        
        function [delta, max_v] = evaluate_step(obj)
            % EVALUATE_STEP Executes one full sweep of Bellman optimality update.
            %
            % V(s) <- max_a Q(s, a) for all non-terminal states.
            
            states = obj.Environment.get_all_states();
            num_states = size(states, 1);
            
            v_old_all = obj.ValueVector;
            delta = 0;
            
            for s_idx = 1:num_states
                state = states(s_idx, :);
                valid_actions = obj.Environment.get_valid_actions(state);
                
                % Terminal states (no valid actions or action = 0 only with fixed value)
                if isempty(valid_actions)
                    continue;
                end
                
                num_actions = length(valid_actions);
                q_values = zeros(num_actions, 1);
                
                for a_idx = 1:num_actions
                    action = valid_actions(a_idx);
                    q_values(a_idx) = obj.compute_q_value(state, action);
                end
                
                v_new = max(q_values);
                delta = max(delta, abs(v_new - v_old_all(s_idx)));
                obj.ValueVector(s_idx) = v_new;
            end
            
            max_v = max(obj.ValueVector);
        end
        
        function policy = extract_policy(obj, tie_break_mode)
            % EXTRACT_POLICY Extracts optimal greedy policy pi(s) = argmax_a Q(s, a).
            %
            % Uses floating-point tolerance check (1e-9) to handle action value ties.
            arguments
                obj
                tie_break_mode (1,1) string = ""
            end
            
            if tie_break_mode == ""
                mode = obj.TieBreakMode;
            else
                mode = tie_break_mode;
            end
            
            states = obj.Environment.get_all_states();
            num_states = size(states, 1);
            policy = zeros(num_states, 1);
            
            for s_idx = 1:num_states
                state = states(s_idx, :);
                valid_actions = obj.Environment.get_valid_actions(state);
                
                if isempty(valid_actions)
                    policy(s_idx) = 0;
                    continue;
                end
                
                num_actions = length(valid_actions);
                q_values = zeros(num_actions, 1);
                
                for a_idx = 1:num_actions
                    action = valid_actions(a_idx);
                    q_values(a_idx) = obj.compute_q_value(state, action);
                end
                
                max_q = max(q_values);
                % Find all actions within floating-point tolerance of max Q
                tied_indices = find(abs(q_values - max_q) <= obj.TieTolerance);
                tied_actions = valid_actions(tied_indices);
                
                switch mode
                    case "min_stake"
                        chosen_action = min(tied_actions);
                    case "max_stake"
                        chosen_action = max(tied_actions);
                    case "first"
                        chosen_action = tied_actions(1);
                    otherwise
                        chosen_action = min(tied_actions);
                end
                
                policy(s_idx) = chosen_action;
            end
            
            obj.PolicyVector = policy;
        end
        
        function [V, pi, history] = run(obj, opts)
            % RUN Executes complete Value Iteration loop until max delta < Theta.
            arguments
                obj
                opts.MaxIter      (1,1) double {mustBePositive, mustBeInteger} = 1000
                opts.TrackSweeps  double = []
            end
            
            sweep_count = 0;
            delta_history = [];
            
            % Cache initial state (Sweep 0) if requested
            if ismember(0, opts.TrackSweeps)
                obj.cache_sweep_(0);
            end
            
            while sweep_count < opts.MaxIter
                sweep_count = sweep_count + 1;
                
                [delta, ~] = obj.evaluate_step();
                delta_history(end+1) = delta; %#ok<AGROW>
                
                if ismember(sweep_count, opts.TrackSweeps)
                    obj.cache_sweep_(sweep_count);
                end
                
                if delta < obj.Theta
                    break;
                end
            end
            
            % Always cache final converged sweep
            obj.cache_sweep_(sweep_count, 'Final Converged');
            
            % Final Policy Extraction
            pi = obj.extract_policy();
            V = obj.ValueVector;
            
            obj.History.Sweeps   = sweep_count;
            obj.History.MaxDelta = delta_history;
            history = obj.History;
        end
    end
    
    methods (Access = private)
        function cache_sweep_(obj, sweep_num, label)
            if nargin < 3, label = sprintf('Sweep %d', sweep_num); end
            
            if isempty(obj.SweepHistory.Sweeps)
                idx = 1;
            else
                idx = numel(obj.SweepHistory.Sweeps) + 1;
            end
            obj.SweepHistory.Sweeps(idx) = sweep_num;
            obj.SweepHistory.Labels{idx} = label;
            obj.SweepHistory.Values{idx} = obj.ValueVector;
        end
    end
end
