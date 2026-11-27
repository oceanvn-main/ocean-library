classdef PolicyIterationSolver < handle
    % POLICYITERATIONSOLVER Decoupled Dynamic Programming Policy Iteration Solver.
    %
    % Solves finite tabular Markov Decision Processes (MDPs) using exact
    % Policy Evaluation and Policy Improvement routines.
    %
    % Interface Dependency: Interacts purely with OceanStats.RL.DP.TabularEnvironment.
    %
    % Package: OceanStats.RL.DP
    
    properties
        Environment    % Injected instance of TabularEnvironment
        DiscountFactor (1,1) double {mustBeNonnegative, mustBeLessThanOrEqual(DiscountFactor, 1)} = 0.9
        Theta          (1,1) double {mustBePositive} = 1e-4
        ValueVector    double % (N_states x 1) state-value array V(s)
        PolicyVector   double % (N_states x 1) policy action array pi(s)
        ValueMatrix    double % Grid reshaped state value matrix (if 2D state space)
        PolicyMatrix   double % Grid reshaped policy action matrix (if 2D state space)
        History        struct % Convergence and execution history tracking
    end
    
    methods
        function obj = PolicyIterationSolver(env, opts)
            % POLICYITERATIONSOLVER Constructor injecting tabular environment and options.
            arguments
                env
                opts.DiscountFactor   (1,1) double = 0.9
                opts.Theta            (1,1) double = 1e-4
                opts.InitialPolicy    double = []
            end
            
            env = OceanStats.RL.Compat.LegacyEnvironment.adaptModel(env);
            obj.Environment    = env;
            obj.DiscountFactor = opts.DiscountFactor;
            obj.Theta          = opts.Theta;
            
            num_states = obj.Environment.get_num_states();
            obj.ValueVector  = zeros(num_states, 1);
            
            if isempty(opts.InitialPolicy)
                % a LEGAL starting action in every state: 0 where 0 is an action (car rental: move no cars), else
                % the first valid action; terminal states get none (0) and are never evaluated (fixed 2026-10-01:
                % grid worlds have no action 0 and refuse transitions out of a terminal state)
                obj.PolicyVector = zeros(num_states, 1);
                states = obj.Environment.get_all_states();
                for s_idx = 1:num_states
                    if obj.isTerminal_(states(s_idx, :)), continue; end
                    valid = obj.Environment.get_valid_actions(states(s_idx, :));
                    if ~isempty(valid) && ~any(valid == 0)
                        obj.PolicyVector(s_idx) = valid(1);
                    end
                end
            else
                obj.PolicyVector = opts.InitialPolicy(:);
            end
            
            obj.History = struct('PolicySteps', 0, 'EvalSweeps', [], 'MaxDelta', []);
            obj.update_grid_matrices_();
        end
        
        function q_val = compute_q_value(obj, state, action)
            % COMPUTE_Q_VALUE Computes Q(s, a) using Bellman expectation operator.
            %
            % Q(s, a) = sum_s' P(s' | s, a) * [ R(s, a, s') + gamma * V(s') ]
            
            [next_states, probabilities, immediate_rewards] = obj.Environment.get_transitions(state, action);
            
            % Map next states to linear indices
            num_next = size(next_states, 1);
            next_indices = zeros(num_next, 1);
            
            for k = 1:num_next
                next_indices(k) = obj.Environment.state_to_index(next_states(k, :));
            end
            
            v_next = obj.ValueVector(next_indices);
            
            if scalar_reward_check_(immediate_rewards)
                % Expected immediate reward model
                q_val = immediate_rewards(1) + obj.DiscountFactor * sum(probabilities .* v_next);
            else
                % Joint state-transition reward model
                q_val = sum(probabilities .* (immediate_rewards + obj.DiscountFactor * v_next));
            end
        end
        
        function [delta, sweep_count] = evaluate_policy(obj)
            % EVALUATE_POLICY Runs policy evaluation until max state value change < Theta.
            states = obj.Environment.get_all_states();
            num_states = size(states, 1);
            
            sweep_count = 0;
            
            while true
                sweep_count = sweep_count + 1;
                delta = 0;
                
                for s_idx = 1:num_states
                    state = states(s_idx, :);
                    if obj.isTerminal_(state)
                        obj.ValueVector(s_idx) = 0;           % a terminal state is worth 0 (Sutton & Barto 4.1)
                        continue
                    end
                    action = obj.PolicyVector(s_idx);

                    v_old = obj.ValueVector(s_idx);
                    v_new = obj.compute_q_value(state, action);
                    
                    obj.ValueVector(s_idx) = v_new;
                    delta = max(delta, abs(v_old - v_new));
                end
                
                if delta < obj.Theta
                    break;
                end
            end
            
            obj.update_grid_matrices_();
        end
        
        function policy_stable = improve_policy(obj)
            % IMPROVE_POLICY Greedy policy update over state space.
            states = obj.Environment.get_all_states();
            num_states = size(states, 1);
            
            policy_stable = true;
            
            for s_idx = 1:num_states
                state = states(s_idx, :);
                old_action = obj.PolicyVector(s_idx);
                
                if obj.isTerminal_(state), continue; end     % no action is ever taken in a terminal state
                valid_actions = obj.Environment.get_valid_actions(state);
                num_actions = length(valid_actions);
                if num_actions == 0, continue; end
q_values = zeros(num_actions, 1);
                for a_idx = 1:num_actions
                    action = valid_actions(a_idx);
                    q_values(a_idx) = obj.compute_q_value(state, action);
                end
                
                % Argmax with robust precision handling
                [max_q, max_action_idx] = max(q_values);
                best_action = valid_actions(max_action_idx);
                
                % Handle ties gracefully: if old action yields identical max Q value, keep old action
                if ismember(old_action, valid_actions)
                    old_a_idx = find(valid_actions == old_action, 1);
                    if abs(q_values(old_a_idx) - max_q) < 1e-10
                        best_action = old_action;
                    end
                end
                
                if best_action ~= old_action
                    policy_stable = false;
                    obj.PolicyVector(s_idx) = best_action;
                end
            end
            
            obj.update_grid_matrices_();
        end
        
        function [V_mat, pi_mat, history] = run(obj, max_policy_steps)
            % RUN Executes complete Policy Iteration loop (Evaluation + Improvement).
            arguments
                obj
                max_policy_steps (1,1) double {mustBePositive, mustBeInteger} = 50
            end
            
            step_count = 0;
            eval_sweeps_history = [];
            max_delta_history = [];
            
            while step_count < max_policy_steps
                step_count = step_count + 1;
                
                % 1. Policy Evaluation
                [delta, sweeps] = obj.evaluate_policy();
                eval_sweeps_history(end+1) = sweeps; %#ok<AGROW>
                max_delta_history(end+1) = delta; %#ok<AGROW>
                
                % 2. Policy Improvement
                stable = obj.improve_policy();
                
                if stable
                    break;
                end
            end
            
            obj.History.PolicySteps = step_count;
            obj.History.EvalSweeps  = eval_sweeps_history;
            obj.History.MaxDelta    = max_delta_history;
            
            V_mat   = obj.ValueMatrix;
            pi_mat  = obj.PolicyMatrix;
            history = obj.History;
        end
    end
    
    methods (Access = private)
        function tf = isTerminal_(obj, state)
            tf = ismethod(obj.Environment, 'is_terminal') && logical(obj.Environment.is_terminal(state));
        end

        function update_grid_matrices_(obj)
            % Updates 2D grid matrix properties if environment is 2D (e.g. MaxCars x MaxCars).
            if isprop(obj.Environment, 'MaxCars') && ~isempty(obj.Environment.MaxCars)
                grid_dim = obj.Environment.MaxCars + 1;
                obj.ValueMatrix  = reshape(obj.ValueVector, grid_dim, grid_dim);
                obj.PolicyMatrix = reshape(obj.PolicyVector, grid_dim, grid_dim);
            else
                obj.ValueMatrix  = obj.ValueVector;
                obj.PolicyMatrix = obj.PolicyVector;
            end
        end
    end
end

function tf = scalar_reward_check_(rewards)
    % Helper checking if rewards is expected scalar value per state transition
    tf = isscalar(rewards) || (length(rewards) > 1 && all(rewards == rewards(1)));
end
