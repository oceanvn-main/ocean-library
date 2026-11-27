classdef ModifiedCarRentalEnv < OceanStats.RL.Env.IModelEnvironment
    % MODIFIEDCARRENTALENV Jack's Car Rental environment (Sutton & Barto Ex 4.2 & Ex 4.7).
    %
    % Models the continuous finite MDP for Jack's Car Rental with configurable
    % parameters for capacity, action bounds, Poisson rental/return dynamics,
    % employee shuttle discount, and parking threshold penalties.
    %
    % Model-only environment (IModelEnvironment): it exposes its exact dynamics but has no sampler (reset/step).
    % Actions are the net number of cars moved from location 1 to 2, -MaxMove..MaxMove.
    
    properties
        MaxCars          (1,1) double {mustBePositive, mustBeInteger} = 20
        MaxMove          (1,1) double {mustBeNonnegative, mustBeInteger} = 5
        RentReward       (1,1) double = 10
        MoveCost         (1,1) double = 2
        ParkingThreshold (1,1) double = 10
        ParkingPenalty   (1,1) double = 4
        FreeShuttle      (1,1) logical = true
        ReqLambda        (1,2) double = [3, 4]
        RetLambda        (1,2) double = [3, 2]
        PoissonCutoff    (1,1) double {mustBePositive, mustBeInteger} = 11
        
        % Cached Probability Tables & Precomputed Transition Matrices
        AllStates          double % (N_states x 2) matrix of [n1, n2]
        NumStatesTotal     double % N_states = (MaxCars + 1)^2
        Loc1TransitionProb double % (MaxCars+1) x (MaxCars+1) transition matrix for loc 1
        Loc2TransitionProb double % (MaxCars+1) x (MaxCars+1) transition matrix for loc 2
        Loc1ExpectedReward double % (MaxCars+1) x 1 expected rental reward vector for loc 1
        Loc2ExpectedReward double % (MaxCars+1) x 1 expected rental reward vector for loc 2
    end
    
    methods
        function obj = ModifiedCarRentalEnv(opts)
            % MODIFIEDCARRENTALENV Constructor initializing environment options and caching distributions.
            arguments
                opts.MaxCars          (1,1) double = 20
                opts.MaxMove          (1,1) double = 5
                opts.RentReward       (1,1) double = 10
                opts.MoveCost         (1,1) double = 2
                opts.ParkingThreshold (1,1) double = 10
                opts.ParkingPenalty   (1,1) double = 4
                opts.FreeShuttle      (1,1) logical = true
                opts.ReqLambda        (1,2) double = [3, 4]
                opts.RetLambda        (1,2) double = [3, 2]
                opts.PoissonCutoff    (1,1) double = 11
            end
            
            obj.MaxCars          = opts.MaxCars;
            obj.MaxMove          = opts.MaxMove;
            obj.RentReward       = opts.RentReward;
            obj.MoveCost         = opts.MoveCost;
            obj.ParkingThreshold = opts.ParkingThreshold;
            obj.ParkingPenalty   = opts.ParkingPenalty;
            obj.FreeShuttle      = opts.FreeShuttle;
            obj.ReqLambda        = opts.ReqLambda;
            obj.RetLambda        = opts.RetLambda;
            obj.PoissonCutoff    = opts.PoissonCutoff;
            
            obj.initialize_state_space_();
            obj.precompute_poisson_tables_();
        end
        
        function states = get_all_states(obj)
            % GET_ALL_STATES Returns all possible state pairs [n1, n2].
            states = obj.AllStates;
        end
        
        function actions = get_valid_actions(obj, state)
            % GET_VALID_ACTIONS Returns valid discrete actions for state [n1, n2].
            n1 = state(1);
            n2 = state(2);
            
            % Action 'a' moves 'a' cars from Location 1 to Location 2
            % Constraints: cannot move more than n1 from loc 1, cannot move more than n2 from loc 2
            min_action = max(-obj.MaxMove, -n2);
            max_action = min(obj.MaxMove, n1);
            
            actions = min_action:max_action;
        end
        
        function [next_states, probabilities, immediate_rewards] = get_transitions(obj, state, action)
            % GET_TRANSITIONS Computes next state distribution and expected reward.
            % Input:
            %   state  - [n1, n2] state vector
            %   action - scalar action 'a' (cars moved from loc 1 to loc 2)
            
            n1 = state(1);
            n2 = state(2);
            
            % 1. Overnight move & capacity capping
            n1_overnight = min(obj.MaxCars, max(0, n1 - action));
            n2_overnight = min(obj.MaxCars, max(0, n2 + action));
            
            % 2. Calculate Action Cost & Penalties
            if action > 0 && obj.FreeShuttle
                % First car moved 1->2 is free
                moving_cars_charged = action - 1;
            else
                moving_cars_charged = abs(action);
            end
            moving_cost = obj.MoveCost * moving_cars_charged;
            
            % Parking lot penalty step-function
            parking_penalty = 0;
            if n1_overnight > obj.ParkingThreshold
                parking_penalty = parking_penalty + obj.ParkingPenalty;
            end
            if n2_overnight > obj.ParkingThreshold
                parking_penalty = parking_penalty + obj.ParkingPenalty;
            end
            
            action_cost = moving_cost + parking_penalty;
            
            % 3. Look up single-location expected rewards and transition distributions
            % (Indexing is 1-based, so car count + 1)
            idx1 = n1_overnight + 1;
            idx2 = n2_overnight + 1;
            
            exp_rental_reward = obj.Loc1ExpectedReward(idx1) + obj.Loc2ExpectedReward(idx2);
            exp_immediate_reward = exp_rental_reward - action_cost;
            
            % Outer product of location 1 and location 2 transition probabilities
            p1 = obj.Loc1TransitionProb(:, idx1); % (MaxCars+1) x 1
            p2 = obj.Loc2TransitionProb(:, idx2); % (MaxCars+1) x 1
            
            P_matrix = p1 * p2'; % (MaxCars+1) x (MaxCars+1) where row is n1_next, col is n2_next
            
            next_states = obj.AllStates;
            probabilities = P_matrix(:); % Linearized matching AllStates row order
            immediate_rewards = repmat(exp_immediate_reward, obj.NumStatesTotal, 1);
        end
        
        function meta = getMetadata(obj)
            meta = struct('SymbolicLatex', strings(1, 0), 'Symbols', containers.Map(), ...
                'Assumptions', {{sprintf('model-only environment: %d x %d cars (MaxCars=%d), actions = net cars moved, -%d..%d; Poisson requests/returns', ...
                obj.MaxCars + 1, obj.MaxCars + 1, obj.MaxCars, obj.MaxMove, obj.MaxMove)}}, ...
                'IsSymbolic', false, 'Reference', "Sutton & Barto (2018) Example 4.2 / Exercise 4.7");
        end
        function spec = getPipelineSpec(~)
            spec = struct('ComponentName', "ModifiedCarRentalEnv", ...
                'Inputs', struct('Name', "state,action", 'Domain', "RL-Action", 'Dtype', "double", 'Shape', "1 x 3"), ...
                'Outputs', struct('Name', "next,prob,reward", 'Domain', "RL-Transition", 'Dtype', "double", 'Shape', "K x 4"), ...
                'Kind', "NONVIRTUAL", 'DiagramSymbol', struct('Type', "default", 'Label', "car-rental-model"), 'ComposesFrom', {{}});
        end

        function tf = is_terminal(~, ~)
            % a continuing task: no terminal states
            tf = false;
        end

        function k = action_to_index(obj, action)
            % net cars moved, -MaxMove..MaxMove  ->  1..2*MaxMove+1
            k = action + obj.MaxMove + 1;
        end

        function idx = state_to_index(obj, state)
            % STATE_TO_INDEX Fast O(1) state vector to linear index converter.
            n1 = state(1);
            n2 = state(2);
            idx = n1 * (obj.MaxCars + 1) + n2 + 1;
        end
        
        function state = index_to_state(obj, idx)
            % INDEX_TO_STATE Fast O(1) linear index to state vector converter.
            idx_zero = idx - 1;
            n1 = floor(idx_zero / (obj.MaxCars + 1));
            n2 = mod(idx_zero, obj.MaxCars + 1);
            state = [n1, n2];
        end
    end
    
    methods (Access = private)
        function initialize_state_space_(obj)
            % Precomputes all grid pairs [n1, n2] and indexing parameters.
            cars_range = 0:obj.MaxCars;
            [N1, N2] = meshgrid(cars_range, cars_range);
            % Transpose meshgrid so state index order is n1 primary, n2 secondary
            N1_t = N1';
            N2_t = N2';
            obj.AllStates = [N1_t(:), N2_t(:)];
            obj.NumStatesTotal = size(obj.AllStates, 1);
        end
        
        function precompute_poisson_tables_(obj)
            % Computes cached Poisson PMFs and per-location transition matrices.
            k_range = 0:obj.PoissonCutoff;
            
            p_req1 = obj.poisson_pmf_(k_range, obj.ReqLambda(1));
            p_req2 = obj.poisson_pmf_(k_range, obj.ReqLambda(2));
            p_ret1 = obj.poisson_pmf_(k_range, obj.RetLambda(1));
            p_ret2 = obj.poisson_pmf_(k_range, obj.RetLambda(2));
            
            % Normalize truncated distributions
            p_req1 = p_req1 / sum(p_req1);
            p_req2 = p_req2 / sum(p_req2);
            p_ret1 = p_ret1 / sum(p_ret1);
            p_ret2 = p_ret2 / sum(p_ret2);
            
            % Precompute transition probabilities and expected rental rewards per location
            [obj.Loc1TransitionProb, obj.Loc1ExpectedReward] = obj.build_location_tables_(...
                p_req1, p_ret1, obj.ReqLambda(1));
            [obj.Loc2TransitionProb, obj.Loc2ExpectedReward] = obj.build_location_tables_(...
                p_req2, p_ret2, obj.ReqLambda(2));
        end
        
        function [P_trans, exp_reward] = build_location_tables_(obj, p_req, p_ret, lambda_req)
            % Builds (MaxCars+1) x (MaxCars+1) transition matrix and expected rewards for 1 location.
            num_car_states = obj.MaxCars + 1;
            P_trans = zeros(num_car_states, num_car_states); % P_trans(next_state, init_state)
            exp_reward = zeros(num_car_states, 1);
            
            cutoff_len = length(p_req);
            
            for init_cars = 0:obj.MaxCars
                init_idx = init_cars + 1;
                total_reward_acc = 0;
                
                for r_idx = 1:cutoff_len
                    req = r_idx - 1;
                    prob_r = p_req(r_idx);
                    
                    actual_rent = min(init_cars, req);
                    total_reward_acc = total_reward_acc + prob_r * (actual_rent * obj.RentReward);
                    
                    cars_after_rent = init_cars - actual_rent;
                    
                    for ret_idx = 1:cutoff_len
                        ret = ret_idx - 1;
                        prob_ret = p_ret(ret_idx);
                        
                        joint_prob = prob_r * prob_ret;
                        end_cars = min(obj.MaxCars, cars_after_rent + ret);
                        end_idx = end_cars + 1;
                        
                        P_trans(end_idx, init_idx) = P_trans(end_idx, init_idx) + joint_prob;
                    end
                end
                
                exp_reward(init_idx) = total_reward_acc;
            end
        end
        
        function pmf = poisson_pmf_(~, k, lambda)
            % POISSON_PMF Standard Poisson PMF formula lambda^k * exp(-lambda) / k!
            pmf = exp(-lambda) .* (lambda.^k) ./ arrayfun(@factorial, k);
        end
    end
end
