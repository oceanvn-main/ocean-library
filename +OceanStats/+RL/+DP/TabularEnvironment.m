classdef (Abstract) TabularEnvironment < handle
    % TABULARENVIRONMENT Abstract base class for tabular Markov Decision Process (MDP) environments.
    %
    % Defines the decoupled interface contract between domain physics and
    % mathematical solvers (such as Policy Iteration, Value Iteration, Q-Learning).
    %
    % Package: OceanStats.RL.DP
    %
    % Abstract Methods Required:
    %   get_all_states(obj)                - Returns matrix of all valid states (N_states x D)
    %   get_valid_actions(obj, state)      - Returns vector of valid discrete actions for state
    %   get_transitions(obj, state, action)- Returns successor states, probabilities, and rewards
    
    methods (Abstract)
        % GET_ALL_STATES Returns all possible states in the environment.
        % Output:
        %   states - (N_states x D) array of state vectors (or indices)
        states = get_all_states(obj)
        
        % GET_VALID_ACTIONS Returns physically permitted actions for a specific state.
        % Input:
        %   state  - (1 x D) state vector or scalar state index
        % Output:
        %   actions - (1 x A) vector of valid action values
        actions = get_valid_actions(obj, state)
        
        % GET_TRANSITIONS Computes transition dynamics for a given state-action pair.
        % Input:
        %   state  - (1 x D) state vector or scalar state index
        %   action - (1 x 1) discrete action
        % Output:
        %   next_states       - (K x D) or (K x 1) array of successor states
        %   probabilities     - (K x 1) vector of transition probabilities summing to 1
        %   immediate_rewards - (K x 1) vector of immediate rewards (or scalar expected reward)
        [next_states, probabilities, immediate_rewards] = get_transitions(obj, state, action)
    end
    
    methods
        function num_states = get_num_states(obj)
            % GET_NUM_STATES Returns total number of states.
            states = obj.get_all_states();
            num_states = size(states, 1);
        end
        
        function idx = state_to_index(obj, state)
            % STATE_TO_INDEX Default linear search converter (subclasses can override for O(1)).
            states = obj.get_all_states();
            [found, idx] = ismember(state, states, 'rows');
            if ~found
                error('TabularEnvironment:InvalidState', 'State vector is not in state space.');
            end
        end
        
        function state = index_to_state(obj, idx)
            % INDEX_TO_STATE Default linear index to state converter.
            states = obj.get_all_states();
            if idx < 1 || idx > size(states, 1)
                error('TabularEnvironment:InvalidIndex', 'State index out of bounds.');
            end
            state = states(idx, :);
        end
    end
end
