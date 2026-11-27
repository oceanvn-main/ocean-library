classdef (Abstract) IEnumerableEnvironment < handle
    % IENUMERABLEENVIRONMENT  Capability: the state space is finite and can be listed, so a tabular method can index a
    % table with it. Implemented by environments with structured states (Blackjack: [sum dealer ace]; Racetrack:
    % [x y vx vy]) as well as by the table-built environments. It does NOT expose the dynamics (that is
    % IModelEnvironment); a sample-only environment can be enumerable.
    %
    %   states = env.get_all_states()            N x D, one row per state (row order defines the index)
    %   idx    = env.state_to_index(state)       1..N; 0 for a state that is outside the table (a terminal such as a bust)
    %   state  = env.index_to_state(idx)
    %   k      = env.action_to_index(action)     1..A (actions may be labelled 0..A-1)
    %   actions = env.get_valid_actions(state)   admissible action values ([] for a terminal state; a sample-only
    %                                            environment may return its default set for a terminal observation that is
    %                                            outside its table, so use info.action_mask, which is authoritative)
    %   tf     = env.is_terminal(state)
    methods (Abstract)
        states = get_all_states(obj)
        idx = state_to_index(obj, state)
        k = action_to_index(obj, action)
        actions = get_valid_actions(obj, state)
        tf = is_terminal(obj, state)
    end
    methods
        function n = get_num_states(obj)
            n = size(obj.get_all_states(), 1);
        end
        function state = index_to_state(obj, idx)
            states = obj.get_all_states();
            if idx < 1 || idx > size(states, 1)
                error('OceanStats:RL:Env:IEnumerableEnvironment:InvalidIndex', 'State index %d is out of bounds.', idx);
            end
            state = states(idx, :);
        end
    end
end
