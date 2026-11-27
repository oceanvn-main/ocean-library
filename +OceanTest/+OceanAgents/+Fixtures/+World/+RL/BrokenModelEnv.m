classdef BrokenModelEnv < OceanStats.RL.Env.Classic.RandomWalkEnv
    % BROKENMODELENV  Violates exactly RL-8 (ExactModel): get_transitions claims action 2 (right) moves left, so sampled
    % transitions have zero probability in the declared model.
    methods
        function [next_states, probabilities, immediate_rewards] = get_transitions(obj, state, action)
            [next_states, probabilities, immediate_rewards] = ...
                get_transitions@OceanStats.RL.Env.Classic.RandomWalkEnv(obj, state, action);
            if action == 2, next_states = next_states - 2; end
        end
    end
end
