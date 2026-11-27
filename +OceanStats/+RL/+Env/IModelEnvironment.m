classdef (Abstract) IModelEnvironment < OceanStats.RL.Env.IEnumerableEnvironment
    % IMODELENVIRONMENT  Capability: the environment exposes its exact dynamics p(s', r | s, a) (the "env.P" of
    % Gymnasium's toy-text environments). A capability, not a family: an environment that also implements
    % IEnvironment can be sampled AND solved exactly, from the same table, so the two views cannot disagree.
    % Sample-only environments (Blackjack, Racetrack) simply do not implement it. A model needs an enumerable state
    % space (IEnumerableEnvironment), hence the inheritance.
    %
    %   [next, prob, reward] = env.get_transitions(state, action)     K x D successor states / K x 1 probabilities / K x 1 mean rewards
    methods (Abstract)
        [next_states, probabilities, immediate_rewards] = get_transitions(obj, state, action)
    end
end
