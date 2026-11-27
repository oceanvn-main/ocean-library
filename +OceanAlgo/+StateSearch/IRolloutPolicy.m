classdef (Abstract) IRolloutPolicy < handle
    % IROLLOUTPOLICY  Strategy: the rollout policy that picks actions in simulated trajectories (Sutton & Barto
    % 2nd ed., Section 8.10; in Section 8.11 it acts outside the tree).
    %
    %   a = policy.act(model, s, stream)
    %     model    OceanAlgo.StateSearch.ISearchModel (gives the legal actions)
    %     stream   RandStream owned by the engine -- the only allowed source of randomness
    %     a        a legal action label of s
    %
    % See also: OceanAlgo.StateSearch.UniformRandomRolloutPolicy, OceanAlgo.StateSearch.rolloutReturn
    methods (Abstract)
        a = act(obj, model, s, stream)
    end
    methods
        function name = describe(obj), name = string(class(obj)); end
    end
end
