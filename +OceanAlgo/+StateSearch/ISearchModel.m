classdef (Abstract) ISearchModel < handle
    % ISEARCHMODEL  The part of a planning model every state-space search needs: which actions are legal in a
    % state and how to recognise a state again. States are opaque values (an index, a vector, a struct, ...);
    % only the model interprets them.
    %
    %   actions = model.actions(s)   1 x K numeric action labels legal in s (empty: no legal action)
    %   k       = model.key(s)       hashable key: a finite real scalar double or a string scalar. Two states
    %                                with equal keys are treated as the same state (closed sets, transposition
    %                                tables, duplicate merging).
    %   tf      = model.isGoal(s)    optional goal test (default false). Reaching a goal ends a path, like a
    %                                terminated transition.
    %
    % Capabilities (a model implements one or both):
    %   IDistributionModel  outcomes(s,a): every next state with its probability (S&B 8.1 "distribution model")
    %   ISampleModel        sample(s,a,stream): one sampled transition (S&B 8.1 "sample model")
    % A deterministic model is a distribution model whose outcomes(s,a) always has exactly one element.
    %
    % Rewards are maximised. Engines that minimise path cost use cost = -reward.
    %
    % See also: OceanAlgo.StateSearch.IDistributionModel, OceanAlgo.StateSearch.ISampleModel
    methods (Abstract)
        actions = actions(obj, s)
        k = key(obj, s)
    end
    methods
        function tf = isGoal(~, ~)
            tf = false;
        end
    end
end
