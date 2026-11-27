classdef (Abstract) IDistributionModel < OceanAlgo.StateSearch.ISearchModel
    % IDISTRIBUTIONMODEL  Capability: the full one-step outcome distribution of an action (Sutton & Barto 2nd
    % ed., Section 8.1, "distribution model"). Required by expected backups (DepthLimitedSearch) and by the
    % deterministic engines (AStarSearch, IDAStarSearch, BeamSearch), which additionally require exactly one
    % outcome per action.
    %
    %   o = model.outcomes(s, a)    K x 1 struct array with fields
    %         Next          next state (opaque; ignored by engines when Terminated is true)
    %         Reward        finite real scalar
    %         Terminated    logical: the episode ends with this transition (no continuation value)
    %         Probability   in (0, 1], summing to 1 over the K outcomes
    %
    % Use OceanAlgo.StateSearch.checkOutcomes(o) to validate an outcome list.
    %
    % See also: OceanAlgo.StateSearch.ISearchModel, OceanAlgo.StateSearch.DistributionSampleModel
    methods (Abstract)
        o = outcomes(obj, s, a)
    end
end
