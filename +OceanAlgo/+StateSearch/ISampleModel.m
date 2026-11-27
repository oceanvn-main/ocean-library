classdef (Abstract) ISampleModel < OceanAlgo.StateSearch.ISearchModel
    % ISAMPLEMODEL  Capability: one sampled transition (Sutton & Barto 2nd ed., Section 8.1, "sample model").
    % Enough for RolloutSearch and MonteCarloTreeSearch, which only simulate.
    %
    %   o = model.sample(s, a, stream)   scalar struct with fields Next, Reward, Terminated
    %     stream   RandStream owned by the engine -- the only allowed source of randomness
    %
    % A distribution model becomes a sample model through OceanAlgo.StateSearch.DistributionSampleModel.
    %
    % See also: OceanAlgo.StateSearch.ISearchModel, OceanAlgo.StateSearch.IDistributionModel
    methods (Abstract)
        o = sample(obj, s, a, stream)
    end
end
