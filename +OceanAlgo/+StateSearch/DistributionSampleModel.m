classdef DistributionSampleModel < OceanAlgo.StateSearch.ISampleModel
    % DISTRIBUTIONSAMPLEMODEL  Sample model obtained from a distribution model: draws one outcome of
    % outcomes(s,a) with its probability (inverse-CDF on one uniform draw from the engine's stream).
    %
    %   sm = OceanAlgo.StateSearch.DistributionSampleModel(distributionModel);
    %   o  = sm.sample(s, a, stream)        struct: Next, Reward, Terminated
    properties (SetAccess = immutable)
        Source      % OceanAlgo.StateSearch.IDistributionModel
    end
    methods
        function obj = DistributionSampleModel(source)
            arguments
                source (1,1) OceanAlgo.StateSearch.IDistributionModel
            end
            obj.Source = source;
        end
        function a = actions(obj, s), a = obj.Source.actions(s); end
        function k = key(obj, s), k = obj.Source.key(s); end
        function tf = isGoal(obj, s), tf = obj.Source.isGoal(s); end
        function o = sample(obj, s, a, stream)
            arguments
                obj
                s
                a (1,1) double
                stream (1,1) RandStream
            end
            d = OceanAlgo.StateSearch.checkOutcomes(obj.Source.outcomes(s, a));
            k = find(rand(stream) <= cumsum([d.Probability]), 1, 'first');
            if isempty(k), k = numel(d); end            % guard against cumsum rounding just below 1
            o = struct('Next', {d(k).Next}, 'Reward', d(k).Reward, 'Terminated', logical(d(k).Terminated));
        end
    end
end
