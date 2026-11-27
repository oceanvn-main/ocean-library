classdef ReciprocalRankFusion < OceanNLP.Fusion.IFusionStrategy
    % RECIPROCALRANKFUSION  Rank fusion using Reciprocal Rank Fusion (RRF).
    %
    % PURPOSE:
    %   Aggregates multiple ranked lists of document IDs by scoring each document
    %   with the reciprocal of its rank position: score = sum_i(1 / (K + rank_i)).
    %
    % PROPERTIES:
    %   K - Smoothing constant that controls weight assigned to top vs lower ranks (default: 60)
    %
    % EXAMPLES:
    %   rrf = OceanNLP.Fusion.ReciprocalRankFusion('K', 60);
    %   [ids, scores] = rrf.fuse({[1, 2, 3], [2, 1, 4]});
    %
    % See also: OceanNLP.Fusion.IFusionStrategy, OceanNLP.Fusion.BordaCountFusion

    properties (SetAccess = private)
        K (1,1) double = 60
    end

    methods
        function obj = ReciprocalRankFusion(options)
            arguments
                options.K (1,1) double {mustBeNonnegative} = 60
            end
            obj.K = options.K;
        end

        function [ids, scores] = fuse(obj, ranked)
            arguments
                obj
                ranked cell
            end
            allIds = [ranked{:}];
            uniqueIds = unique(allIds, 'stable');
            scores = zeros(1, numel(uniqueIds));

            for r = 1:numel(ranked)
                list = ranked{r};
                for j = 1:numel(list)
                    docIdx = (uniqueIds == list(j));
                    scores(docIdx) = scores(docIdx) + 1 / (obj.K + j);
                end
            end

            [scores, sortOrder] = sort(scores, 'descend');
            ids = uniqueIds(sortOrder);
        end
    end
end
