classdef BordaCountFusion < OceanNLP.Fusion.IFusionStrategy
    % BORDACOUNTFUSION  Rank aggregation by position-based voting (Borda Count).
    %
    % PURPOSE:
    %   Aggregates multiple ranked lists of document IDs using Borda count voting.
    %   For a list of length N, a document at rank position j (1-indexed) receives
    %   (N - j) points. Scores from all lists are summed to produce final fused rankings.
    %
    % EXAMPLES:
    %   borda = OceanNLP.Fusion.BordaCountFusion();
    %   [ids, scores] = borda.fuse({[1, 2, 3], [2, 1, 4]});
    %
    % See also: OceanNLP.Fusion.IFusionStrategy, OceanNLP.Fusion.ReciprocalRankFusion

    methods
        function [ids, scores] = fuse(~, ranked)
            arguments
                ~
                ranked cell
            end
            allIds = unique([ranked{:}], 'stable');
            scores = zeros(1, numel(allIds));

            for r = 1:numel(ranked)
                list = ranked{r};
                nDocs = numel(list);
                for j = 1:nDocs
                    docIdx = (allIds == list(j));
                    scores(docIdx) = scores(docIdx) + (nDocs - j); %First document get N-1 points, Second get N-2 points, and so on.
                end
            end

            [scores, sortOrder] = sort(scores, 'descend');
            ids = allIds(sortOrder);
        end
    end
end
