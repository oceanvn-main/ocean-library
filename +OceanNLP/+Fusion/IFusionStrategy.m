classdef (Abstract) IFusionStrategy < handle
    % IFUSIONSTRATEGY  Abstract strategy interface for rank fusion & aggregation.
    %
    % PURPOSE:
    %   Provides a common contract for merging multiple ranked document ID lists
    %   (from different rankers or expanded queries) into a single fused ranking.
    %
    % See also: OceanNLP.Fusion.ReciprocalRankFusion,
    %           OceanNLP.Fusion.BordaCountFusion,
    %           OceanNLP.Fusion.FusionFactory

    methods (Abstract)
        % FUSE  Fuse a cell array of document ID vectors into a single ranked list.
        %   ranked — cell array of (1 x N) document ID vectors from different retrievers
        %   ids    — (1 x N_unique) fused document ID ranking in descending order
        %   scores — (1 x N_unique) fused numerical scores
        [ids, scores] = fuse(obj, ranked)
    end
end

