classdef (Abstract) ISimilarity < handle
    % ISIMILARITY  Abstract interface for vector/batch similarity scoring.
    %
    %   PyTorch-style contract: stateless strategy exposing forward()
    %   (elementwise, row-vs-corresponding-row) and pairwise() (all-vs-all,
    %   for retrieval). Mirrors torch.nn.CosineSimilarity /
    %   torch.nn.functional.cosine_similarity.
    %
    %   CONVENTION:
    %     A, B — (N x D) row-major batches: N samples, D-dim feature vectors.
    %     forward(A, B)   -> (N x 1)  A(i,:) vs B(i,:), same-size inputs only
    %     pairwise(A, B)  -> (N x M)  A(i,:) vs B(j,:), all pairs
    %
    %   See also: OceanStats.Similarity.SimilarityFactory
    methods (Abstract)
        s = forward(obj, A, B)
        P = pairwise(obj, A, B)
        s = name(obj)
    end
end
