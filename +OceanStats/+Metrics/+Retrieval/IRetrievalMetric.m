classdef (Abstract) IRetrievalMetric < handle
    % IRETRIEVALMETRIC  Abstract interface for a single-query ranking-quality metric.
    %
    %   PyTorch-style: forward() is the compute entry point (mirrors
    %   OceanStats.Similarity.ISimilarity). Stateless strategy — any
    %   per-metric config (K for @K metrics) is a constructor property,
    %   like CosineSimilarity's Eps.
    %
    %   CONVENTION (frozen, see Planning/System/Retrieval/
    %   Retrieval_System_Master_Plan.md §3.2 — this is the ONE relevance
    %   shape used by all five concrete metrics, no per-metric
    %   special-casing of the ground-truth format):
    %     rankedIdx — (1 x N) ranked ORIGINAL document indices, as returned
    %                 by OceanNLP.Retrieval.IRanker.rank() (already sorted
    %                 descending by relevance).
    %     relevance — (1 x Ntotal) relevance grade per ORIGINAL document id
    %                 (Ntotal = full corpus size, indexed the same way
    %                 rankedIdx is indexed). 0 = not relevant. Binary
    %                 metrics (PrecisionAtK, RecallAtK, AveragePrecision,
    %                 ReciprocalRank) treat any nonzero grade as relevant;
    %                 NDCGAtK uses the graded value directly.
    %     score     — (1,1) double
    %
    %   See also: OceanStats.Metrics.Retrieval.RetrievalMetricFactory
    methods (Abstract)
        score = forward(obj, rankedIdx, relevance)
        s = name(obj)
    end
end
