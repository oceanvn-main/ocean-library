classdef PrecisionAtK < OceanStats.Metrics.Retrieval.IRetrievalMetric
    % PRECISIONATK  Precision@K: fraction of the top-K ranked docs that are relevant.
    %
    % THEORY:
    %   Precision@K = |{relevant docs in rankedIdx(1:K)}| / K
    %
    % POLICY (explicit, TREC convention — stated, not left implicit, per
    %   this project's habit, see WhitespaceTokenizer's case-folding
    %   policy for precedent): if numel(rankedIdx) < K, the missing
    %   positions past the end of rankedIdx are treated as NON-RELEVANT
    %   and the denominator STAYS K — Precision@K is NOT rescaled to
    %   min(K, numel(rankedIdx)). A ranker that returns fewer than K
    %   documents is penalized for the missing slots, matching standard
    %   TREC evaluation practice, rather than being scored only on what it
    %   happened to return.
    %
    % CONVENTION: see IRetrievalMetric for the rankedIdx/relevance shape.
    %   Binary metric: any nonzero relevance grade counts as relevant.
    %
    % K is a required constructor name-value argument (no default) —
    %   unlike CosineSimilarity's Eps, there is no universally-sensible
    %   default cutoff, so the caller must state one explicitly.
    %
    % USAGE:
    %   m = OceanStats.Metrics.Retrieval.PrecisionAtK('K', 5);
    %   score = m.forward(rankedIdx, relevance);
    %
    % See also: OceanStats.Metrics.Retrieval.IRetrievalMetric,
    %           OceanStats.Metrics.Retrieval.RecallAtK

    properties (SetAccess = immutable)
        % NOTE: no {mustBePositive, mustBeInteger} validator on the
        % property itself — MATLAB applies a property's validators to its
        % implicit zero-value default at class-property-initialization
        % time (before the constructor body runs), so a strict validator
        % here would fail on every construction regardless of what the
        % constructor later assigns. The real validation (positive
        % integer, REQUIRED — no default) lives on options.K in the
        % constructor's arguments block below, which is the actual
        % enforcement point.
        K (1,1) double
    end

    methods
        function obj = PrecisionAtK(options)
            % PRECISIONATK  Construct with a required rank cutoff K.
            arguments
                options.K (1,1) double {mustBePositive, mustBeInteger}
            end
            obj.K = options.K;
        end

        function score = forward(obj, rankedIdx, relevance)
            % FORWARD  Precision@K for one query's ranking.
            arguments
                obj
                rankedIdx (1,:) double
                relevance (1,:) double
            end
            topK = rankedIdx(1:min(obj.K, numel(rankedIdx)));
            numRelevant = nnz(relevance(topK) ~= 0);
            score = numRelevant / obj.K;   % denominator is K, not min(K, numel(rankedIdx)) — see POLICY
        end

        function s = name(~)
            % NAME  Registry key used by RetrievalMetricFactory.
            s = "precision@k";
        end
    end
end
