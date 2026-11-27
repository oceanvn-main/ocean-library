classdef RecallAtK < OceanStats.Metrics.Retrieval.IRetrievalMetric
    % RECALLATK  Recall@K: fraction of ALL relevant docs found in the top K.
    %
    % THEORY:
    %   Recall@K = |{relevant docs in rankedIdx(1:K)}| / |{all relevant docs}|
    %            = |{relevant docs in rankedIdx(1:K)}| / nnz(relevance)
    %
    % EDGE CASE (explicit — "no faking it," same honesty principle as
    %   everywhere else in this codebase, e.g. R7 in
    %   .agents/pipeline-architecture-spec.md): if nnz(relevance) == 0
    %   (zero relevant documents are defined for this query at all),
    %   Recall is mathematically undefined (0/0). This class returns NaN,
    %   NOT 0 — returning 0 would fabricate an answer for an undefined
    %   case rather than compute one.
    %
    % POLICY: same "missing positions past numel(rankedIdx) count as
    %   non-relevant, K stays fixed" handling as PrecisionAtK.
    %
    % CONVENTION: see IRetrievalMetric for the rankedIdx/relevance shape.
    %   Binary metric: any nonzero relevance grade counts as relevant.
    %
    % K is a required constructor name-value argument (no default), same
    %   reasoning as PrecisionAtK.
    %
    % USAGE:
    %   m = OceanStats.Metrics.Retrieval.RecallAtK('K', 5);
    %   score = m.forward(rankedIdx, relevance);
    %
    % See also: OceanStats.Metrics.Retrieval.IRetrievalMetric,
    %           OceanStats.Metrics.Retrieval.PrecisionAtK

    properties (SetAccess = immutable)
        % NOTE: no {mustBePositive, mustBeInteger} validator on the
        % property itself — see PrecisionAtK's identical note. The real
        % validation (positive integer, REQUIRED — no default) lives on
        % options.K in the constructor's arguments block below.
        K (1,1) double
    end

    methods
        function obj = RecallAtK(options)
            % RECALLATK  Construct with a required rank cutoff K.
            arguments
                options.K (1,1) double {mustBePositive, mustBeInteger}
            end
            obj.K = options.K;
        end

        function score = forward(obj, rankedIdx, relevance)
            % FORWARD  Recall@K for one query's ranking.
            arguments
                obj
                rankedIdx (1,:) double
                relevance (1,:) double
            end
            totalRelevant = nnz(relevance);
            if totalRelevant == 0
                score = NaN;   % undefined query (no relevant docs at all) — see EDGE CASE
                return;
            end
            topK = rankedIdx(1:min(obj.K, numel(rankedIdx)));
            numRelevant = nnz(relevance(topK) ~= 0);
            score = numRelevant / totalRelevant;
        end

        function s = name(~)
            % NAME  Registry key used by RetrievalMetricFactory.
            s = "recall@k";
        end
    end
end
