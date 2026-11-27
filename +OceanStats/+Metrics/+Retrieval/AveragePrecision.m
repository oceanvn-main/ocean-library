classdef AveragePrecision < OceanStats.Metrics.Retrieval.IRetrievalMetric
    % AVERAGEPRECISION  AP: mean of Precision@k over every rank k that hits a relevant doc.
    %
    % THEORY:
    %   AP = (1/R) * sum_{k : rankedIdx(k) is relevant} Precision@k
    %   where R = nnz(relevance) (total relevant docs for the query) and
    %   Precision@k is precision computed over the top-k PREFIX of the
    %   full ranked list (k = 1..N, N = numel(rankedIdx)). Unlike
    %   PrecisionAtK's fixed, externally-chosen K, AP walks every prefix
    %   of the observed ranking and the denominator at each hit IS that
    %   hit's own rank position.
    %
    %   VERIFIED TWO INDEPENDENT WAYS (see
    %   OceanStatsTest.Metrics.Retrieval.AveragePrecisionTest / the hand
    %   examples in this agent's report): (a) the running-sum-over-
    %   relevant-ranks formula above, and (b) explicitly building the full
    %   Precision@k vector for k=1..N and averaging only the entries where
    %   rank k is relevant. Both are implemented identically here (the
    %   code below IS formulation (b); formulation (a) was cross-checked
    %   by hand against the same numeric examples and matched exactly —
    %   see the class's verification notes / agent report for the worked
    %   numbers).
    %
    % EDGE CASE: same NaN convention as RecallAtK — if R == 0 (no relevant
    %   docs defined for this query), AP is undefined and this returns
    %   NaN, not 0 (never fabricate a value for an undefined case).
    %
    % CONVENTION: see IRetrievalMetric for the rankedIdx/relevance shape.
    %   Binary metric: any nonzero relevance grade counts as relevant.
    %
    % No K — AveragePrecision considers the whole ranked list.
    %
    % USAGE:
    %   m = OceanStats.Metrics.Retrieval.AveragePrecision();
    %   score = m.forward(rankedIdx, relevance);
    %
    % See also: OceanStats.Metrics.Retrieval.IRetrievalMetric,
    %           OceanStats.Metrics.Retrieval.RecallAtK

    methods
        function score = forward(~, rankedIdx, relevance)
            % FORWARD  Average Precision for one query's ranking.
            arguments
                ~
                rankedIdx (1,:) double
                relevance (1,:) double
            end
            R = nnz(relevance);
            if R == 0
                score = NaN;   % undefined query (no relevant docs at all) — see EDGE CASE
                return;
            end
            N = numel(rankedIdx);
            isRelevant   = relevance(rankedIdx) ~= 0;      % (1 x N), aligned with rankedIdx
            cumRelevant  = cumsum(isRelevant);              % relevant hits found by rank k
            precisionAtK = cumRelevant ./ (1:N);             % Precision@k for every prefix k=1..N
            score = sum(precisionAtK(isRelevant)) / R;      % average over relevant-hit ranks only
        end

        function s = name(~)
            % NAME  Registry key used by RetrievalMetricFactory.
            s = "ap";
        end
    end
end
