classdef ReciprocalRank < OceanStats.Metrics.Retrieval.IRetrievalMetric
    % RECIPROCALRANK  RR: 1 / (rank of the first relevant document in rankedIdx).
    %
    % THEORY:
    %   RR = 1 / rank_of_first_relevant_doc_in_rankedIdx
    %
    % EDGE CASES (two DISTINCT cases — deliberately NOT conflated, this is
    %   the easy place to get backwards):
    %     1. nnz(relevance) == 0 — no relevant docs are defined for this
    %        query at all. The query itself is undefined, same as
    %        RecallAtK/AveragePrecision -> NaN.
    %     2. nnz(relevance) > 0 but NONE of those relevant docs appear
    %        anywhere in rankedIdx — a genuine retrieval failure, NOT an
    %        undefined query (the query has a well-defined relevant set;
    %        the ranker simply never surfaced any of it) -> RR = 0. This
    %        zero is well-defined and correct (the limit of 1/rank as
    %        rank -> infinity), not a fabricated placeholder, so it must
    %        NOT be NaN.
    %
    % CONVENTION: see IRetrievalMetric for the rankedIdx/relevance shape.
    %   Binary metric: any nonzero relevance grade counts as relevant.
    %
    % No K — ReciprocalRank considers the whole ranked list, stopping at
    %   the first relevant hit.
    %
    % USAGE:
    %   m = OceanStats.Metrics.Retrieval.ReciprocalRank();
    %   score = m.forward(rankedIdx, relevance);
    %
    % See also: OceanStats.Metrics.Retrieval.IRetrievalMetric

    methods
        function score = forward(~, rankedIdx, relevance)
            % FORWARD  Reciprocal Rank for one query's ranking.
            arguments
                ~
                rankedIdx (1,:) double
                relevance (1,:) double
            end
            if nnz(relevance) == 0
                score = NaN;   % case 1: undefined query — see EDGE CASES
                return;
            end
            firstHit = find(relevance(rankedIdx) ~= 0, 1, 'first');
            if isempty(firstHit)
                score = 0;      % case 2: well-defined retrieval failure — see EDGE CASES
            else
                score = 1 / firstHit;
            end
        end

        function s = name(~)
            % NAME  Registry key used by RetrievalMetricFactory.
            s = "rr";
        end
    end
end
