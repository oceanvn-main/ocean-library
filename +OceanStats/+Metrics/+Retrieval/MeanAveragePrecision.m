classdef MeanAveragePrecision < OceanStats.Metrics.Retrieval.AveragePrecision
    % MEANAVERAGEPRECISION  MAP: Average Precision averaged over a query set.
    %
    % THEORY (Manning, Raghavan & Schutze, "Introduction to Information
    % Retrieval", eq. 11.16):
    %   MAP = (1/|Q|) * sum_{q in Q} AP(q)
    %   where AP(q) is exactly AveragePrecision's per-query value. For a
    %   single query MAP == AP (the text's own worked example: "The MAP for
    %   the single query (hence = AP) ... is 0.6").
    %
    % EXTENSION, NOT DUPLICATION: this class INHERITS AveragePrecision, so
    %   forward(rankedIdx, relevance) is UNCHANGED — it still returns ONE
    %   query's AP, and a MeanAveragePrecision object is still a valid
    %   IRetrievalMetric anywhere an AveragePrecision is (Liskov-safe; no
    %   contract change). The only additions are aggregate() — the
    %   multi-query mean — and name() = "map". The per-query AP math lives
    %   in AveragePrecision ONLY; nothing is re-implemented here.
    %
    % WHY aggregate() AND NOT forward(): IRetrievalMetric.forward() is a
    %   single-query contract (one rankedIdx, one relevance vector). MAP is
    %   defined over a SET of queries, so it needs a multi-query entry point;
    %   overloading forward() with a different signature would break the
    %   uniform contract RetrievalEvaluator relies on.
    %
    % RELATION TO RetrievalEvaluator: an evaluator run with this metric shows
    %   per-query AP(q) in each query row and MAP in its Mean row (the mean
    %   of the per-query column), which equals aggregate()'s result on the
    %   same rankings — verified by MeanAveragePrecisionTest.
    %
    % EDGE CASES (same honesty convention as AveragePrecision/RecallAtK):
    %   - A query with zero relevant docs has AP = NaN (undefined). aggregate()
    %     EXCLUDES such queries from the mean (|Q| in eq. 11.16 is the number
    %     of queries for which AP is DEFINED) and REPORTS how many were
    %     excluded, rather than silently averaging them in as 0 or letting
    %     one degenerate query poison the mean to NaN.
    %   - If every query is undefined (or Q is empty), MAP is NaN — never a
    %     fabricated 0.
    %
    % USAGE:
    %   m = OceanStats.Metrics.Retrieval.MeanAveragePrecision();
    %   [mapScore, perQueryAP, numExcluded] = m.aggregate(rankedIdxList, relevanceList);
    %     rankedIdxList — (1 x Q) cell, each a (1 x N) ranked-doc-index vector
    %     relevanceList — (1 x Q) cell, each a (1 x Ntotal) relevance vector
    %                     (same contract as IRetrievalMetric, one per query)
    %
    % See also: OceanStats.Metrics.Retrieval.AveragePrecision,
    %           OceanStats.Metrics.Retrieval.IRetrievalMetric,
    %           OceanNLP.Retrieval.RetrievalEvaluator

    methods
        function [mapScore, perQueryAP, numExcluded] = aggregate(obj, rankedIdxList, relevanceList)
            % AGGREGATE  MAP over a query set (eq. 11.16), NaN-excluding.
            arguments
                obj
                rankedIdxList (1,:) cell
                relevanceList (1,:) cell
            end
            Q = numel(rankedIdxList);
            if numel(relevanceList) ~= Q
                error('OceanStats:Metrics:Retrieval:MeanAveragePrecision:QueryCountMismatch', ...
                    ['rankedIdxList has %d queries but relevanceList has %d — they must ' ...
                     'be paired one relevance vector per ranked list.'], Q, numel(relevanceList));
            end

            perQueryAP = zeros(1, Q);
            for q = 1:Q
                perQueryAP(q) = obj.forward(rankedIdxList{q}, relevanceList{q});
            end

            isDefined   = ~isnan(perQueryAP);
            numExcluded = nnz(~isDefined);
            if any(isDefined)
                mapScore = sum(perQueryAP(isDefined)) / nnz(isDefined);   % (1/|Q|) * sum AP(q)
            else
                mapScore = NaN;
            end
        end

        function s = name(~)
            % NAME  Registry key used by RetrievalMetricFactory.
            s = "map";
        end
    end
end
