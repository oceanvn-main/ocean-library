classdef NDCGAtK < OceanStats.Metrics.Retrieval.IRetrievalMetric
    % NDCGATK  Normalized Discounted Cumulative Gain @ K.
    %
    % THEORY:
    %   DCG@K  = sum_{i=1}^{K} (2^rel_i - 1) / log2(i + 1)
    %            rel_i is the relevance GRADE (not binarized) of the
    %            document at rank i in rankedIdx; rel_i = 0 if rankedIdx
    %            has fewer than i entries (0-padded, same "missing
    %            positions" policy as PrecisionAtK/RecallAtK).
    %   IDCG@K = the same DCG@K formula applied to the IDEAL ranking: sort
    %            ALL grades in relevance descending, take the top K.
    %   NDCG@K = DCG@K / IDCG@K
    %
    % EDGE CASE (standard NDCG convention — DELIBERATELY DIFFERENT from
    %   RecallAtK/AveragePrecision/ReciprocalRank's NaN choice, not an
    %   inconsistency): if IDCG@K == 0 (every grade in relevance is 0, no
    %   relevant docs at all), NDCG@K is defined to be 0, NOT NaN. This
    %   matches the standard IR-literature convention (trec_eval,
    %   pytrec_eval, scikit-learn's ndcg_score all treat "nothing to rank
    %   well" as a trivial 0). The reasoning genuinely differs from
    %   Recall/AP/RR: those measure coverage of a KNOWN relevant set and
    %   are undefined with no such set; NDCG measures how close the
    %   achieved gain is to the best-possible gain, and with zero possible
    %   gain the ranking trivially cannot do better than the (zero) ideal,
    %   so 0 is the well-defined answer, not a fabricated one.
    %
    % CONVENTION: see IRetrievalMetric for the rankedIdx/relevance shape.
    %   Graded metric: uses relevance's actual numeric grade, NOT
    %   binarized (unlike PrecisionAtK/RecallAtK/AveragePrecision/
    %   ReciprocalRank).
    %
    % K is a required constructor name-value argument (no default), same
    %   reasoning as PrecisionAtK/RecallAtK.
    %
    % USAGE:
    %   m = OceanStats.Metrics.Retrieval.NDCGAtK('K', 5);
    %   score = m.forward(rankedIdx, relevance);
    %
    % See also: OceanStats.Metrics.Retrieval.IRetrievalMetric

    properties (SetAccess = immutable)
        % NOTE: no {mustBePositive, mustBeInteger} validator on the
        % property itself — see PrecisionAtK's identical note. The real
        % validation (positive integer, REQUIRED — no default) lives on
        % options.K in the constructor's arguments block below.
        K (1,1) double
    end

    methods
        function obj = NDCGAtK(options)
            % NDCGATK  Construct with a required rank cutoff K.
            arguments
                options.K (1,1) double {mustBePositive, mustBeInteger}
            end
            obj.K = options.K;
        end

        function score = forward(obj, rankedIdx, relevance)
            % FORWARD  NDCG@K for one query's ranking.
            arguments
                obj
                rankedIdx (1,:) double %orderd produced by retrieval model
                relevance (1,:) double %ground truth (graded)
            end
            % 1. Calculate the Actual DCG (Discounted Cumulative Gain)
            % This is the raw score of how well the ranking algorithm did.
            % Highly relevant documents give more gain, but they are discounted
            % if they are placed lower down the list.
            achievedGrades = relevance(rankedIdx);          % graded, aligned with rankedIdx
            dcg  = OceanStats.Metrics.Retrieval.NDCGAtK.dcgAtK_(achievedGrades, obj.K);

            % 2. Calculate the IDCG (Ideal Discounted Cumulative Gain)
            % This is the theoretical maximum possible score for this specific query.
            % We take all known documents, sort them perfectly from highest relevance 
            % to lowest, and calculate the DCG of that perfect list.
            idealGrades = sort(relevance, 'descend');
            idcg = OceanStats.Metrics.Retrieval.NDCGAtK.dcgAtK_(idealGrades, obj.K);

            % 3. Calculate NDCG (Normalized Discounted Cumulative Gain)
            % We divide the actual gain (DCG) by the perfect score (IDCG). 
            % This normalizes the metric to a 0.0 to 1.0 scale.
            if idcg == 0
                score = 0;   % well-defined by NDCG convention — see EDGE CASE (not NaN)
            else
                score = dcg / idcg;
            end
        end

        function s = name(~)
            % NAME  Registry key used by RetrievalMetricFactory.
            s = "ndcg@k";
        end
    end

    methods (Static, Access = private)
        function d = dcgAtK_(grades, K)
            % DCGATK_  sum_{i=1}^{K} (2^grades(i) - 1) / log2(i+1).
            %   grades shorter than K are treated as 0-padded (missing
            %   positions contribute zero gain).
            n = min(K, numel(grades));
            g = zeros(1, K);
            if n > 0
                g(1:n) = grades(1:n);
            end
            i = 1:K;
            d = sum((2 .^ g - 1) ./ log2(i + 1));
        end
    end
end
