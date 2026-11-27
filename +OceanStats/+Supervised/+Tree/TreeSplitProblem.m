classdef TreeSplitProblem < handle
    % TREESPLITPROBLEM  Adapts tree splitting to the Generic Greedy Search.
    %
    % TWO PATHS:
    %   FAST PATH — findBestSplit():
    %     Per feature: sort once O(N logN), cumsum O(N), index lookup O(k).
    %     Total: O(N*p*logN + k*p).  No threshold->index conversion needed.
    %
    %   SLOW PATH — GreedySearch.optimize() calling evaluate():
    %     Per candidate: O(N) rssEvaluate. Total: O(N*k*p). ~63x slower.

    properties (Access = private)
        X           % Feature matrix (N x P)
        Y           % Target vector (N x 1)
        MinLeafSize (1,1) double = 1
        EvaluateFn  % function_handle: @(yLeft, yRight) -> scalar score
        IncrementalFn % function_handle: @(y_sorted) -> N-1 scores
        CandidateGenerator OceanStats.Supervised.Tree.ICandidateGenerator = ...
            OceanStats.Supervised.Tree.ExhaustiveCandidateGenerator()
        NumBins     (1,1) double = 64  % quantile bins for fast path
    end

    methods
        function obj = TreeSplitProblem(X, Y, minLeaf, evaluateFn, candidateGenerator, incrementalFn)
            arguments
                X double
                Y
                minLeaf (1,1) double = 1
                evaluateFn = []
                candidateGenerator = []
                incrementalFn = []
            end
            obj.X = X;
            obj.Y = Y;
            obj.MinLeafSize = minLeaf;

            if isempty(evaluateFn)
                obj.EvaluateFn = @rssEvaluate;
            else
                obj.EvaluateFn = evaluateFn;
            end
            
            if isempty(incrementalFn)
                obj.IncrementalFn = @rssIncremental;
            else
                obj.IncrementalFn = incrementalFn;
            end

            if isempty(candidateGenerator)
                import OceanStats.Supervised.Tree.ExhaustiveCandidateGenerator
                obj.CandidateGenerator = ExhaustiveCandidateGenerator();
            else
                obj.CandidateGenerator = candidateGenerator;
                % Mirror NumBins from QuantileCandidateGenerator if available
                if isprop(candidateGenerator, 'NumBins')
                    obj.NumBins = candidateGenerator.NumBins;
                end
            end
        end

        function candidates = generateCandidates(obj)
            candidates = obj.CandidateGenerator.generate(obj.X, obj.Y, obj.MinLeafSize);
        end

        function score = evaluate(obj, candidate)
            % EVALUATE  Per-candidate O(N) evaluation — fallback slow path.
            j = candidate.FeatureIndex;
            t = candidate.Threshold;

            x_col  = obj.X(:, j);
            valid  = ~isnan(x_col);
            x_valid = x_col(valid);
            y_valid = obj.Y(valid);

            maskLeft = x_valid <= t;
            yLeft    = y_valid(maskLeft);
            yRight   = y_valid(~maskLeft);

            if isempty(yLeft) || isempty(yRight)
                score = Inf;
                return;
            end

            score = obj.EvaluateFn(yLeft, yRight);
        end

        function [bestCand, bestScore] = findBestSplit(obj)
            % FINDBESTSPLIT  Fast O(N*p*logN + k*p) best split.
            %
            % SPLIT CANDIDATE STRATEGY — auto-selected per feature:
            %
            %   LOW CARDINALITY PATH  (nUnique <= NumBins):
            %     Uses diff(x_sorted) > 0 to find ALL value-boundary positions.
            %     Correct for discrete/categorical features — never skips a
            %     valid split regardless of how few unique values exist.
            %
            %   HIGH CARDINALITY PATH (nUnique > NumBins):
            %     Original quantile-position sampling via linspace — O(k) lookups.
            %     Efficient for near-continuous features where ties are extremely
            %     rare, so the skip-on-tie check loses negligible information.
            %
            % Both paths share the O(N) rssIncremental cumsum trick.

            [~, P]    = size(obj.X);
            minLeaf   = obj.MinLeafSize;
            k         = obj.NumBins;
            bestScore = Inf;
            bestCand  = [];

            for j = 1:P
                x_col = obj.X(:, j);
                valid = ~isnan(x_col);

                x_valid = x_col(valid);
                y_valid = obj.Y(valid);
                nValid  = numel(y_valid);

                if nValid < 2 * minLeaf
                    continue;
                end

                % Sort feature ONCE
                [x_sorted, ord] = sort(x_valid);
                y_sorted        = y_valid(ord);

                % RSS/Impurity for ALL N-1 positions — one O(N) vectorized pass
                scores_all = obj.IncrementalFn(y_sorted);

                % Count unique values from the sorted array — O(N), no extra alloc
                nUnique = 1 + sum(diff(x_sorted) > 0);

                if nUnique <= k
                    % ---- LOW CARDINALITY: exact boundary positions ----
                    % Find every position where x_sorted changes value.
                    % All are valid split points; no tie-skip needed.
                    positions = find(diff(x_sorted) > 0);
                    positions = positions( ...
                        positions >= minLeaf & positions <= nValid - minLeaf);
                    if isempty(positions), continue; end

                else
                    % ---- HIGH CARDINALITY: quantile position sampling ----
                    % Original fast path. Ties are rare for near-continuous
                    % features, so rejecting them loses negligible splits.
                    nPos = min(k - 1, nValid - 2 * minLeaf);
                    if nPos < 1, continue; end
                    positions = unique(round(linspace(minLeaf, nValid - minLeaf, nPos)));
                    positions = positions( ...
                        positions >= minLeaf & positions <= nValid - minLeaf);
                    % Vectorized tie-skip (avoids per-iteration branch)
                    positions = positions( ...
                        x_sorted(positions) ~= x_sorted(positions + 1));
                    if isempty(positions), continue; end
                end

                for pos = positions(:)'
                    s = scores_all(pos);
                    if s < bestScore
                        bestScore             = s;
                        bestCand.FeatureIndex = j;
                        % Threshold = midpoint — avoids boundary ambiguity
                        bestCand.Threshold    = 0.5 * (x_sorted(pos) + x_sorted(pos + 1));
                    end
                end
            end
        end

    end
end

function score = rssEvaluate(yLeft, yRight)
    rssLeft  = sum((yLeft  - mean(yLeft)).^2);
    rssRight = sum((yRight - mean(yRight)).^2);
    score    = rssLeft + rssRight;
end

function scores = rssIncremental(y_sorted)
    % RSSINCREMENTAL  All N-1 RSS split scores in one O(N) vectorized pass.

    N        = numel(y_sorted);
    y2       = y_sorted .^ 2;
    cumSumY  = cumsum(y_sorted);
    cumSumY2 = cumsum(y2);

    totSumY  = cumSumY(end);
    totSumY2 = cumSumY2(end);

    k      = (1:N-1)';
    sumYL  = cumSumY(k);
    sumY2L = cumSumY2(k);
    sumYR  = totSumY  - sumYL;
    sumY2R = totSumY2 - sumY2L;

    rssL   = sumY2L - sumYL.^2 ./ k;
    rssR   = sumY2R - sumYR.^2 ./ (N - k);
    scores = rssL + rssR;
end
