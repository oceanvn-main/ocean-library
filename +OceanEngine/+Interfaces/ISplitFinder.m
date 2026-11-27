classdef (Abstract) ISplitFinder < handle
    % ISPLITFINDER  Abstract interface for tree split-finding strategies.
    %
    %   OVERVIEW:
    %     Defines the contract for finding the optimal (feature, threshold)
    %     split given a node's gradient histogram. Different strategies trade
    %     between speed, accuracy, and memory usage.
    %
    %   CONTRACT:
    %     result = findBestSplit(obj, H, dm)
    %
    %   SPLIT RESULT FORMAT:
    %     Implementations must return a struct with fields:
    %       result.FeatureIdx  double   — feature column (1-indexed)
    %       result.BinIdx      double   — bin index (0-indexed, uint8 convention)
    %       result.Threshold   double   — decoded float split point
    %       result.Gain        double   — information gain
    %       result.IsValid     logical  — false if no profitable split found
    %
    %   KNOWN IMPLEMENTATIONS:
    %     HistogramSplitFinder  — O(bins) cumulative scan. XGBoost gain formula.
    %                             Lambda (L2) + Gamma (min gain) regularization.
    %     ExactGreedySplitFinder — O(N log N) sort. Exact splits. (Planned)
    %
    %   EXTENSION:
    %     1. Subclass ISplitFinder in +OceanEngine/+SplitFinders/
    %     2. Implement findBestSplit(H, dm) → SplitResult struct
    %     3. Inject into tree builder via constructor
    %
    %   See also: OceanEngine.SplitFinders.HistogramSplitFinder
    %             OceanEngine.Interfaces.IHistogramBuilder
    %             OceanEngine.Data.DataMatrix

    methods (Abstract)
        % FINDBESTSPLIT  Scan the histogram and return the optimal split.
        %
        %   result = findBestSplit(obj, H, dm)
        %
        %   Inputs:
        %     H  — histogram struct from IHistogramBuilder.build()
        %         (fields: gradSum, hessSum, count — each NumBins×p)
        %     dm — OceanEngine.Data.DataMatrix (needed for threshold decoding)
        %
        %   Output:
        %     result — struct: FeatureIdx, BinIdx, Threshold, Gain, IsValid
        result = findBestSplit(obj, H, dm)

        % NAME  Human-readable label for this split strategy.
        n = name(obj)
    end
end
