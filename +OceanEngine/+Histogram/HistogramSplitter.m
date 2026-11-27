classdef HistogramSplitter < handle
    % HISTOGRAMSPLITTER  Finds the optimal split by scanning a gradient histogram.
    %
    %   OVERVIEW:
    %     Given a histogram built by HistogramBuilder, this class performs a
    %     left-to-right cumulative scan over all bins for all features to find
    %     the split (feature, bin) that maximizes the XGBoost gain formula.
    %
    %   TIME COMPLEXITY:
    %     O(NumBins * p) per node — independent of N (number of samples).
    %     For 256 bins, the entire histogram fits in L1 CPU cache → extremely fast.
    %
    %   GAIN FORMULA (XGBoost, Eq. 7):
    %     Gain = 0.5 * [G_L^2/(H_L + λ) + G_R^2/(H_R + λ) - G_P^2/(H_P + λ)] - γ
    %
    %     Where:
    %       G_L, G_R = sum of gradients in left / right partition
    %       H_L, H_R = sum of hessians in left / right partition
    %       G_P, H_P = parent totals
    %       λ (Lambda) = L2 regularization on leaf weights (prevents large weights)
    %       γ (Gamma)  = minimum gain threshold to accept a split (pruning)
    %
    %   USAGE:
    %     splitter = OceanEngine.Histogram.HistogramSplitter(dataMatrix);
    %     result   = splitter.findBestSplit(H);
    %
    %     result.FeatureIdx   — which feature column (1-indexed)
    %     result.BinIdx       — which bin (0-indexed uint8)
    %     result.Threshold    — decoded float threshold (from BinEdges)
    %     result.Gain         — gain achieved by this split
    %     result.IsValid      — false if no profitable split found
    %
    %   See also: OceanEngine.Histogram.HistogramBuilder
    %             OceanEngine.Data.DataMatrix

    properties
        % Lambda  L2 regularization on leaf weights. Prevents large leaf
        %         predictions; higher = more conservative tree. Default: 1.
        Lambda double = 1.0

        % Gamma  Minimum gain required to accept a split. Acts as a natural
        %        pruning mechanism — splits below Gamma are ignored. Default: 0.
        Gamma  double = 0.0
    end

    properties (SetAccess = private)
        DataMatrix_  % OceanEngine.Data.DataMatrix
    end

    methods
        function obj = HistogramSplitter(dataMatrix, options)
            % HISTOGRAMSPLITTER  Constructor.
            %
            %   splitter = HistogramSplitter(dataMatrix)
            %   splitter = HistogramSplitter(dataMatrix, 'Lambda', 1.0, 'Gamma', 0.1)
            arguments
                dataMatrix OceanEngine.Data.DataMatrix
                options.Lambda double {mustBeNonnegative} = 1.0
                options.Gamma  double {mustBeNonnegative} = 0.0
            end
            obj.DataMatrix_ = dataMatrix;
            obj.Lambda      = options.Lambda;
            obj.Gamma       = options.Gamma;
        end

        function result = findBestSplit(obj, H)
            % FINDBESTSPLIT  Scan histogram to find the optimal split.
            %
            %   result = splitter.findBestSplit(H)
            %
            %   Input:
            %     H — histogram struct from HistogramBuilder.build()
            %         H.gradSum (NumBins × p), H.hessSum (NumBins × p)
            %
            %   Output:
            %     result — struct with fields:
            %       .FeatureIdx  double  — feature column (1-indexed)
            %       .BinIdx      double  — bin index (0-indexed, uint8 convention)
            %       .Threshold   double  — float split point (decoded from BinEdges)
            %       .Gain        double  — information gain of this split
            %       .IsValid     logical — false if no split beats Gamma threshold

            lam  = obj.Lambda;
            gam  = obj.Gamma;
            p    = obj.DataMatrix_.p;

            % Parent totals (sum over all bins)
            G_P = sum(H.gradSum, 1);   % 1×p
            H_P = sum(H.hessSum, 1);   % 1×p

            bestGain     = -inf;
            bestFeature  = -1;
            bestBin      = -1;

            % Scan all features and all bin boundaries
            for j = 1:p
                G_L = 0;
                H_L = 0;

                gj = H.gradSum(:, j);   % NumBins×1
                hj = H.hessSum(:, j);   % NumBins×1
                Gp = G_P(j);
                Hp = H_P(j);

                % Left-to-right cumulative scan (one pass, O(NumBins))
                for b = 1 : obj.DataMatrix_.NumBins - 1
                    G_L = G_L + gj(b);
                    H_L = H_L + hj(b);
                    G_R = Gp - G_L;
                    H_R = Hp - H_L;

                    % XGBoost gain formula
                    gain = 0.5 * ( G_L^2 / (H_L + lam) ...
                                 + G_R^2 / (H_R + lam) ...
                                 - Gp^2  / (Hp  + lam) ) - gam;

                    if gain > bestGain
                        bestGain    = gain;
                        bestFeature = j;
                        bestBin     = b - 1;   % convert to 0-indexed
                    end
                end
            end

            % Package result
            result.FeatureIdx = bestFeature;
            result.BinIdx     = bestBin;
            result.Gain       = bestGain;
            result.IsValid    = bestGain > 0;

            if result.IsValid
                result.Threshold = obj.DataMatrix_.decodeThreshold(bestFeature, bestBin);
            else
                result.Threshold = NaN;
            end
        end
    end
end
