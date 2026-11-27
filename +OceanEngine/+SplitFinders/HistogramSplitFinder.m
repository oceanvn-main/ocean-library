classdef HistogramSplitFinder < OceanEngine.Interfaces.ISplitFinder
    % HISTOGRAMSPLITFINDER  Finds the optimal split by scanning a gradient histogram.
    %
    %   OVERVIEW:
    %     Concrete ISplitFinder strategy that performs a left-to-right cumulative
    %     scan over all bins for all features to find the (feature, threshold) split
    %     that maximizes the XGBoost gain formula.
    %
    %     Time complexity: O(NumBins * p) per node — strictly independent of N.
    %     For 256 bins, the entire histogram fits in the CPU L1 cache.
    %
    %   GAIN FORMULA (XGBoost, Eq. 7):
    %     Gain = 0.5 * [G_L^2/(H_L + λ) + G_R^2/(H_R + λ) - G_P^2/(H_P + λ)] - γ
    %
    %     Lambda (λ) — L2 regularization on leaf weights
    %     Gamma  (γ) — Minimum gain to accept a split (natural pruning)
    %
    %   USAGE:
    %     finder = OceanEngine.SplitFinders.HistogramSplitFinder('Lambda', 1, 'Gamma', 0);
    %     result = finder.findBestSplit(H, dm);
    %
    %   See also: OceanEngine.Interfaces.ISplitFinder
    %             OceanEngine.Interfaces.IHistogramBuilder
    %             OceanEngine.Data.DataMatrix

    properties
        % Lambda  L2 regularization on leaf weights (default: 1)
        Lambda double = 1.0

        % Gamma  Minimum gain required to accept a split (default: 0)
        Gamma  double = 0.0
    end

    methods
        function obj = HistogramSplitFinder(options)
            % HISTOGRAMSPLITFINDER  Constructor.
            %
            %   finder = HistogramSplitFinder()
            %   finder = HistogramSplitFinder('Lambda', 1.0, 'Gamma', 0.1)
            arguments
                options.Lambda double {mustBeNonnegative} = 1.0
                options.Gamma  double {mustBeNonnegative} = 0.0
            end
            obj.Lambda = options.Lambda;
            obj.Gamma  = options.Gamma;
        end

        function result = findBestSplit(obj, H, dm)
            % FINDBESTSPLIT  Scan histogram to find the optimal split.
            %
            %   result = finder.findBestSplit(H, dm)
            %
            %   Inputs:
            %     H  — histogram struct (gradSum, hessSum — NumBins×p)
            %     dm — OceanEngine.Data.DataMatrix (for threshold decoding)
            %
            %   Output:
            %     result — struct: FeatureIdx, BinIdx, Threshold, Gain, IsValid

            lam = obj.Lambda;
            gam = obj.Gamma;
            p   = dm.p;

            G_P = sum(H.gradSum, 1);   % 1×p parent gradient totals
            H_P = sum(H.hessSum, 1);   % 1×p parent hessian totals

            bestGain    = -inf;
            bestFeature = -1;
            bestBin     = -1;

            for j = 1:p
                G_L = 0; H_L = 0;
                gj  = H.gradSum(:, j);
                hj  = H.hessSum(:, j);
                Gp  = G_P(j);
                Hp  = H_P(j);

                % Left-to-right cumulative scan — O(NumBins)
                for b = 1 : dm.NumBins - 1
                    G_L  = G_L + gj(b);
                    H_L  = H_L + hj(b);
                    G_R  = Gp - G_L;
                    H_R  = Hp - H_L;
                    gain = 0.5 * ( G_L^2 / (H_L + lam) ...
                                 + G_R^2 / (H_R + lam) ...
                                 - Gp^2  / (Hp  + lam) ) - gam;
                    if gain > bestGain
                        bestGain    = gain;
                        bestFeature = j;
                        bestBin     = b - 1;   % 0-indexed
                    end
                end
            end

            result.FeatureIdx = bestFeature;
            result.BinIdx     = bestBin;
            result.Gain       = bestGain;
            result.IsValid    = bestGain > 0;
            if result.IsValid
                result.Threshold = dm.decodeThreshold(bestFeature, bestBin);
            else
                result.Threshold = NaN;
            end
        end

        function n = name(obj)
            n = sprintf("Histogram Split Finder (λ=%.2f, γ=%.2f)", obj.Lambda, obj.Gamma);
        end
    end
end
