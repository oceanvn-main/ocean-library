classdef HistogramBuilder < OceanEngine.Interfaces.IHistogramBuilder
    % HISTOGRAMBUILDER  Builds per-feature gradient/hessian histograms in O(N).
    %
    %   OVERVIEW:
    %     At each tree node, instead of sorting (O(N log N)), we iterate once
    %     over the node's samples and accumulate gradient and hessian sums into
    %     a fixed-size histogram (one slot per bin per feature).
    %
    %     Time complexity: O(N * p) per node — strictly linear in samples.
    %     Space: O(NumBins * p) — for 256 bins and typical p, fits in L1 cache.
    %
    %   HISTOGRAM SUBTRACTION TRICK (LightGBM optimization):
    %     When splitting a node into left and right children:
    %       H_parent is already built.
    %       Build H_left for the SMALLER child only (fewer samples → less work).
    %       H_right = H_parent - H_left   (element-wise, O(bins) only)
    %     This halves the work for every split.
    %
    %   USAGE:
    %     builder = OceanEngine.Histogram.HistogramBuilder(dataMatrix);
    %
    %     % Build histogram for root node (all samples)
    %     H = builder.build(g, h, 1:N);
    %
    %     % Build for a child node (subtraction trick)
    %     H_left  = builder.build(g, h, leftIdx);
    %     H_right = builder.subtract(H_parent, H_left);
    %
    %   OUTPUT FORMAT (H):
    %     H is a struct array of size (NumBins × p):
    %       H(b, j).gradSum  — sum of gradients for samples in bin b of feature j
    %       H(b, j).hessSum  — sum of hessians for samples in bin b of feature j
    %       H(b, j).count    — number of samples in bin b of feature j
    %
    %   See also: OceanEngine.Data.DataMatrix
    %             OceanEngine.Interfaces.IHistogramBuilder
    %             OceanEngine.SplitFinders.HistogramSplitFinder

    properties (SetAccess = private)
        DataMatrix_    % OceanEngine.Data.DataMatrix — the compressed data
        NumBins_  double
        p_        double
    end

    methods
        function obj = HistogramBuilder(dataMatrix)
            % HISTOGRAMBUILDER  Constructor.
            %
            %   builder = HistogramBuilder(dataMatrix)
            %
            %   Input:
            %     dataMatrix — OceanEngine.Data.DataMatrix instance
            arguments
                dataMatrix OceanEngine.Data.DataMatrix
            end
            obj.DataMatrix_ = dataMatrix;
            obj.NumBins_    = dataMatrix.NumBins;
            obj.p_          = dataMatrix.p;
        end

        function H = build(obj, g, h, sampleIdx)
            % BUILD  Accumulate gradient/hessian histograms for one node.
            %
            %   H = builder.build(g, h, sampleIdx)
            %
            %   Inputs:
            %     g         — N×1 gradient vector (from ILoss.gradient)
            %     h         — N×1 hessian vector  (from ILoss.hessian)
            %     sampleIdx — M×1 indices of samples belonging to this node
            %
            %   Output:
            %     H — (NumBins × p) struct with fields: gradSum, hessSum, count
            %
            %   Complexity: O(M * p) — one pass over node samples, p features.

            B = obj.NumBins_;
            p = obj.p_;

            % Pre-allocate flat arrays (faster than struct array init)
            gradSum = zeros(B, p);
            hessSum = zeros(B, p);
            count   = zeros(B, p);

            % O(M*p) accumulation pass
            % For each sample, look up its uint8 bin index per feature,
            % then add gradient/hessian into that bin slot.
            for i = sampleIdx(:)'
                gi = g(i);
                hi = h(i);
                for j = 1:p
                    b = double(obj.DataMatrix_.BinnedX(i, j)) + 1; % 1-indexed
                    gradSum(b, j) = gradSum(b, j) + gi;
                    hessSum(b, j) = hessSum(b, j) + hi;
                    count(b, j)   = count(b, j)   + 1;
                end
            end

            % Pack into output struct
            H.gradSum = gradSum;   % NumBins × p
            H.hessSum = hessSum;   % NumBins × p
            H.count   = count;     % NumBins × p
        end

        function H_right = subtract(~, H_parent, H_left)
            % SUBTRACT  Compute right-child histogram via subtraction trick.
            %
            %   H_right = builder.subtract(H_parent, H_left)
            %
            %   H_right = H_parent - H_left  (element-wise per bin per feature)
            %   Complexity: O(NumBins * p) — independent of N.
            H_right.gradSum = H_parent.gradSum - H_left.gradSum;
            H_right.hessSum = H_parent.hessSum - H_left.hessSum;
            H_right.count   = H_parent.count   - H_left.count;
        end

        function n = name(~)
            n = "Dense Histogram Builder";
        end
    end
end
