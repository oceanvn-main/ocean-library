classdef (Abstract) IHistogramBuilder < handle
    % IHISTOGRAMBUILDER  Abstract interface for gradient histogram builders.
    %
    %   OVERVIEW:
    %     Defines the contract for building gradient/hessian histograms from
    %     a DataMatrix and a set of sample indices. The histogram is the core
    %     data structure in modern gradient boosted tree methods.
    %
    %   CONTRACT:
    %     H           = build(obj, dm, g, h, sampleIdx)
    %     H_right     = subtract(obj, H_parent, H_left)
    %
    %   HISTOGRAM FORMAT:
    %     Implementations must return a struct H with fields:
    %       H.gradSum — NumBins×p double (gradient sums per bin per feature)
    %       H.hessSum — NumBins×p double (hessian sums per bin per feature)
    %       H.count   — NumBins×p double (sample counts per bin per feature)
    %
    %   KNOWN IMPLEMENTATIONS:
    %     DenseHistogramBuilder  — CPU, dense column-major accumulation. O(N*p).
    %     (Planned) SparseHistogramBuilder — skips zero-gradient samples.
    %     (Planned) ParallelHistogramBuilder — splits work across features.
    %
    %   EXTENSION:
    %     1. Subclass IHistogramBuilder in +OceanEngine/+Histogram/
    %     2. Implement build() and subtract()
    %     3. Inject into tree builder via constructor
    %
    %   See also: OceanEngine.Histogram.DenseHistogramBuilder
    %             OceanEngine.Interfaces.ISplitFinder
    %             OceanEngine.Data.DataMatrix

    methods (Abstract)
        % BUILD  Accumulate gradient/hessian histograms for one tree node.
        %
        %   H = build(obj, dm, g, h, sampleIdx)
        %
        %   Inputs:
        %     dm        — OceanEngine.Data.DataMatrix
        %     g         — N×1 gradient vector (from ILoss.gradient)
        %     h         — N×1 hessian vector  (from ILoss.hessian)
        %     sampleIdx — M×1 indices of samples in this node
        %
        %   Output:
        %     H — struct with gradSum, hessSum, count (each NumBins×p)
        H = build(obj, dm, g, h, sampleIdx)

        % SUBTRACT  Compute right-child histogram via subtraction trick.
        %
        %   H_right = subtract(obj, H_parent, H_left)
        %
        %   Right child = Parent - Left child (element-wise). O(bins) only.
        %   Avoids building the larger child's histogram from scratch.
        H_right = subtract(obj, H_parent, H_left)

        % NAME  Human-readable label for this builder.
        n = name(obj)
    end
end
