classdef DataMatrix < handle
    % DATAMATRIX  System-layer data container for binned feature data.
    %
    %   OVERVIEW:
    %     A pure data-holding object produced by an IQuantizer strategy.
    %     Stores the compressed uint8 bin representation of a feature matrix
    %     and the BinEdges needed to decode bin indices back to float thresholds.
    %
    %     DataMatrix does NOT contain quantization logic — that belongs to
    %     an IQuantizer concrete strategy (e.g. UniformBinQuantizer).
    %
    %   COLUMN-MAJOR LAYOUT (explicit):
    %     BinnedX is stored N×p with MATLAB's default column-major order.
    %     getColumn(j) returns a contiguous N×1 uint8 vector — one cache-line
    %     read per chunk. This is the memory access pattern that powers LightGBM.
    %
    %   CREATION:
    %     Do not construct directly. Use an IQuantizer strategy:
    %
    %       q  = OceanEngine.Quantizers.UniformBinQuantizer('NumBins', 256);
    %       dm = q.quantize(X);
    %
    %   See also: OceanEngine.Interfaces.IQuantizer
    %             OceanEngine.Quantizers.UniformBinQuantizer
    %             OceanEngine.Interfaces.IHistogramBuilder

    properties (SetAccess = {?OceanEngine.Interfaces.IQuantizer})
        % BinnedX  N×p uint8 bin indices (0-indexed).
        % Column j is contiguous in memory → cache-friendly column scan.
        BinnedX     uint8

        % BinEdges  (NumBins+1)×p float cut-points per feature.
        % Edge b of feature j = BinEdges(b, j) — left boundary of bin b-1.
        BinEdges    double

        % NumBins  Number of bins per feature (≤256 for uint8 storage)
        NumBins     double

        % N  Number of samples
        N           double

        % p  Number of features
        p           double
    end

    methods
        function obj = DataMatrix(BinnedX, BinEdges, NumBins)
            % DATAMATRIX  Construct from pre-computed binned data.
            %
            %   Called by IQuantizer implementations — do not call directly.
            %
            %   Inputs:
            %     BinnedX  — N×p uint8 bin indices
            %     BinEdges — (NumBins+1)×p double cut-points
            %     NumBins  — scalar, number of bins
            arguments
                BinnedX  uint8  {mustBeNonempty}
                BinEdges double {mustBeNonempty}
                NumBins  double {mustBePositive, mustBeInteger}
            end
            obj.BinnedX  = BinnedX;
            obj.BinEdges = BinEdges;
            obj.NumBins  = NumBins;
            [obj.N, obj.p] = size(BinnedX);
        end

        function col = getColumn(obj, featureIdx)
            % GETCOLUMN  Return uint8 bin indices for one feature column.
            %
            %   col = dm.getColumn(j)   → N×1 uint8, contiguous in memory
            col = obj.BinnedX(:, featureIdx);
        end

        function threshold = decodeThreshold(obj, featureIdx, binIdx)
            % DECODETHRESHOLD  Convert a 0-indexed uint8 bin back to a float split point.
            %
            %   threshold = dm.decodeThreshold(j, b)
            %
            %   The split rule is: go left if X(:,j) <= threshold.
            b         = double(binIdx) + 1;
            b         = max(1, min(b, obj.NumBins));
            threshold = obj.BinEdges(b, featureIdx);
        end

        function printSummary(obj)
            % PRINTSUMMARY  Show compression statistics.
            doubleBytes = obj.N * obj.p * 8;
            uint8Bytes  = obj.N * obj.p * 1;
            fprintf('DataMatrix Summary\n');
            fprintf('  Samples (N):     %d\n',   obj.N);
            fprintf('  Features (p):    %d\n',   obj.p);
            fprintf('  Bins per feat:   %d\n',   obj.NumBins);
            fprintf('  Raw double RAM:  %.2f MB\n', doubleBytes / 1e6);
            fprintf('  Binned uint8:    %.2f MB\n', uint8Bytes  / 1e6);
            fprintf('  Compression:     %.1fx\n',   doubleBytes / uint8Bytes);
        end
    end
end
