classdef UniformBinQuantizer < OceanEngine.Interfaces.IQuantizer
    % UNIFORMBINQUANTIZER  Equal-width bin quantization strategy.
    %
    %   OVERVIEW:
    %     The simplest and fastest quantization strategy. Divides each feature's
    %     range [min, max] into NumBins equal-width intervals and maps each
    %     value to its bin index (uint8, 0-indexed).
    %
    %   WHEN TO USE:
    %     Good default for regression and classification data with roughly
    %     uniform feature distributions. Fast and simple.
    %
    %   LIMITATION:
    %     If data is skewed (many values near one end), most of the useful
    %     splits will land in just a few bins, wasting resolution. In that
    %     case, prefer QuantileBinQuantizer (equal-frequency bins, planned).
    %
    %   USAGE:
    %     q  = OceanEngine.Quantizers.UniformBinQuantizer('NumBins', 256);
    %     dm = q.quantize(X);   % dm is an OceanEngine.Data.DataMatrix
    %
    %   See also: OceanEngine.Interfaces.IQuantizer
    %             OceanEngine.Data.DataMatrix

    properties
        % NumBins  Number of equal-width bins per feature (default: 256)
        NumBins double = 256
    end

    methods
        function obj = UniformBinQuantizer(options)
            % UNIFORMBINQUANTIZER  Constructor.
            %
            %   q = UniformBinQuantizer()
            %   q = UniformBinQuantizer('NumBins', 128)
            arguments
                options.NumBins double {mustBePositive, mustBeInteger, ...
                    mustBeLessThanOrEqual(options.NumBins, 256)} = 256
            end
            obj.NumBins = options.NumBins;
        end

        function dm = quantize(obj, X)
            % QUANTIZE  Convert raw double feature matrix to a DataMatrix.
            %
            %   dm = q.quantize(X)
            %
            %   Input:
            %     X — N×p double feature matrix
            %
            %   Output:
            %     dm — OceanEngine.Data.DataMatrix with uniform bins
            %
            %   Complexity: O(N*p) — one pass per column.
            arguments
                obj
                X double {mustBeNumeric, mustBeNonempty}
            end

            [N, p]   = size(X);
            B        = obj.NumBins;

            BinnedX  = zeros(N, p, 'uint8');
            BinEdges = zeros(B + 1, p);

            for j = 1:p
                col = X(:, j);

                % NumBins+1 equal-spaced edges covering [min, max]
                edges      = linspace(min(col), max(col), B + 1);
                edges(1)   = edges(1)   - eps(edges(1));     % capture left boundary
                edges(end) = edges(end) + eps(edges(end));   % capture right boundary

                binIdx = discretize(col, edges);   % 1-indexed
                binIdx(isnan(binIdx)) = 1;         % guard NaN → bin 1

                BinnedX(:, j)  = uint8(binIdx - 1);    % 0-indexed uint8
                BinEdges(:, j) = edges; % Store the actual mapping to float thresholds
            end

            dm = OceanEngine.Data.DataMatrix(BinnedX, BinEdges, B);
        end

        function n = name(obj)
            n = sprintf("Uniform (%d bins)", obj.NumBins);
        end
    end
end
