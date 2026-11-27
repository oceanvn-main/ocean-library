classdef (Abstract) IQuantizer < handle
    % IQUANTIZER  Abstract interface for feature matrix quantization strategies.
    %
    %   OVERVIEW:
    %     Defines the contract for converting a raw double feature matrix X
    %     into a compressed DataMatrix (uint8 bins). Different concrete strategies
    %     implement different binning policies.
    %
    %   KNOWN IMPLEMENTATIONS:
    %     UniformBinQuantizer  — Equal-width bins [min, max] / NumBins.
    %                            Simple, fast. Good default.
    %     QuantileBinQuantizer — Equal-frequency bins (same N per bin).
    %                            Better statistical coverage. LightGBM default.
    %                            (Planned)
    %
    %   EXTENSION:
    %     1. Subclass IQuantizer in +OceanEngine/+Quantizers/
    %     2. Implement quantize(X) → DataMatrix
    %     3. Inject into any pipeline that accepts an IQuantizer
    %
    %   See also: OceanEngine.Quantizers.UniformBinQuantizer
    %             OceanEngine.Data.DataMatrix

    methods (Abstract)
        % QUANTIZE  Convert raw feature matrix into a binned DataMatrix.
        %
        %   dm = quantize(obj, X)
        %
        %   Input:
        %     X  — N×p double feature matrix (rows = samples, cols = features)
        %
        %   Output:
        %     dm — OceanEngine.Data.DataMatrix (uint8 bins + BinEdges)
        dm = quantize(obj, X)

        % NAME  Human-readable label for this quantization strategy.
        %
        %   Output:
        %     n — string (e.g. "Uniform (256 bins)")
        n = name(obj)
    end
end
