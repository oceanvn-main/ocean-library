classdef (Abstract) IPreprocessor < handle
    % IPREPROCESSOR  Abstract interface for data-side input encoders.
    %
    %   Encoders transform raw input coordinates BEFORE they enter the
    %   neural network training pipeline. Parameters are frozen after fit.
    %
    %   CONVENTION:
    %     X_raw — (d_in x N) raw input (column-major, N samples)
    %     X_enc — (d_out x N) encoded output
    %
    %   CONTRACT:
    %     encode(X_raw) → X_enc    apply the encoding
    %     d = outputDim()          output feature dimension after encoding
    %     s = name()               human-readable string identifier
    %
    %   Implementations:
    %     FourierFeatureMap   — random Fourier features (Tancik et al., 2020)
    %     PolynomialMapper    — polynomial power expansion
    %
    %   See also: OceanStats.Preprocessing.FourierFeatureMap
    %             OceanStats.Preprocessing.PolynomialMapper

    methods (Abstract)
        X_enc = encode(obj, X_raw)
        % ENCODE  Apply the encoding to X_raw (d_in x N) → X_enc (d_out x N).

        d = outputDim(obj)
        % OUTPUTDIM  Return the encoded feature dimension (scalar).

        s = name(obj)
        % NAME  Return a string identifier, e.g. 'fourier_features'.
    end
end
