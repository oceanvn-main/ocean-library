classdef (Abstract) INormalization < handle
    % INORMALIZATION  Abstract interface for normalization layers.
    %
    %   Normalization is applied AFTER the affine transform (Z = W*A + b)
    %   and BEFORE the activation function inside VectorizedEngine.
    %
    %   CONVENTION:
    %     Z_in  — (d x N) pre-activation from the affine step
    %     Z_out — (d x N) normalized output, same shape
    %     mode  — 'train' or 'infer'. Affects running-stat behaviour
    %             in BatchNorm (LayerNorm/RMSNorm are mode-agnostic).
    %
    %   LEARNABLE PARAMETERS:
    %     gamma — scale (d x 1), initialized to 1
    %     beta  — shift (d x 1), initialized to 0 (LayerNorm/BatchNorm only)
    %     parameters() returns a struct with these fields.
    %
    %   USAGE:
    %     norm   = OceanDL.Norm.LayerNormLayer(d);
    %     Z_norm = norm.normalize(Z, 'train');
    %
    %   See also: OceanDL.Norm.BatchNormLayer
    %             OceanDL.Norm.LayerNormLayer
    %             OceanDL.Norm.RMSNormLayer

    methods (Abstract)
        Z_out = normalize(obj, Z_in, mode)
        % NORMALIZE  Apply normalization to Z_in (d x N).
        %   mode — 'train' or 'infer'

        p = parameters(obj)
        % PARAMETERS  Return struct of learnable params (gamma, beta).

        s = name(obj)
        % NAME  Return string identifier, e.g. 'layernorm'.
    end
end
