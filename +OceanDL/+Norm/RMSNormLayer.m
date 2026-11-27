classdef RMSNormLayer < OceanDL.Norm.INormalization
    % RMSNORMLAYER  Root Mean Square Normalization (Zhang & Sennrich, 2019).
    %
    %   Simpler than LayerNorm: no mean subtraction, only RMS scaling.
    %   Cheaper to compute; used in LLaMA, Gemma, and other modern LLMs.
    %
    %   FORMULA (per sample n):
    %     rms_n = sqrt(mean(Z(:,n).^2) + eps)
    %     Z_out = gamma .* Z(:,n) / rms_n
    %
    %   PROPERTIES:
    %     - Mode-agnostic: train and infer behave identically
    %     - Learnable: gamma (d x 1), init 1. No beta.
    %
    %   Ref: Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer
    %        Normalization. arXiv:1910.07467
    %
    %   Usage:
    %     norm   = OceanDL.Norm.RMSNormLayer(64);
    %     Z_norm = norm.normalize(Z, 'train');

    properties
        Gamma   (:,1) double    % Scale parameter (d x 1), init 1
        Epsilon (1,1) double = 1e-8
    end

    methods
        function obj = RMSNormLayer(d, opts)
            % RMSNORMLAYER  Construct for a layer of width d.
            %
            %   norm = RMSNormLayer(64)
            arguments
                d            (1,1) double {mustBePositive}
                opts.Epsilon (1,1) double = 1e-8
            end
            obj.Gamma   = ones(d, 1);
            obj.Epsilon = opts.Epsilon;
        end

        function Z_out = normalize(obj, Z_in, mode) %#ok<INUSD>
            % mode is ignored — RMSNorm is stateless
            rms   = sqrt(mean(Z_in.^2, 1) + obj.Epsilon);   % (1 x N)
            Z_out = obj.Gamma .* (Z_in ./ rms);             % (d x N)
        end

        function p = parameters(obj)
            p.gamma = obj.Gamma;
            p.beta  = [];   % RMSNorm has no beta
        end

        function s = name(~)
            s = 'rmsnorm';
        end
    end
end
