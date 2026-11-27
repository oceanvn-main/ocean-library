classdef LayerNormLayer < OceanDL.Norm.INormalization
    % LAYERNORMLAYER  Layer Normalization (Ba et al., 2016).
    %
    %   Normalizes across the FEATURE dimension for each sample independently.
    %   Each column of Z (one sample) is zero-meaned and unit-variance scaled,
    %   then rescaled by learnable gamma and shifted by beta.
    %
    %   FORMULA (per sample n):
    %     mu_n    = mean(Z(:, n))               scalar
    %     sigma_n = std(Z(:, n), 1)  + eps      scalar  (BIASED/population variance,
    %                                                     divisor N — see note below)
    %     Z_hat   = (Z(:,n) - mu_n) / sigma_n  (d x 1)
    %     Z_out   = gamma .* Z_hat + beta       (d x 1)
    %
    %   VARIANCE ESTIMATOR — must be the BIASED (population, divide-by-N)
    %   estimator, NOT MATLAB's std() default (unbiased, divide-by-(N-1)).
    %   This is what Ba et al. 2016 defines and what every mainstream
    %   framework's LayerNorm (PyTorch nn.LayerNorm, TensorFlow) actually
    %   implements. Confirmed empirically (2026-09-18) against a real
    %   prajjwal1/bert-tiny checkpoint: this class previously used the
    %   unbiased estimator (MATLAB's default), which reproduced BERT's
    %   embedding-layer output to only ~0.3-0.4% relative accuracy — small
    %   enough to look "almost right" (cosine similarity ~0.99999 after 2
    %   full transformer blocks) but NOT bit-correct, and the gap compounds
    %   with depth. Switching to std(Z,1,1) matched the real model's output
    %   to full displayed double precision. See
    %   Planning/System/Retrieval/Retrieval_System_Master_Plan.md §7 for the
    %   full diagnostic trail — this was found via external validation
    %   against real trained weights, not a synthetic-weight unit test,
    %   because no synthetic ground truth exists to catch a wrong variance
    %   divisor against.
    %
    %   PROPERTIES:
    %     - Mode-agnostic: train and infer behave identically (no running stats)
    %     - Learnable: gamma (d x 1) init 1, beta (d x 1) init 0
    %
    %   Ref: Ba, J., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization.
    %        arXiv:1607.06450
    %
    %   Usage:
    %     norm   = OceanDL.Norm.LayerNormLayer(64);
    %     Z_norm = norm.normalize(Z, 'train');   % or 'infer' — same behaviour

    properties
        Gamma   (:,1) double    % Scale parameter (d x 1), init 1
        Beta    (:,1) double    % Shift parameter (d x 1), init 0
        Epsilon (1,1) double = 1e-5
    end

    methods
        function obj = LayerNormLayer(d, opts)
            % LAYERNORMLAYER  Construct for a layer of width d.
            %
            %   norm = LayerNormLayer(64)
            %   norm = LayerNormLayer(64, 'Epsilon', 1e-8)
            arguments
                d           (1,1) double {mustBePositive}
                opts.Epsilon (1,1) double = 1e-5
            end
            obj.Gamma   = ones(d, 1);
            obj.Beta    = zeros(d, 1);
            obj.Epsilon = opts.Epsilon;
        end

        function Z_out = normalize(obj, Z_in, mode) %#ok<INUSD>
            % mode is ignored — LayerNorm is stateless
            mu    = mean(Z_in, 1);                       % (1 x N)
            sig   = std(Z_in, 1, 1) + obj.Epsilon;       % (1 x N) — biased/population variance (see class note)
            Z_hat = (Z_in - mu) ./ sig;                  % (d x N)
            Z_out = obj.Gamma .* Z_hat + obj.Beta;       % (d x N) broadcast
        end

        function p = parameters(obj)
            p.gamma = obj.Gamma;
            p.beta  = obj.Beta;
        end

        function s = name(~)
            s = 'layernorm';
        end
    end
end
