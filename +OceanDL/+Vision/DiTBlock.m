classdef DiTBlock < OceanDL.nn.Module
    % DITBLOCK  Diffusion-Transformer block with adaLN-Zero conditioning
    %   (Peebles & Xie 2023) on tokens x (D x N x T) and a condition vector
    %   c (D x N) = time embedding [+ class embedding]:
    %
    %     [sh1 sc1 g1 sh2 sc2 g2] = Linear(SiLU(c))        (6D x N, ZERO-initialised)
    %     x = x + g1 .* Attn( Norm(x) .* (1 + sc1) + sh1 )
    %     x = x + g2 .* MLP ( Norm(x) .* (1 + sc2) + sh2 )
    %
    %   Norm has no parameters of its own (the condition supplies scale and
    %   shift). Zero init makes every block the identity at the start, so a
    %   deep stack trains stably -- the "Zero" in adaLN-Zero.
    %   Options: 'Norm' "layer" (DiT) | "rms"; 'FFN' "gelu" (DiT) | "swiglu";
    %   'QKNorm' "none" | "rms" (SD3 / FLUX); rotary positions via forward(..., rope).

    properties (SetAccess = protected)
        AdaLN
        Attention
        MLP
        NormKind (1,1) string
        Dim (1,1) double
    end

    methods
        function obj = DiTBlock(dim, heads, opts)
            arguments
                dim (1,1) double {mustBePositive, mustBeInteger}
                heads (1,1) double {mustBePositive, mustBeInteger}
                opts.MLPRatio (1,1) double = 4
                opts.Norm (1,1) string {mustBeMember(opts.Norm, ["layer", "rms"])} = "layer"
                opts.FFN (1,1) string {mustBeMember(opts.FFN, ["gelu", "swiglu"])} = "gelu"
                opts.QKNorm (1,1) string {mustBeMember(opts.QKNorm, ["none", "rms", "layer"])} = "none"
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            s = @(k) OceanDL.Vision.ConvEncoder.offset(opts.Seed, k);
            dt = {'DataType', opts.DataType};
            obj.Dim = dim; obj.NormKind = opts.Norm;
            obj.Attention = OceanDL.Transformer.MultiHeadAttention(dim, heads, 'QKNorm', opts.QKNorm, 'Seed', s(1), dt{:});
            obj.MLP = OceanDL.Transformer.FeedForward(dim, round(opts.MLPRatio * dim), 'Activation', opts.FFN, 'Seed', s(10), dt{:});
            obj.AdaLN = OceanDL.nn.Linear(dim, 6 * dim, 'Seed', s(20), dt{:});
            obj.AdaLN.Weight.Data(:) = 0; obj.AdaLN.Bias.Data(:) = 0;
        end

        function x = forward(obj, x, c, rope)
            if nargin < 4, rope = []; end
            m = obj.AdaLN.forward(OceanDL.Autograd.F.silu(c));
            [sh1, sc1, g1, sh2, sc2, g2] = OceanDL.Vision.DiTBlock.chunks(m, obj.Dim, 6);
            h = OceanDL.Vision.DiTBlock.modulate(OceanDL.Vision.DiTBlock.norm(x, obj.NormKind), sh1, sc1);
            x = x + g1 .* obj.Attention.forward(h, 'Rope', rope);
            h = OceanDL.Vision.DiTBlock.modulate(OceanDL.Vision.DiTBlock.norm(x, obj.NormKind), sh2, sc2);
            x = x + g2 .* obj.MLP.forward(h);
        end
    end

    methods (Static)
        function y = modulate(x, shift, scale)
            y = x .* (1 + scale) + shift;
        end

        function y = norm(x, kind)
            % Parameter-free LayerNorm / RMSNorm over features (eps 1e-6, as DiT).
            raw = OceanDL.Autograd.Tensor.raw(x);
            D = size(raw, 1);
            if kind == "rms"
                x = OceanDL.Autograd.Tensor.wrap(x);
                y = x ./ sqrt(mean(x .^ 2, 1) + 1e-6);
            else
                y = OceanDL.Autograd.F.layerNorm(x, ones(D, 1, 'like', raw), zeros(D, 1, 'like', raw), 1e-6);
            end
        end

        function varargout = chunks(m, D, k)
            % Split a (kD x N) modulation into k (D x N x 1) pieces (broadcast over tokens).
            N = size(m, 2);
            for i = 1:k
                varargout{i} = reshape(m((i - 1) * D + 1:i * D, :), D, N, 1); %#ok<AGROW>
            end
        end
    end
end
