classdef MMDiTBlock < OceanDL.nn.Module
    % MMDITBLOCK  Multimodal DiT block (SD3 / FLUX double-stream): image
    %   tokens x (D x N x Ti) and context tokens z (D x N x Tc) each keep their
    %   own adaLN-Zero modulation, norms and MLP, and meet in JointAttention:
    %
    %     per stream s:  [sh1 sc1 g1 sh2 sc2 g2]_s = Linear_s(SiLU(c))   (zero init)
    %     [a_x, a_z] = JointAttention( mod(Norm(x)), mod(Norm(z)) )
    %     x = x + g1_x .* a_x;          z = z + g1_z .* a_z
    %     x = x + g2_x .* MLP_x(mod(Norm(x)));   z = z + g2_z .* MLP_z(mod(Norm(z)))

    properties (SetAccess = protected)
        AdaX
        AdaZ
        Attention
        MLPX
        MLPZ
        NormKind (1,1) string
        Dim (1,1) double
    end

    methods
        function obj = MMDiTBlock(dim, heads, opts)
            arguments
                dim (1,1) double
                heads (1,1) double
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
            obj.Attention = OceanDL.Transformer.JointAttention(dim, heads, 'QKNorm', opts.QKNorm, 'Seed', s(1), dt{:});
            hid = round(opts.MLPRatio * dim);
            obj.MLPX = OceanDL.Transformer.FeedForward(dim, hid, 'Activation', opts.FFN, 'Seed', s(20), dt{:});
            obj.MLPZ = OceanDL.Transformer.FeedForward(dim, hid, 'Activation', opts.FFN, 'Seed', s(30), dt{:});
            obj.AdaX = OceanDL.nn.Linear(dim, 6 * dim, 'Seed', s(40), dt{:});
            obj.AdaZ = OceanDL.nn.Linear(dim, 6 * dim, 'Seed', s(50), dt{:});
            for a = {obj.AdaX, obj.AdaZ}
                a{1}.Weight.Data(:) = 0; a{1}.Bias.Data(:) = 0;
            end
        end

        function [x, z] = forward(obj, x, z, c, rope)
            if nargin < 5, rope = []; end
            chunks = @(m) OceanDL.Vision.DiTBlock.chunks(m, obj.Dim, 6);
            md = @(v, sh, sc) OceanDL.Vision.DiTBlock.modulate(OceanDL.Vision.DiTBlock.norm(v, obj.NormKind), sh, sc);
            sc = OceanDL.Autograd.F.silu(c);
            [xs1, xc1, xg1, xs2, xc2, xg2] = chunks(obj.AdaX.forward(sc));
            [zs1, zc1, zg1, zs2, zc2, zg2] = chunks(obj.AdaZ.forward(sc));
            [ax, az] = obj.Attention.forward(md(x, xs1, xc1), md(z, zs1, zc1), rope);
            x = x + xg1 .* ax;
            z = z + zg1 .* az;
            x = x + xg2 .* obj.MLPX.forward(md(x, xs2, xc2));
            z = z + zg2 .* obj.MLPZ.forward(md(z, zs2, zc2));
        end
    end
end
