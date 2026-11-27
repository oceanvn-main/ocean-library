classdef TopDownBlock < OceanDL.nn.Module
    % TOPDOWNBLOCK  One latent layer of a hierarchical VAE (Ladder VAE,
    %   Sonderby 2016; NVAE, Vahdat & Kautz 2020; VDVAE, Child 2021), acting on
    %   the top-down state h (H x W x Width x N) and, when encoding, the
    %   bottom-up feature e of the same resolution:
    %
    %     prior      p(z | h)    = PriorHead(PriorBlock(h))          (zero-init head:
    %                                                                  starts as N(0, I))
    %                or N(0, I) when 'Standard' (the top latent)
    %     posterior  q(z | h, e) = PostHead(PostBlock([h; e]))
    %                'Posterior' "residual" (NVAE): q = N(mu_p + dmu, logvar_p + dlv)
    %     z          ~ q (training / encoding) or ~ p (generation, 'Temperature')
    %     KL         = KL(q || p) per element (DiagonalGaussian.klTo)
    %     h          = Block(h + ZProj(z))
    %
    %   forward(h, e, mode, z, temperature, stream) with mode
    %     "sample"   z ~ q (training)        "mode"  z = mean of q (reconstruction)
    %     "given"    z supplied (decoding)   "prior" z ~ p (generation; e unused)
    %   returns [h, kl, z, q, p]; kl, q = [] for "prior" (and "given" with e = []).
    %   'LogVarClamp' c > 0 soft-clamps prior and posterior log-variances to
    %   c tanh(logvar / c) (NVAE); default 0 = off.

    properties (SetAccess = protected)
        PriorBlock = []
        PriorHead = []
        PostBlock
        PostHead
        ZProj
        Block
        Standard (1,1) logical
        Posterior (1,1) string
        LatentChannels (1,1) double
        LogVarClamp (1,1) double        % c > 0: log-variances -> c tanh(logvar / c); 0 = off
    end

    methods
        function obj = TopDownBlock(width, latentChannels, opts)
            arguments
                width (1,1) double {mustBePositive, mustBeInteger}
                latentChannels (1,1) double {mustBePositive, mustBeInteger}
                opts.Standard (1,1) logical = false
                opts.Posterior (1,1) string {mustBeMember(opts.Posterior, ["residual", "direct"])} = "residual"
                opts.BlockType (1,1) string {mustBeMember(opts.BlockType, ["res", "bottleneck"])} = "res"
                opts.Groups (1,1) double = 32
                opts.LogVarClamp (1,1) double {mustBeNonnegative} = 0
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            s = @(k) OceanDL.Vision.ConvEncoder.offset(opts.Seed, k);
            dt = {'DataType', opts.DataType};
            blk = @(ci, co, k) OceanDL.Generative.TopDownBlock.makeBlock(opts.BlockType, ci, co, opts.Groups, s(k), dt);
            obj.Standard = opts.Standard; obj.Posterior = opts.Posterior; obj.LatentChannels = latentChannels;
            obj.LogVarClamp = opts.LogVarClamp;
            if ~opts.Standard
                obj.PriorBlock = blk(width, width, 1);
                obj.PriorHead = OceanDL.nn.Conv2d(width, 2 * latentChannels, 3, 'Padding', 1, 'Seed', s(2), dt{:});
                obj.PriorHead.Weight.Data(:) = 0; obj.PriorHead.Bias.Data(:) = 0;
            end
            obj.PostBlock = blk(2 * width, width, 3);
            obj.PostHead = OceanDL.nn.Conv2d(width, 2 * latentChannels, 3, 'Padding', 1, 'Seed', s(4), dt{:});
            obj.ZProj = OceanDL.nn.Conv2d(latentChannels, width, 1, 'Seed', s(5), dt{:});
            obj.Block = blk(width, width, 6);
        end

        function [h, kl, z, q, p] = forward(obj, h, e, mode, z, temperature, stream)
            if nargin < 5, z = []; end
            if nargin < 6 || isempty(temperature), temperature = 1; end
            if nargin < 7 || isempty(stream), stream = RandStream.getGlobalStream(); end
            raw = OceanDL.Autograd.Tensor.raw(h);
            [H, W, ~, N] = size(raw, 1:4);
            if obj.Standard
                zero = zeros(H, W, obj.LatentChannels, N, 'like', raw);
                p = OceanDL.Generative.DiagonalGaussian(zero, zero);
            else
                p = OceanDL.Generative.DiagonalGaussian.fromChannels(obj.PriorHead.forward(obj.PriorBlock.forward(h)), "SSCB");
                p = OceanDL.Generative.DiagonalGaussian(p.Mu, obj.clampLogVar(p.LogVar));
            end
            kl = []; q = [];
            if mode == "prior"
                pm = OceanDL.Autograd.Tensor.raw(p.Mu); pl = OceanDL.Autograd.Tensor.raw(p.LogVar);
                z = pm + temperature * exp(0.5 * pl) .* cast(randn(stream, size(pm)), 'like', pm);
            elseif ~isempty(e)                                             % posterior needs the encoder feature
                d = OceanDL.Generative.DiagonalGaussian.fromChannels(obj.PostHead.forward(obj.PostBlock.forward(cat(3, h, e))), "SSCB");
                if obj.Posterior == "residual" && ~obj.Standard
                    q = OceanDL.Generative.DiagonalGaussian(p.Mu + d.Mu, obj.clampLogVar(p.LogVar + d.LogVar));
                else
                    q = OceanDL.Generative.DiagonalGaussian(d.Mu, obj.clampLogVar(d.LogVar));
                end
                if obj.Standard, kl = q.klStandardNormal(); else, kl = q.klTo(p); end
                switch mode
                    case "sample", z = q.rsample(stream);
                    case "mode",   z = q.Mu;
                end
            end
            h = obj.Block.forward(h + obj.ZProj.forward(z));
        end
    end

    methods (Access = private)
        function lv = clampLogVar(obj, lv)
            if obj.LogVarClamp > 0, lv = obj.LogVarClamp * tanh(lv / obj.LogVarClamp); end
        end
    end

    methods (Static)
        function b = makeBlock(kind, ci, co, groups, seed, dt)
            if kind == "bottleneck"
                b = OceanDL.Vision.BottleneckBlock(ci, co, 'Seed', seed, dt{:});
            else
                b = OceanDL.Vision.ConvResBlock(ci, co, 'Groups', groups, 'Seed', seed, dt{:});
            end
        end
    end
end
