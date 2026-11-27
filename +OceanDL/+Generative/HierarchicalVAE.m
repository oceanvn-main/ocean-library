classdef HierarchicalVAE < OceanDL.nn.Module & OceanDL.Interpret.ITraceable
    % HIERARCHICALVAE  Configurable hierarchical VAE (Ladder VAE, Sonderby
    %   2016; NVAE, Vahdat & Kautz 2020; VDVAE, Child 2021) with ANY number of
    %   resolution levels and ANY number of latent layers per level.
    %
    %   CONFIGURATION (one entry per resolution level, level 1 = full size,
    %   each next level halves H and W):
    %     'Widths'          [32 64 64]  channel width of each level
    %     'LatentsPerLevel' [0 1 1]     latent layers at each level (>= 0, total >= 1)
    %     'LatentChannels'  4           scalar or one per level
    %     'BlocksPerLevel'  1           bottom-up and top-down blocks per level
    %     'BlockType'       "res" (ConvResBlock) | "bottleneck" (VDVAE, ~10x cheaper)
    %     'Posterior'       "residual" (NVAE: q = prior + offset) | "direct"
    %     'TopPrior'        "standard" N(0, I) for the top latent | "learned"
    %   e.g. [32 64] + [0 1] is a plain VAE; [32 64 64] + [0 1 1] a 2-layer
    %   HVAE; [32 64 64] + [0 4 2] + "bottleneck" a 6-layer VDVAE-style model.
    %
    %   GENERATIVE MODEL (top-down):  p(z_1) prod_l p(z_l | z_<l)  p(x | z)
    %   latent layers ordered TOP (lowest resolution) FIRST. The top-down state
    %   starts from a learned per-channel constant at the lowest level; every
    %   TopDownBlock adds its latent; between levels: nearest x2 + conv.
    %   INFERENCE: one bottom-up pass stores a feature per level; each
    %   TopDownBlock's posterior reads [top-down state; that feature].
    %
    %   LOSS = negative ELBO per sample (nats):
    %     L = -log p(x | z) + Beta * sum_l sum_g max(KL_lg, FreeBits)
    %     KL_l = KL( q(z_l | x, z_<l) || p(z_l | z_<l) )   (closed form, per layer)
    %     'FreeBitsPer' g = "layer" (whole layer) | "channel" | "unit" (each latent
    %     number; the only granularity that stops individual units collapsing).
    %   'LogVarClamp' c > 0 soft-clamps every prior / posterior log-variance to
    %     c tanh(logvar / c) (NVAE-style; guards exp() against overflow). 0 = off.
    %   Levels above the top latent layer are not built (they carry no information).
    %   LastStats: Recon, KL, KLPerLayer (top first), NegELBO, BitsPerDim.

    properties
        Beta (1,1) double {mustBeNonnegative} = 1
    end

    properties (SetAccess = protected)
        ConvIn
        Up cell = {}            % bottom-up: Up{l} = cell of blocks at level l
        Down cell = {}          % Down{l} = stride-2 conv from level l to l + 1 ([] at the last level)
        H0                      % learned top-down start (1 x 1 x Width_L)
        Lift cell = {}          % Lift{l} = conv after upsampling from level l + 1 to l
        TD cell = {}            % TD{l} = cell of TopDownBlocks at level l (top of the level first)
        Res cell = {}           % Res{l} = cell of extra top-down blocks at level l
        OutNorm
        ConvOut
        Widths (1,:) double
        LatentsPerLevel (1,:) double
        LatentChannels (1,:) double
        Likelihood (1,1) string
        LogSigma
        FreeBits (1,1) double
        FreeBitsPer (1,1) string
        OutChannels (1,1) double
    end

    properties (SetAccess = private)
        Stream
        LastStats = struct()
    end

    methods
        function obj = HierarchicalVAE(opts)
            arguments
                opts.InChannels (1,1) double = 1
                opts.OutChannels (1,1) double = 0
                opts.Widths (1,:) double {mustBePositive, mustBeInteger} = [32 64 64]
                opts.LatentsPerLevel (1,:) double {mustBeNonnegative, mustBeInteger} = [0 1 1]
                opts.LatentChannels (1,:) double {mustBePositive, mustBeInteger} = 4
                opts.BlocksPerLevel (1,1) double {mustBeNonnegative, mustBeInteger} = 1
                opts.BlockType (1,1) string {mustBeMember(opts.BlockType, ["res", "bottleneck"])} = "res"
                opts.Posterior (1,1) string {mustBeMember(opts.Posterior, ["residual", "direct"])} = "residual"
                opts.TopPrior (1,1) string {mustBeMember(opts.TopPrior, ["standard", "learned"])} = "standard"
                opts.Likelihood (1,1) string {mustBeMember(opts.Likelihood, ["gaussian", "bernoulli"])} = "bernoulli"
                opts.Sigma (1,1) double {mustBePositive} = 1
                opts.LearnSigma (1,1) logical = false
                opts.Beta (1,1) double {mustBeNonnegative} = 1
                opts.FreeBits (1,1) double {mustBeNonnegative} = 0
                opts.FreeBitsPer (1,1) string {mustBeMember(opts.FreeBitsPer, ["layer", "channel", "unit"])} = "layer"
                opts.LogVarClamp (1,1) double {mustBeNonnegative} = 10
                opts.Groups (1,1) double = 32
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            L = numel(opts.Widths);
            if numel(opts.LatentsPerLevel) ~= L
                error('OceanDL:Generative:HierarchicalVAE:Config', ...
                    'LatentsPerLevel needs one entry per level (%d), got %d.', L, numel(opts.LatentsPerLevel));
            end
            if sum(opts.LatentsPerLevel) == 0
                error('OceanDL:Generative:HierarchicalVAE:Config', 'At least one latent layer is required.');
            end
            zc = opts.LatentChannels;
            if isscalar(zc), zc = repmat(zc, 1, L); end
            if numel(zc) ~= L
                error('OceanDL:Generative:HierarchicalVAE:Config', 'LatentChannels must be a scalar or one per level.');
            end
            % Levels ABOVE the top latent layer carry no information: nothing reads
            % their encoder features and their decoder part only transforms the
            % learned constant H0. They are not built (e.g. [32 64 64] + [0 1 0]
            % is the same model as [32 64] + [0 1]).
            L = find(opts.LatentsPerLevel > 0, 1, 'last');
            W = opts.Widths(1:L); zc = zc(1:L);
            obj.Widths = W; obj.LatentsPerLevel = opts.LatentsPerLevel(1:L); obj.LatentChannels = zc;
            obj.Likelihood = opts.Likelihood; obj.FreeBits = opts.FreeBits; obj.Beta = opts.Beta;
            obj.FreeBitsPer = opts.FreeBitsPer;
            obj.OutChannels = opts.OutChannels; if obj.OutChannels == 0, obj.OutChannels = opts.InChannels; end
            if opts.LearnSigma, obj.LogSigma = OceanDL.nn.Parameter(log(opts.Sigma)); else, obj.LogSigma = log(opts.Sigma); end
            s = @(k) OceanDL.Vision.ConvEncoder.offset(opts.Seed, k);
            dt = {'DataType', opts.DataType};
            blk = @(ci, co, k) OceanDL.Generative.TopDownBlock.makeBlock(opts.BlockType, ci, co, opts.Groups, s(k), dt);

            % Bottom-up encoder.
            obj.ConvIn = OceanDL.nn.Conv2d(opts.InChannels, W(1), 3, 'Padding', 1, 'Seed', s(1), dt{:});
            k = 100;
            for l = 1:L
                obj.Up{l} = {};                                          % input already has W(l) channels
                for b = 1:opts.BlocksPerLevel
                    k = k + 10; obj.Up{l}{b} = blk(W(l), W(l), k);
                end
                if l < L
                    k = k + 10;
                    obj.Down{l} = OceanDL.nn.Conv2d(W(l), W(l + 1), 3, 'Stride', 2, 'Padding', 1, 'Seed', s(k), dt{:});
                else
                    obj.Down{l} = [];
                end
            end

            % Top-down decoder / generative path.
            obj.H0 = OceanDL.nn.Parameter(zeros(1, 1, W(L), opts.DataType));
            first = true; k = 1000;
            for l = L:-1:1
                obj.TD{l} = {};
                for j = 1:opts.LatentsPerLevel(l)
                    k = k + 100;
                    obj.TD{l}{j} = OceanDL.Generative.TopDownBlock(W(l), zc(l), 'Standard', first && opts.TopPrior == "standard", ...
                        'Posterior', opts.Posterior, 'BlockType', opts.BlockType, 'Groups', opts.Groups, ...
                        'LogVarClamp', opts.LogVarClamp, 'Seed', s(k), dt{:});
                    first = false;
                end
                obj.Res{l} = {};
                for b = 1:opts.BlocksPerLevel
                    k = k + 10; obj.Res{l}{b} = blk(W(l), W(l), k);
                end
                if l > 1
                    k = k + 10;
                    obj.Lift{l - 1} = OceanDL.nn.Conv2d(W(l), W(l - 1), 3, 'Padding', 1, 'Seed', s(k), dt{:});
                end
            end
            obj.OutNorm = OceanDL.nn.GroupNorm(OceanDL.nn.GroupNorm.groupsFor(W(1), opts.Groups), W(1), dt{:});
            obj.ConvOut = OceanDL.nn.Conv2d(W(1), obj.OutChannels, 3, 'Padding', 1, 'Seed', s(9000), dt{:});
            if isempty(opts.Seed), obj.Stream = RandStream.getGlobalStream();
            else, obj.Stream = RandStream('mt19937ar', 'Seed', s(9001)); end
        end

        function n = numLatentLayers(obj)
            n = sum(obj.LatentsPerLevel);
        end

        function L = forward(obj, X)
            N = size(X, 4);
            feats = obj.bottomUp(X);
            [out, kls] = obj.topDown(feats, N, size(X, 1:2), "sample");
            nll = OceanDL.Generative.VAE.nll(out, X, obj.Likelihood, obj.LogSigma) / N;
            klTerm = 0; perLayer = zeros(1, numel(kls));
            for i = 1:numel(kls)
                kl_i = sum(kls{i}, 'all') / N;                                  % nats per sample, layer i
                perLayer(i) = double(gather(kl_i.Data));
                if obj.FreeBits > 0
                    % max(KL_g, lambda) per group g (batch-mean nats): the whole
                    % layer, each channel, or each latent unit (position x channel).
                    switch obj.FreeBitsPer
                        case "layer",   g = kl_i;
                        case "channel", g = sum(kls{i}, [1 2 4]) / N;
                        case "unit",    g = sum(kls{i}, 4) / N;
                    end
                    kl_i = sum(relu(g - obj.FreeBits) + obj.FreeBits, 'all');
                end
                klTerm = klTerm + kl_i;
            end
            L = nll + obj.Beta * klTerm;
            recon = double(gather(nll.Data)); kl = sum(perLayer);
            dims = numel(OceanDL.Autograd.Tensor.raw(X)) / N;
            obj.LastStats = struct('Recon', recon, 'KL', kl, 'KLPerLayer', perLayer, ...
                'NegELBO', recon + kl, 'BitsPerDim', (recon + kl) / (dims * log(2)));
        end

        function feats = bottomUp(obj, X, rec)
            % One feature map per level (after that level's blocks).
            % rec(name, h) (optional) records every intermediate activation (trace).
            if nargin < 3, rec = []; end
            h = obj.ConvIn.forward(X);
            emit(rec, "enc.L1.in", h);
            feats = cell(1, numel(obj.Widths));
            for l = 1:numel(obj.Widths)
                for b = 1:numel(obj.Up{l})
                    h = obj.Up{l}{b}.forward(h);
                    emit(rec, sprintf("enc.L%d.block%d", l, b), h);
                end
                feats{l} = h;
                if l < numel(obj.Widths)
                    h = obj.Down{l}.forward(h);
                    emit(rec, sprintf("enc.L%d.in", l + 1), h);
                end
            end
        end

        function [out, kls, zs, qs, ps] = topDown(obj, feats, N, imageSize, mode, zsGiven, temperature, stream, rec)
            % mode: "sample" | "mode" (posterior, needs feats) | "prior" | "given" (zsGiven).
            % "given" with an EMPTY entry zsGiven{i}: that layer takes its prior MEAN
            % (p(z_i | z_<i) at temperature 0), e.g. decode from the top latent only.
            % rec(name, value) (optional) records intermediate activations (trace).
            if nargin < 6, zsGiven = {}; end
            if nargin < 7 || isempty(temperature), temperature = 1; end
            if nargin < 8 || isempty(stream), stream = obj.Stream; end
            if nargin < 9, rec = []; end
            Lv = numel(obj.Widths);
            like = obj.H0.Data;
            f = 2 ^ (Lv - 1);
            h = obj.H0 + zeros(imageSize(1) / f, imageSize(2) / f, obj.Widths(Lv), N, 'like', like);
            emit(rec, sprintf("dec.L%d.start", Lv), h);
            kls = {}; zs = {}; qs = {}; ps = {}; i = 0;
            for l = Lv:-1:1
                e = [];
                if ~isempty(feats), e = feats{l}; end
                for j = 1:numel(obj.TD{l})
                    i = i + 1;
                    zin = [];
                    if mode == "given", zin = zsGiven{i}; end
                    if mode == "given" && isempty(zin)          % this layer: prior mean
                        [h, kl, z, q, p] = obj.TD{l}{j}.forward(h, [], "prior", [], 0, stream);
                    else
                        [h, kl, z, q, p] = obj.TD{l}{j}.forward(h, e, mode, zin, temperature, stream);
                    end
                    kls{end+1} = kl; zs{end+1} = z; qs{end+1} = q; ps{end+1} = p; %#ok<AGROW>
                    if ~isempty(q)
                        emit(rec, sprintf("z%d.mu", i), q.Mu);
                        emit(rec, sprintf("z%d.kl", i), kl);
                    end
                    emit(rec, sprintf("dec.L%d.td%d", l, j), h);
                end
                for b = 1:numel(obj.Res{l})
                    h = obj.Res{l}{b}.forward(h);
                    emit(rec, sprintf("dec.L%d.res%d", l, b), h);
                end
                if l > 1
                    h = obj.Lift{l - 1}.forward(OceanDL.Autograd.F.upsample(h, 2));
                    emit(rec, sprintf("dec.L%d.lift", l - 1), h);
                end
            end
            a = OceanDL.Autograd.F.silu(obj.OutNorm.forward(h));
            emit(rec, "dec.out.act", a);
            out = obj.ConvOut.forward(a);
            emit(rec, "dec.out", out);
        end

        function m = mean(obj, out)
            if obj.Likelihood == "bernoulli", m = sigmoid(OceanDL.Autograd.Tensor.wrap(out)); else, m = out; end
        end

        function [qs, zs] = encode(obj, X)
            % Posteriors of every latent layer (top first), propagated with posterior means.
            [~, ~, zs, qs] = OceanDL.Autograd.GradMode.noGrad(@() obj.topDown(obj.bottomUp(X), size(X, 4), size(X, 1:2), "mode"));
        end

        function Xr = reconstruct(obj, X)
            out = OceanDL.Autograd.GradMode.noGrad(@() obj.mean(obj.topDown(obj.bottomUp(X), size(X, 4), size(X, 1:2), "mode")));
            Xr = OceanDL.Autograd.Tensor.raw(out);
        end

        function out = decodeLatents(obj, zs, imageSize)
            % Decoder output (likelihood parameters) for given latents of every layer
            % (top first); an EMPTY entry makes that layer use its prior mean.
            k = find(~cellfun(@isempty, zs), 1);
            N = size(OceanDL.Autograd.Tensor.raw(zs{k}), 4);
            out = obj.topDown({}, N, imageSize, "given", zs);
        end

        function W = weightsFor(obj, layer)
            % WEIGHTSFOR  Weights (kh x kw x Cin x Cout) of the convolution that PRODUCES a traced
            %   layer: enc.L1.in = ConvIn; enc.L<l>.in = the stride-2 conv from level l-1;
            %   enc.L<l>.block<b> / dec.L<l>.res<b> = the block's last conv; dec.L<l>.lift = the
            %   lift conv; dec.out = ConvOut; dec.L<l>.td<j> = its block's last conv; dec.out.act
            %   (GroupNorm + SiLU, no conv of its own) = the last conv BEFORE it.
            %   [] for layers without one (latents, dec.L<l>.start = a learned constant).
            layer = string(layer); W = [];
            tok = regexp(char(layer), '^(enc|dec)\.L(\d+)\.(\w+?)(\d*)$', 'tokens', 'once');
            conv = [];
            if layer == "dec.out"
                conv = obj.ConvOut;
            elseif layer == "dec.out.act"
                if ~isempty(obj.Res{1}), conv = OceanDL.Generative.HierarchicalVAE.lastConv(obj.Res{1}{end});
                elseif ~isempty(obj.TD{1}), conv = OceanDL.Generative.HierarchicalVAE.lastConv(obj.TD{1}{end}.Block);
                else, conv = obj.Lift{1}; end
            elseif ~isempty(tok)
                l = str2double(tok{2}); b = str2double(tok{4});
                switch string(tok{1}) + "." + string(tok{3})
                    case "enc.in",    if l == 1, conv = obj.ConvIn; else, conv = obj.Down{l - 1}; end
                    case "enc.block", conv = OceanDL.Generative.HierarchicalVAE.lastConv(obj.Up{l}{b});
                    case "dec.res",   conv = OceanDL.Generative.HierarchicalVAE.lastConv(obj.Res{l}{b});
                    case "dec.lift",  conv = obj.Lift{l};
                    case "dec.td",    conv = OceanDL.Generative.HierarchicalVAE.lastConv(obj.TD{l}{b}.Block);
                end
            end
            if ~isempty(conv), W = OceanDL.Autograd.Tensor.raw(conv.Weight.Data); end
        end

        function [acts, order] = trace(obj, X, opts)
            % TRACE  Every intermediate activation by name (OceanDL.Interpret.ITraceable).
            %   Encoder "enc.L<l>.in" / "enc.L<l>.block<b>"; latents (posterior, top first)
            %   "z<i>.mu" and the per-unit KL "z<i>.kl"; decoder (latents = posterior means,
            %   as in reconstruct) "dec.L<l>.start|td<j>|res<b>|lift", "dec.out.act",
            %   "dec.out" (logits). 'Parts' "encoder" stops after the bottom-up pass.
            arguments
                obj
                X
                opts.Parts (1,1) string {mustBeMember(opts.Parts, ["all", "encoder"])} = "all"
            end
            acts = containers.Map('KeyType', 'char', 'ValueType', 'any');
            order = strings(1, 0);
            function rec(name, value)
                acts(char(name)) = value;
                order(end+1) = string(name);
            end
            raw = OceanDL.Autograd.Tensor.raw(X);
            feats = obj.bottomUp(X, @rec);
            if opts.Parts == "all"
                obj.topDown(feats, size(raw, 4), size(raw, 1:2), "mode", {}, 1, [], @rec);
            end
        end

        function X = sample(obj, n, imageSize, opts)
            % Ancestral sampling from the priors, top to bottom (no encoder).
            arguments
                obj
                n (1,1) double
                imageSize (1,2) double
                opts.Temperature (1,1) double {mustBeNonnegative} = 1
                opts.Stream = []
            end
            stream = opts.Stream; if isempty(stream), stream = obj.Stream; end
            out = OceanDL.Autograd.GradMode.noGrad(@() obj.mean(obj.topDown({}, n, imageSize, "prior", {}, opts.Temperature, stream)));
            X = OceanDL.Autograd.Tensor.raw(out);
        end

        function [n, variances] = activeUnits(obj, X, threshold)
            % Active units per latent layer (top first): Var_x(E_q[z_u]) > threshold.
            if nargin < 3, threshold = 0.01; end
            qs = obj.encode(X);
            n = zeros(1, numel(qs)); variances = cell(1, numel(qs));
            for i = 1:numel(qs)
                mu = double(gather(OceanDL.Autograd.Tensor.raw(qs{i}.Mu)));
                variances{i} = var(mu, 0, 4);
                n(i) = nnz(variances{i} > threshold);
            end
        end

        function T = costEstimate(obj, imageSize, opts)
            % Convolution FLOPs per level for one image (forward), and the
            % training-step estimate (forward + backward ~ 3 x forward). Conv
            % layers dominate; norms, activations and additions are ignored.
            % 'GFLOPS' (measured throughput of this machine) adds seconds per step.
            arguments
                obj
                imageSize (1,2) double
                opts.Batch (1,1) double = 64
                opts.GFLOPS (1,1) double = NaN
            end
            Lv = numel(obj.Widths);
            res = @(l) imageSize / 2 ^ (l - 1);
            conv = @(mods, l) OceanDL.Generative.HierarchicalVAE.convFlops(mods, res(l));
            enc = zeros(Lv, 1); dec = zeros(Lv, 1);
            enc(1) = conv({obj.ConvIn}, 1);
            for l = 1:Lv
                enc(l) = enc(l) + conv(obj.Up{l}, l);
                if l < Lv, enc(l) = enc(l) + conv({obj.Down{l}}, l); end
                dec(l) = conv(obj.TD{l}, l) + conv(obj.Res{l}, l);
                if l < Lv, dec(l) = dec(l) + conv({obj.Lift{l}}, l); end
            end
            dec(1) = dec(1) + conv({obj.ConvOut}, 1);
            Level = (1:Lv).'; Resolution = arrayfun(@(l) sprintf('%dx%d', res(l)), Level, 'UniformOutput', false);
            Latents = obj.LatentsPerLevel(:);
            EncoderMFLOP = enc / 1e6; DecoderMFLOP = dec / 1e6;
            T = table(Level, Resolution, Latents, EncoderMFLOP, DecoderMFLOP);
            fwd = sum(enc + dec);
            T.Properties.UserData = struct('ForwardGFLOPPerImage', fwd / 1e9, ...
                'StepGFLOP', 3 * fwd * opts.Batch / 1e9, 'Batch', opts.Batch, ...
                'SecondsPerStep', 3 * fwd * opts.Batch / 1e9 / opts.GFLOPS);
        end
    end

    methods (Static)
        function c = lastConv(block)
            % The final convolution of a residual / bottleneck block.
            if isprop(block, 'Conv4'), c = block.Conv4; else, c = block.Conv2; end
        end

        function f = convFlops(mods, hw)
            % 2 * Ho * Wo * KH * KW * Cin * Cout summed over every Conv2d inside mods.
            f = 0;
            stack = mods;
            while ~isempty(stack)
                m = stack{end}; stack(end) = [];
                if isempty(m), continue; end
                if iscell(m), stack = [stack, m]; continue; end %#ok<AGROW>
                if isa(m, 'OceanDL.nn.Conv2d')
                    [kh, kw, ci, co] = size(m.Weight.Data, 1:4);
                    o = ceil(hw ./ m.Stride);
                    f = f + 2 * o(1) * o(2) * kh * kw * ci * co;
                else
                    stack = [stack, m.children()]; %#ok<AGROW>
                end
            end
        end
    end

    methods (Access = protected)
        function applyBuffers(obj, fn)
            if ~isa(obj.LogSigma, 'OceanDL.nn.Parameter'), obj.LogSigma = fn(obj.LogSigma); end
        end
    end
end

function emit(rec, name, value)
% Record an activation when tracing (rec = [] otherwise: no cost).
if ~isempty(rec), rec(string(name), value); end
end
