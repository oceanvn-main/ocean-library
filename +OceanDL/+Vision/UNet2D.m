classdef UNet2D < OceanDL.nn.Module
    % UNET2D  Diffusion U-Net (Ronneberger 2015 skip structure; DDPM / ADM
    %   blocks, Ho 2020, Dhariwal & Nichol 2021): forward(x, t, y) -> same
    %   shape as x (the prediction: eps, x0, v or velocity).
    %
    %   emb = TimestepEmbedding(t) [+ ClassEmbed(y)]            (E x N, E = 4 Widths(1))
    %   Conv3x3 in -> push
    %   DOWN  level l = 1..L:  ResPerLevel x { ConvResBlock(emb) [+ attention] -> push }
    %                          stride-2 conv (l < L) -> push
    %   MID   ConvResBlock, SpatialSelfAttention, ConvResBlock
    %   UP    level l = L..1:  (ResPerLevel + 1) x { cat(h, pop) -> ConvResBlock [+ attention] }
    %                          nearest x2 + conv (l > 1)
    %   OUT   GN -> SiLU -> Conv3x3, ZERO-initialised (the network starts by
    %         predicting 0 -- the mean of eps -- which stabilises early training)
    %
    %   CLASS CONDITIONING ('NumClasses' K > 0): embedding table with K + 1
    %   rows; row K + 1 is the NULL label used for classifier-free guidance
    %   (y = [] or y = K + 1 -> unconditional). H and W must be divisible by 2^(L-1).

    properties (SetAccess = protected)
        TimeEmbed
        ClassEmbed          % [] or Embedding(K + 1, E)
        ConvIn
        Down cell = {}      % modules in execution order
        DownOps (1,:) string = strings(1, 0)   % "res" | "attn" | "down" | "push"
        Mid cell = {}
        Up cell = {}
        UpOps (1,:) string = strings(1, 0)     % "pop" | "res" | "attn" | "up"
        OutNorm
        ConvOut
        NumClasses (1,1) double
        Factor (1,1) double
    end

    methods
        function obj = UNet2D(opts)
            arguments
                opts.InChannels (1,1) double = 1
                opts.OutChannels (1,1) double = 0              % 0 = InChannels
                opts.Widths (1,:) double = [32 64]
                opts.ResPerLevel (1,1) double = 1
                opts.AttnLevels (1,:) logical = logical([])     % default: attention at the lowest level only
                opts.Heads (1,1) double = 1
                opts.NumClasses (1,1) double = 0
                opts.Groups (1,1) double = 32
                opts.Dropout (1,1) double = 0
                opts.ScaleShift (1,1) logical = true
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            W = opts.Widths; L = numel(W);
            attn = opts.AttnLevels; if isempty(attn), attn = (1:L) == L; end
            outC = opts.OutChannels; if outC == 0, outC = opts.InChannels; end
            E = 4 * W(1);
            seed = @(k) OceanDL.Vision.ConvEncoder.offset(opts.Seed, k);
            dt = {'DataType', opts.DataType};
            res = @(ci, co, k) OceanDL.Vision.ConvResBlock(ci, co, 'Groups', opts.Groups, 'Dropout', opts.Dropout, ...
                'EmbedDim', E, 'ScaleShift', opts.ScaleShift, 'Seed', seed(k), dt{:});
            att = @(c, k) OceanDL.Vision.SpatialSelfAttention(c, 'Heads', opts.Heads, 'Groups', opts.Groups, 'Seed', seed(k), dt{:});

            obj.TimeEmbed = OceanDL.nn.TimestepEmbedding(E, 'FreqDim', W(1), 'Seed', seed(1), dt{:});
            obj.NumClasses = opts.NumClasses;
            if opts.NumClasses > 0
                obj.ClassEmbed = OceanDL.nn.Embedding(opts.NumClasses + 1, E, 'Seed', seed(2), dt{:});
            end
            obj.ConvIn = OceanDL.nn.Conv2d(opts.InChannels, W(1), 3, 'Padding', 1, 'Seed', seed(3), dt{:});
            ch = W(1); skips = ch; k = 100;
            for l = 1:L
                for r = 1:opts.ResPerLevel
                    k = k + 10; obj.addDown(res(ch, W(l), k), "res"); ch = W(l);
                    if attn(l), k = k + 10; obj.addDown(att(ch, k), "attn"); end
                    obj.DownOps(end+1) = "push"; skips(end+1) = ch; %#ok<AGROW>
                end
                if l < L
                    k = k + 10;
                    obj.addDown(OceanDL.nn.Conv2d(ch, ch, 3, 'Stride', 2, 'Padding', 1, 'Seed', seed(k), dt{:}), "down");
                    obj.DownOps(end+1) = "push"; skips(end+1) = ch; %#ok<AGROW>
                end
            end
            obj.Mid = {res(ch, ch, 500), att(ch, 510), res(ch, ch, 520)};
            k = 600;
            for l = L:-1:1
                for r = 1:opts.ResPerLevel + 1
                    sc = skips(end); skips(end) = [];
                    obj.UpOps(end+1) = "pop";
                    k = k + 10; obj.addUp(res(ch + sc, W(l), k), "res"); ch = W(l);
                    if attn(l), k = k + 10; obj.addUp(att(ch, k), "attn"); end
                end
                if l > 1
                    k = k + 10;
                    obj.addUp(OceanDL.nn.Conv2d(ch, ch, 3, 'Padding', 1, 'Seed', seed(k), dt{:}), "up");
                end
            end
            obj.OutNorm = OceanDL.nn.GroupNorm(OceanDL.nn.GroupNorm.groupsFor(ch, opts.Groups), ch, dt{:});
            obj.ConvOut = OceanDL.nn.Conv2d(ch, outC, 3, 'Padding', 1, 'Seed', seed(900), dt{:});
            obj.ConvOut.Weight.Data(:) = 0; obj.ConvOut.Bias.Data(:) = 0;
            obj.Factor = 2 ^ (L - 1);
        end

        function y = forward(obj, x, t, labels)
            if nargin < 4, labels = []; end
            N = size(x, 4);
            emb = obj.TimeEmbed.forward(t);
            if ~isempty(obj.ClassEmbed)
                if isempty(labels), labels = repmat(obj.NumClasses + 1, N, 1); end
                emb = emb + obj.ClassEmbed.forward(reshape(labels, N, 1));      % ids N x 1 -> E x N
            end
            F = OceanDL.Autograd.F;
            h = obj.ConvIn.forward(x);
            stack = {h}; m = 0;
            for op = obj.DownOps
                switch op
                    case "push", stack{end+1} = h; %#ok<AGROW>
                    case "res",  m = m + 1; h = obj.Down{m}.forward(h, emb);
                    otherwise,   m = m + 1; h = obj.Down{m}.forward(h);
                end
            end
            h = obj.Mid{1}.forward(h, emb); h = obj.Mid{2}.forward(h); h = obj.Mid{3}.forward(h, emb);
            m = 0;
            for op = obj.UpOps
                switch op
                    case "pop", h = cat(3, h, stack{end}); stack(end) = [];
                    case "res", m = m + 1; h = obj.Up{m}.forward(h, emb);
                    case "up",  m = m + 1; h = obj.Up{m}.forward(F.upsample(h, 2));
                    otherwise,  m = m + 1; h = obj.Up{m}.forward(h);
                end
            end
            y = obj.ConvOut.forward(F.silu(obj.OutNorm.forward(h)));
        end
    end

    methods (Access = private)
        function addDown(obj, m, kind)
            obj.Down{end+1} = m; obj.DownOps(end+1) = kind;
        end

        function addUp(obj, m, kind)
            obj.Up{end+1} = m; obj.UpOps(end+1) = kind;
        end
    end
end
