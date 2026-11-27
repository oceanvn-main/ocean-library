classdef MaskedAutoencoder < OceanDL.nn.Module
    % MASKEDAUTOENCODER  MAE self-supervised pre-training (He et al. 2022;
    %   Szeliski 2nd ed. Ch. 5 takeaway 2) -- the vision analogue of BERT's
    %   masked-word prediction: hide most patches, reconstruct their pixels.
    %
    %   1. patchify + positions (the wrapped ViT's embed(): sincos, no [CLS])
    %   2. per image, keep a random (1 - MaskRatio) subset of the T tokens
    %   3. ENCODER = the ViT's own Transformer encoder, on VISIBLE tokens only
    %      (75% masking -> 4x fewer tokens -> ~16x cheaper attention)
    %   4. DECODER (small): project to DecoderDim, insert a shared learned
    %      [MASK] vector at every hidden position, restore the original order,
    %      add fixed 2-D positions, a few Transformer blocks, a linear map to
    %      the p x p x C pixels of each patch
    %   5. LOSS = mean squared error on the MASKED patches only (visible ones
    %      are trivially copyable); 'NormPixLoss' normalises each target patch
    %      to zero mean / unit variance first (MAE's default, sharper features).
    %      CAVEAT (measured): on images with flat regions this turns sensor-level
    %      noise into unit-variance targets that cannot be predicted -- 25% of
    %      USPS digit patches are flat and the loss plateaus at ~0.6 vs ~0.43
    %      without it. Prefer NormPixLoss = false for such data.
    %
    %   After pre-training, the wrapped ViT (PatchEmbedding + Encoder) is the
    %   pre-trained backbone: fine-tune it directly with vit.forward().
    %
    %   vit = OceanDL.Vision.ViT(..., 'Positions', "sincos", 'Pooling', "mean");
    %   mae = OceanDL.Vision.MaskedAutoencoder(vit, 'MaskRatio', 0.75);
    %   L = mae.forward(X);  L.backward();                   % pre-training step

    properties (SetAccess = protected)
        Encoder             % the wrapped ViT (Pooling "mean", Positions "sincos")
        DecoderEmbed        % Linear D -> Dd
        MaskToken           % Parameter (Dd x 1 x 1)
        Decoder             % TransformerEncoder
        DecoderPred         % Linear Dd -> p*p*C
        MaskRatio (1,1) double
        NormPixLoss (1,1) logical
        PatchSize (1,2) double
    end

    properties (SetAccess = private)
        Stream
        LastMask = []       % (N x T) logical, true = hidden, from the last forward()
        LastPrediction = [] % (p*p*C x N x T) raw predictions, from the last forward()
    end

    methods
        function obj = MaskedAutoencoder(vit, opts)
            arguments
                vit OceanDL.Vision.ViT
                opts.MaskRatio (1,1) double {mustBeInRange(opts.MaskRatio, 0, 1, 'exclusive')} = 0.75
                opts.DecoderDim (1,1) double = 64
                opts.DecoderDepth (1,1) double = 2
                opts.DecoderHeads (1,1) double = 4
                opts.NormPixLoss (1,1) logical = true
                opts.Seed = []
            end
            if vit.Pooling ~= "mean" || vit.PosMode ~= "sincos"
                error('OceanDL:Vision:MaskedAutoencoder:Backbone', ...
                    'Wrap a ViT built with ''Pooling'', "mean" and ''Positions'', "sincos" (no [CLS]; fixed positions).');
            end
            s = @(k) OceanDL.Vision.MaskedAutoencoder.offset(opts.Seed, k);
            D = vit.Patch.Dim;
            p = vit.Patch.PatchSize;
            C = size(vit.Patch.Proj.Weight, 3);
            obj.Encoder = vit;
            obj.PatchSize = p;
            obj.DecoderEmbed = OceanDL.nn.Linear(D, opts.DecoderDim, 'Seed', s(1));
            if isempty(opts.Seed), z = randn(opts.DecoderDim, 1, 1); else, z = randn(RandStream('mt19937ar', 'Seed', s(2)), opts.DecoderDim, 1, 1); end
            obj.MaskToken = OceanDL.nn.Parameter(0.02 * z);
            obj.Decoder = OceanDL.Transformer.TransformerEncoder(opts.DecoderDepth, opts.DecoderDim, opts.DecoderHeads, ...
                4 * opts.DecoderDim, 'Seed', s(3));
            obj.DecoderPred = OceanDL.nn.Linear(opts.DecoderDim, prod(p) * C, 'Seed', s(4));
            obj.MaskRatio = opts.MaskRatio;
            obj.NormPixLoss = opts.NormPixLoss;
            if isempty(opts.Seed), obj.Stream = RandStream.getGlobalStream();
            else, obj.Stream = RandStream('mt19937ar', 'Seed', s(5)); end
        end

        function L = forward(obj, X)
            vit = obj.Encoder;
            [tokens, grid] = vit.embed(X);                                  % D x N x T (positions added)
            [D, N, T] = size(tokens, 1:3);
            nKeep = max(1, round(T * (1 - obj.MaskRatio)));
            [~, order] = sort(rand(obj.Stream, N, T), 2);                  % a random permutation per image
            keep = order(:, 1:nKeep);                                       % N x nKeep token indices

            % 1. encode visible tokens only
            visible = OceanDL.Vision.MaskedAutoencoder.gather(tokens, keep);   % D x N x nKeep
            enc = vit.Encoder.forward(visible);

            % 2. decoder input: visible features scattered back, [MASK] elsewhere
            z = obj.DecoderEmbed.forward(enc);                              % Dd x N x nKeep
            Dd = size(z, 1);
            hidden = true(N, T);
            hidden(sub2ind([N T], repmat((1:N).', 1, nKeep), keep)) = false;
            full = cat(3, z, obj.MaskToken + zeros(Dd, N, T - nKeep));      % visible first, then masks
            inv = zeros(N, T);                                              % position of every token in `full`
            inv(sub2ind([N T], repmat((1:N).', 1, nKeep), keep)) = repmat(1:nKeep, N, 1);
            rest = sort(order(:, nKeep+1:end), 2);                          % masked token ids, any fixed order
            inv(sub2ind([N T], repmat((1:N).', 1, T - nKeep), rest)) = repmat(nKeep + (1:T - nKeep), N, 1);
            dec = OceanDL.Vision.MaskedAutoencoder.gather(full, inv);       % original token order
            P = OceanDL.Transformer.PositionalEncoding.sincos2d(grid(1), grid(2), Dd);
            dec = obj.Decoder.forward(dec + P);
            pred = obj.DecoderPred.forward(dec);                            % (p p C) x N x T

            % 3. loss on masked patches
            target = OceanDL.Vision.MaskedAutoencoder.patchify(X, obj.PatchSize);   % (p p C) x N x T
            if obj.NormPixLoss
                target = (target - mean(target, 1)) ./ sqrt(var(target, 1, 1) + 1e-6);
            end
            w = reshape(double(hidden), 1, N, T);
            err = mean((pred - target) .^ 2, 1);                           % per-patch MSE (1 x N x T)
            L = sum(err .* w, 'all') / sum(w, 'all');
            obj.LastMask = hidden;
            obj.LastPrediction = OceanDL.Autograd.Tensor.raw(pred);
        end
    end

    methods (Static)
        function Y = gather(X, idx)
            % Y(:, n, k) = X(:, n, idx(n, k)) for a Tensor or array X (D x N x T),
            % different indices per sample, via one linear index (differentiable).
            [D, N, T] = size(X, 1:3);
            K = size(idx, 2);
            d = (1:D).';
            n = reshape(repmat((1:N).', 1, K), 1, N * K);
            t = reshape(idx, 1, N * K);
            lin = d + D * (n - 1) + D * N * (t - 1);                       % D x (N K), n fastest
            Y = reshape(X(lin(:)), D, N, K);
            if T == 0, Y = X; end
        end

        function P = patchify(X, p)
            % (H x W x C x N) -> (p p C) x N x T, same token order as PatchEmbedding
            % (grid row fastest) and same in-patch order as the conv weights (kh, kw, c).
            [H, W, C, N] = size(X, 1:4);
            Hp = H / p(1); Wp = W / p(2);
            P = reshape(X, p(1), Hp, p(2), Wp, C, N);
            P = permute(P, [1 3 5 6 2 4]);                                 % kh, kw, c, n, r, col
            P = reshape(P, p(1) * p(2) * C, N, Hp * Wp);
        end

        function X = unpatchify(P, p, Hp, Wp)
            [K, N, ~] = size(P, 1:3);
            C = K / prod(p);
            X = reshape(P, p(1), p(2), C, N, Hp, Wp);
            X = ipermute(X, [1 3 5 6 2 4]);
            X = reshape(X, p(1) * Hp, p(2) * Wp, C, N);
        end
    end

    methods (Static, Access = private)
        function s = offset(seed, k)
            if isempty(seed), s = []; else, s = seed + k; end
        end
    end
end
