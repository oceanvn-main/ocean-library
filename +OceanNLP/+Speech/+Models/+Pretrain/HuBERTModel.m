classdef HuBERTModel < OceanDL.nn.Module
    % HUBERTMODEL  HuBERT-style masked hidden-unit prediction encoder (Hsu et al. 2021), CPU scale.
    %
    %   m = OceanNLP.Speech.Models.Pretrain.HuBERTModel(inputDim, 'ConvChannels', [128 128], ...
    %         'ConvKernels', [5 3], 'ConvStrides', [2 1], 'Dim', 128, 'Depth', 3, 'Heads', 4, ...
    %         'FFNDim', 256, 'Dropout', 0.1, 'NumUnits', 50, 'ProjDim', 64, 'Temperature', 0.1, ...
    %         'MaskProb', 0.13, 'MaskLength', 5, 'MaxSpanFraction', 0.5, 'MinSpans', 1, 'Alpha', 1, 'Seed', 0)
    %   [H, lens] = m.forward(X, lens)       ENCODER CONTRACT (no masking): X F x N x T, H Dim x N x T'
    %   out = m.forwardMasked(X, lens, M)    M: N x T' logical (true = masked); out.H, .Logits (K x N x T'), .Lengths
    %   [L, st] = m.pretrainLoss(X, lens, Y, M)   Y: N x T' unit ids at the OUTPUT frame rate
    %   M = HuBERTModel.spanMask(lens, T, p, l, stream, ...)   (static, pure)
    %   idx = m.targetFrameIndex(T')         input frame aligned with each output frame
    %   Z = m.hiddenStates(X, lens, layer)   raw D x N x T' after Transformer layer `layer` (0 = input)
    %   m.dropPretrainHead()                 remove projection + unit embeddings (before CTC fine-tuning)
    %   cfg = m.config();  m2 = HuBERTModel.fromConfig(cfg)
    %
    %   ARCHITECTURE, utterance with T valid input frames x_1..x_T (F x 1 log-mel):
    %     (1) feature encoder  z = ConvEncoder(x)  Conv1d stack (padding floor(k/2), stride s_l),
    %         LayerNorm, GELU, exact length propagation T' = outputLength(T)       [D5 deviation]
    %     (2) projection       e_t = W_p z_t + b_p                     (Dim)
    %     (3) masking          e~_t = m_t ? w_mask : e_t,  w_mask a learned Dim-vector
    %     (4) positions        e~_t + PE(t-1),  PE = fixed sinusoids (Vaswani et al. 2017)
    %     (5) context          h = TransformerEncoder(dropout(e~)) with a key padding mask
    %         (pre-LN blocks + final LayerNorm), padded frames re-zeroed
    %     (6) unit logits      u_t = W_o h_t + b_o (ProjDim);
    %         l_{t,c} = cos(u_t, g_c) / tau,  cos(a, b) = a.b / (||a|| ||b||),  tau = 0.1,
    %         g_c (c = 1..K) learned unit embeddings.
    %   LOSS (HuBERT eq. 3/4, alpha = 1 by default):
    %     L = alpha * L_m + (1 - alpha) * L_u,
    %     L_m = -(1/|M|) sum_{(n,t) in M} log softmax(l_{n,t})_{y_{n,t}},   M = masked VALID frames,
    %     L_u the same over unmasked valid frames. Padded frames are in neither set and are never
    %     masked. With alpha = 1 the unmasked targets do not affect L (tested by perturbing them).
    %
    %   SPAN MASKING (wav2vec 2.0 / HuBERT): every valid output frame t <= T'_n is a span start
    %   independently with probability p; a start s masks frames s .. min(s + l_n - 1, T'_n).
    %   Short utterances: l_n = min(l, max(1, floor(MaxSpanFraction * T'_n))) so one span cannot
    %   swallow a short digit; if fewer than MinSpans starts are drawn, the missing ones are
    %   drawn uniformly among the remaining frames. With MinSpans = 0 the expected masked
    %   fraction has the closed form  E = (1/T') sum_{t=1}^{T'} [1 - (1-p)^{min(t, l_n)}].
    %   Defaults p = 0.13, l = 5 at the 20 ms output rate (stride 2 of 10 ms frames) give
    %   ~50% masked frames in 100 ms spans -- the HuBERT paper's p = 8%, l = 10 (200 ms spans,
    %   ~57%) rescaled for ~0.5 s isolated digits.
    %
    %   DEVIATIONS from Hsu et al. 2021 (CPU scale, contract D5): conv encoder on 10 ms log-mel
    %   frames instead of the raw-waveform 7-layer CNN; fixed sinusoidal instead of convolutional
    %   relative positional embedding; small Dim/Depth. Targets at input-frame rate are mapped to
    %   output frames by targetFrameIndex (centre of each output frame's receptive field):
    %   through layer l, input index = (t - 1) s_l + 1 (exact centre for odd k_l).
    %
    %   FINE-TUNING: forward(X, lens) never masks, so the model satisfies the encoder contract
    %   (InputDim, OutputDim, forward, outputLength, config) and can be passed to
    %   CTCRecognizer; call dropPretrainHead() first (HuBERT discards the projection and
    %   unit embeddings). Register with RecognizerFactory for checkpoint round trips.
    %   Input normalisation is NOT inside the model: pretraining (MaskedUnitTrainer) applies the
    %   same per-utterance CMVN as CTCRecognizer 'InputNorm' "utterance" (normalizeUtterance).

    properties (SetAccess = protected)
        FeatureEncoder
        Project
        MaskEmbedding
        Drop
        Transformer
        OutProject
        UnitEmbeddings
        InputDim (1,1) double
        ConvChannels (1,:) double
        ConvKernels (1,:) double
        ConvStrides (1,:) double
        Dim (1,1) double
        Depth (1,1) double
        Heads (1,1) double
        FFNDim (1,1) double
        DropoutP (1,1) double
        NumUnits (1,1) double
        ProjDim (1,1) double
        Temperature (1,1) double
        MaskProb (1,1) double
        MaskLength (1,1) double
        MaxSpanFraction (1,1) double
        MinSpans (1,1) double
        Alpha (1,1) double
        HasPretrainHead (1,1) logical
        Seed (1,1) double
        CodebookVersion (1,1) string = ""
    end

    properties (Dependent)
        OutputDim
    end

    methods
        function obj = HuBERTModel(inputDim, opts)
            arguments
                inputDim (1,1) double {mustBePositive, mustBeInteger}
                opts.ConvChannels (1,:) double {mustBePositive, mustBeInteger} = [128 128]
                opts.ConvKernels (1,:) double {mustBePositive, mustBeInteger} = [5 3]
                opts.ConvStrides (1,:) double {mustBePositive, mustBeInteger} = [2 1]
                opts.Dim (1,1) double {mustBePositive, mustBeInteger} = 128
                opts.Depth (1,1) double {mustBePositive, mustBeInteger} = 3
                opts.Heads (1,1) double {mustBePositive, mustBeInteger} = 4
                opts.FFNDim (1,1) double {mustBePositive, mustBeInteger} = 256
                opts.Dropout (1,1) double {mustBeGreaterThanOrEqual(opts.Dropout, 0), mustBeLessThan(opts.Dropout, 1)} = 0.1
                opts.NumUnits (1,1) double {mustBePositive, mustBeInteger} = 50
                opts.ProjDim (1,1) double {mustBePositive, mustBeInteger} = 64
                opts.Temperature (1,1) double {mustBePositive} = 0.1
                opts.MaskProb (1,1) double {mustBeGreaterThanOrEqual(opts.MaskProb, 0), mustBeLessThanOrEqual(opts.MaskProb, 1)} = 0.13
                opts.MaskLength (1,1) double {mustBePositive, mustBeInteger} = 5
                opts.MaxSpanFraction (1,1) double {mustBePositive, mustBeLessThanOrEqual(opts.MaxSpanFraction, 1)} = 0.5
                opts.MinSpans (1,1) double {mustBeNonnegative, mustBeInteger} = 1
                opts.Alpha (1,1) double {mustBeGreaterThanOrEqual(opts.Alpha, 0), mustBeLessThanOrEqual(opts.Alpha, 1)} = 1
                opts.PretrainHead (1,1) logical = true
                opts.Seed (1,1) double {mustBeInteger, mustBeNonnegative} = 0
                opts.CodebookVersion (1,1) string = ""
            end
            if mod(opts.Dim, 2) || mod(opts.Dim, opts.Heads)
                error('OceanNLP:Speech:Models:HuBERTModel:Dim', 'Dim (%d) must be even and divisible by Heads (%d).', opts.Dim, opts.Heads);
            end
            obj.InputDim = inputDim;
            obj.ConvChannels = opts.ConvChannels;
            obj.ConvKernels = opts.ConvKernels;
            obj.ConvStrides = opts.ConvStrides;
            obj.Dim = opts.Dim;
            obj.Depth = opts.Depth;
            obj.Heads = opts.Heads;
            obj.FFNDim = opts.FFNDim;
            obj.DropoutP = opts.Dropout;
            obj.NumUnits = opts.NumUnits;
            obj.ProjDim = opts.ProjDim;
            obj.Temperature = opts.Temperature;
            obj.MaskProb = opts.MaskProb;
            obj.MaskLength = opts.MaskLength;
            obj.MaxSpanFraction = opts.MaxSpanFraction;
            obj.MinSpans = opts.MinSpans;
            obj.Alpha = opts.Alpha;
            obj.Seed = opts.Seed;
            obj.CodebookVersion = opts.CodebookVersion;
            s = opts.Seed;
            obj.FeatureEncoder = OceanNLP.Speech.Models.Encoders.ConvEncoder(inputDim, 'Channels', opts.ConvChannels, ...
                'KernelSizes', opts.ConvKernels, 'Strides', opts.ConvStrides, 'Activation', "gelu", ...
                'LayerNorm', true, 'Dropout', 0, 'Seed', s + 1);
            obj.Project = OceanDL.nn.Linear(opts.ConvChannels(end), opts.Dim, 'Seed', s + 20);
            obj.MaskEmbedding = OceanDL.nn.Parameter(rand(RandStream('mt19937ar', 'Seed', s + 21), opts.Dim, 1));
            obj.Drop = OceanDL.nn.Dropout(opts.Dropout);
            obj.Transformer = OceanDL.Transformer.TransformerEncoder(opts.Depth, opts.Dim, opts.Heads, opts.FFNDim, ...
                'Norm', "pre", 'Dropout', opts.Dropout, 'Seed', s + 30);
            obj.HasPretrainHead = false;
            if opts.PretrainHead
                obj.HasPretrainHead = true;
                obj.OutProject = OceanDL.nn.Linear(opts.Dim, opts.ProjDim, 'Seed', s + 40);
                obj.UnitEmbeddings = OceanDL.nn.Parameter(randn(RandStream('mt19937ar', 'Seed', s + 41), opts.ProjDim, opts.NumUnits));
            end
        end

        function d = get.OutputDim(obj)
            d = obj.Dim;
        end

        function [H, lens] = forward(obj, X, lens)
            % FORWARD  Encoder contract: unmasked context representations (Dim x N x T').
            arguments
                obj
                X
                lens (1,:) double {mustBeNonnegative, mustBeInteger}
            end
            [H, lens] = obj.encode(X, lens, []);
        end

        function out = forwardMasked(obj, X, lens, M)
            % FORWARDMASKED  Context with frames M replaced by the mask embedding + unit logits.
            arguments
                obj
                X
                lens (1,:) double {mustBeNonnegative, mustBeInteger}
                M logical
            end
            [H, lens2] = obj.encode(X, lens, M);
            out = struct('H', H, 'Lengths', lens2, 'Mask', M, 'Logits', obj.unitLogits(H));
        end

        function Z = unitLogits(obj, H)
            % UNITLOGITS  l = cos(W_o h + b_o, g_c) / tau  ->  K x N x T'.
            if ~obj.HasPretrainHead
                error('OceanNLP:Speech:Models:HuBERTModel:NoHead', 'The pretraining head was dropped.');
            end
            [~, N, T] = size(H, 1:3);
            U = reshape(obj.OutProject.forward(H), obj.ProjDim, N * T);
            Un = U ./ sqrt(sum(U .^ 2, 1) + 1e-12);
            G = obj.UnitEmbeddings;
            Gn = G ./ sqrt(sum(G .^ 2, 1) + 1e-12);
            Z = reshape((Gn.' * Un) * (1 / obj.Temperature), obj.NumUnits, N, T);
        end

        function [L, st] = pretrainLoss(obj, X, lens, Y, M)
            % PRETRAINLOSS  alpha * CE(masked valid) + (1 - alpha) * CE(unmasked valid).
            %   Y: N x T' unit ids (any value at padded frames is ignored).
            %   st: Loss, MaskedLoss, UnmaskedLoss, MaskedAccuracy, UnmaskedAccuracy, NumMasked, NumUnmasked.
            arguments
                obj
                X
                lens (1,:) double {mustBeNonnegative, mustBeInteger}
                Y double
                M logical
            end
            out = obj.forwardMasked(X, lens, M);
            T = size(out.Logits, 3);
            N = numel(lens);
            if ~isequal(size(Y), [N T]) || ~isequal(size(M), [N T])
                error('OceanNLP:Speech:Models:HuBERTModel:TargetShape', ...
                    'Targets and mask must be N x T'' = %d x %d.', N, T);
            end
            valid = OceanDL.Layout.SequenceMask.fromLengths(out.Lengths, T);
            if any(M & ~valid, 'all')
                error('OceanNLP:Speech:Models:HuBERTModel:MaskPadding', 'The mask covers padded frames.');
            end
            Mm = M & valid;
            Mu = ~M & valid;
            Ys = Y;
            Ys(~valid) = 1;                                     % any legal id; excluded from the loss
            if any(Ys(valid) < 1 | Ys(valid) > obj.NumUnits | Ys(valid) ~= round(Ys(valid)))
                error('OceanNLP:Speech:Models:HuBERTModel:TargetRange', 'Targets must be unit ids in 1..%d.', obj.NumUnits);
            end
            F = OceanDL.Autograd.F;
            Lm = []; Lu = [];
            if obj.Alpha > 0 && any(Mm, 'all')
                Lm = F.crossEntropy(out.Logits, Ys, 'Mask', Mm);
                L = Lm * obj.Alpha;
            else
                L = OceanDL.Autograd.Tensor(0) * sum(out.Logits, 'all');   % zero, still in the graph
            end
            if obj.Alpha < 1 && any(Mu, 'all')
                Lu = F.crossEntropy(out.Logits, Ys, 'Mask', Mu);
                L = L + Lu * (1 - obj.Alpha);
            end
            if nargout > 1
                Z = OceanDL.Autograd.Tensor.raw(out.Logits);
                [~, pred] = max(Z, [], 1);
                hit = reshape(pred, N, T) == Ys;
                st = struct('Loss', double(L.Data), 'MaskedLoss', valOrNaN(Lm), 'UnmaskedLoss', valOrNaN(Lu), ...
                    'MaskedAccuracy', nnz(hit & Mm) / max(nnz(Mm), 1), ...
                    'UnmaskedAccuracy', nnz(hit & Mu) / max(nnz(Mu), 1), ...
                    'NumMasked', nnz(Mm), 'NumUnmasked', nnz(Mu));
                if ~any(Mm, 'all'), st.MaskedAccuracy = NaN; end
                if ~any(Mu, 'all'), st.UnmaskedAccuracy = NaN; end
            end
        end

        function M = sampleMask(obj, lens, T, stream)
            % SAMPLEMASK  Span mask with this model's MaskProb / MaskLength / limits.
            M = OceanNLP.Speech.Models.Pretrain.HuBERTModel.spanMask(lens, T, obj.MaskProb, obj.MaskLength, stream, ...
                'MaxSpanFraction', obj.MaxSpanFraction, 'MinSpans', obj.MinSpans);
        end

        function Z = hiddenStates(obj, X, lens, layer)
            % HIDDENSTATES  Raw D x N x T' output of Transformer layer LAYER (0 = its input),
            %   eval mode, no masking, padded frames zero. For iteration-2 re-clustering.
            arguments
                obj
                X
                lens (1,:) double {mustBeNonnegative, mustBeInteger}
                layer (1,1) double {mustBeInteger, mustBeNonnegative}
            end
            if layer > obj.Depth
                error('OceanNLP:Speech:Models:HuBERTModel:Layer', 'Layer %d of %d.', layer, obj.Depth);
            end
            was = obj.Training;
            obj.eval();
            restore = onCleanup(@() obj.train(was));
            guard = OceanDL.Autograd.GradMode.disable(); %#ok<NASGU>
            [E, ~, keyMask, valid] = obj.embed(X, lens, []);
            for l = 1:layer
                E = obj.Transformer.Layers{l}.forward(E, 'KeyMask', keyMask);
            end
            Z = OceanDL.Autograd.Tensor.raw(OceanDL.Layout.SequenceMask.apply(E, valid));
            clear restore
        end

        function lens = outputLength(obj, lens)
            arguments
                obj
                lens double {mustBeNonnegative, mustBeInteger}
            end
            lens = obj.FeatureEncoder.outputLength(lens);
        end

        function idx = targetFrameIndex(obj, Tout, Tin)
            % TARGETFRAMEINDEX  1 x Tout input-frame index aligned with each output frame:
            %   back through every conv layer, t_in = (t_out - 1) s + 1, clamped to Tin.
            arguments
                obj
                Tout (1,1) double {mustBeNonnegative, mustBeInteger}
                Tin (1,1) double {mustBeNonnegative, mustBeInteger} = inf
            end
            idx = 1:Tout;
            for l = numel(obj.ConvStrides):-1:1
                idx = (idx - 1) * obj.ConvStrides(l) + 1;
            end
            idx = min(idx, Tin);
        end

        function dropPretrainHead(obj)
            % DROPPRETRAINHEAD  Remove OutProject + UnitEmbeddings (unused by fine-tuning).
            obj.OutProject = [];
            obj.UnitEmbeddings = [];
            obj.HasPretrainHead = false;
        end

        function setCodebookVersion(obj, v)
            arguments
                obj
                v (1,1) string
            end
            obj.CodebookVersion = v;
        end

        function cfg = config(obj)
            cfg = struct('Type', "HuBERTModel", 'InputDim', obj.InputDim, 'ConvChannels', obj.ConvChannels, ...
                'ConvKernels', obj.ConvKernels, 'ConvStrides', obj.ConvStrides, 'Dim', obj.Dim, ...
                'Depth', obj.Depth, 'Heads', obj.Heads, 'FFNDim', obj.FFNDim, 'Dropout', obj.DropoutP, ...
                'NumUnits', obj.NumUnits, 'ProjDim', obj.ProjDim, 'Temperature', obj.Temperature, ...
                'MaskProb', obj.MaskProb, 'MaskLength', obj.MaskLength, 'MaxSpanFraction', obj.MaxSpanFraction, ...
                'MinSpans', obj.MinSpans, 'Alpha', obj.Alpha, 'PretrainHead', obj.HasPretrainHead, ...
                'Seed', obj.Seed, 'CodebookVersion', obj.CodebookVersion, 'OutputDim', obj.OutputDim);
        end
    end

    methods (Static)
        function obj = fromConfig(cfg)
            arguments
                cfg (1,1) struct
            end
            if string(cfg.Type) ~= "HuBERTModel"
                error('OceanNLP:Speech:Models:HuBERTModel:ConfigType', 'Config Type is "%s", not "HuBERTModel".', string(cfg.Type));
            end
            obj = OceanNLP.Speech.Models.Pretrain.HuBERTModel(cfg.InputDim, 'ConvChannels', cfg.ConvChannels, ...
                'ConvKernels', cfg.ConvKernels, 'ConvStrides', cfg.ConvStrides, 'Dim', cfg.Dim, 'Depth', cfg.Depth, ...
                'Heads', cfg.Heads, 'FFNDim', cfg.FFNDim, 'Dropout', cfg.Dropout, 'NumUnits', cfg.NumUnits, ...
                'ProjDim', cfg.ProjDim, 'Temperature', cfg.Temperature, 'MaskProb', cfg.MaskProb, ...
                'MaskLength', cfg.MaskLength, 'MaxSpanFraction', cfg.MaxSpanFraction, 'MinSpans', cfg.MinSpans, ...
                'Alpha', cfg.Alpha, 'PretrainHead', logical(cfg.PretrainHead), 'Seed', cfg.Seed, ...
                'CodebookVersion', string(cfg.CodebookVersion));
        end

        function M = spanMask(lens, T, p, l, stream, opts)
            % SPANMASK  N x T logical span mask over VALID frames only (see class help).
            arguments
                lens (1,:) double {mustBeNonnegative, mustBeInteger}
                T (1,1) double {mustBeNonnegative, mustBeInteger}
                p (1,1) double {mustBeGreaterThanOrEqual(p, 0), mustBeLessThanOrEqual(p, 1)}
                l (1,1) double {mustBePositive, mustBeInteger}
                stream (1,1) RandStream
                opts.MaxSpanFraction (1,1) double {mustBePositive} = 1
                opts.MinSpans (1,1) double {mustBeNonnegative, mustBeInteger} = 0
            end
            if any(lens > T)
                error('OceanNLP:Speech:Models:HuBERTModel:Lengths', 'A length exceeds T = %d.', T);
            end
            N = numel(lens);
            M = false(N, T);
            for n = 1:N
                Tn = lens(n);
                if Tn == 0, continue; end
                ln = OceanNLP.Speech.Models.Pretrain.HuBERTModel.effectiveSpan(Tn, l, opts.MaxSpanFraction);
                starts = find(rand(stream, 1, Tn) < p);
                need = min(opts.MinSpans, Tn) - numel(starts);
                if need > 0
                    rest = setdiff(1:Tn, starts);
                    starts = [starts, rest(randperm(stream, numel(rest), need))]; %#ok<AGROW>
                end
                for s = starts
                    M(n, s:min(s + ln - 1, Tn)) = true;
                end
            end
        end

        function e = expectedMaskFraction(Tn, p, l, maxSpanFraction)
            % EXPECTEDMASKFRACTION  Closed-form E[masked fraction] of spanMask with MinSpans = 0.
            arguments
                Tn (1,1) double {mustBePositive, mustBeInteger}
                p (1,1) double
                l (1,1) double
                maxSpanFraction (1,1) double = 1
            end
            ln = OceanNLP.Speech.Models.Pretrain.HuBERTModel.effectiveSpan(Tn, l, maxSpanFraction);
            t = 1:Tn;
            e = mean(1 - (1 - p) .^ min(t, ln));
        end

        function ln = effectiveSpan(Tn, l, frac)
            ln = min(l, max(1, floor(frac * Tn)));
        end

        function X = normalizeUtterance(X, lens)
            % NORMALIZEUTTERANCE  Per-utterance CMVN over valid frames, padding zeroed (raw arrays);
            %   the same formula as CTCRecognizer 'InputNorm' "utterance".
            X = double(X);
            for n = 1:size(X, 2)
                if lens(n) == 0, continue; end
                v = X(:, n, 1:lens(n));
                mu = mean(v, 3);
                s2 = mean((v - mu).^2, 3);
                X(:, n, 1:lens(n)) = (v - mu) ./ sqrt(s2 + 1e-8);
            end
            X = X .* reshape((1:size(X, 3)) <= lens(:), 1, size(X, 2), size(X, 3));
        end
    end

    methods (Access = private)
        function [E, lens, keyMask, valid] = embed(obj, X, lens, M)
            % Steps (1)-(4): conv features, projection, masking, positions, dropout.
            [Z, lens] = obj.FeatureEncoder.forward(X, lens);
            E = obj.Project.forward(Z);
            [~, N, T] = size(E, 1:3);
            valid = OceanDL.Layout.SequenceMask.fromLengths(lens, T);
            if ~isempty(M)
                if ~isequal(size(M), [N T])
                    error('OceanNLP:Speech:Models:HuBERTModel:MaskShape', 'Mask must be N x T'' = %d x %d.', N, T);
                end
                if any(M & ~valid, 'all')
                    error('OceanNLP:Speech:Models:HuBERTModel:MaskPadding', 'The mask covers padded frames.');
                end
                m = reshape(double(M), 1, N, T);
                E = E .* (1 - m) + obj.MaskEmbedding .* m;
            end
            if T > 0
                E = E + OceanNLP.Speech.Models.Pretrain.HuBERTModel.positions(T, obj.Dim);
            end
            E = OceanDL.Layout.SequenceMask.apply(obj.Drop.forward(E), valid);
            keyMask = valid;
            keyMask(lens == 0, 1) = true;                 % an empty utterance attends to a dummy (zeroed later)
        end

        function [H, lens] = encode(obj, X, lens, M)
            [E, lens, keyMask, valid] = obj.embed(X, lens, M);
            if size(E, 3) == 0
                H = E;
                return;
            end
            H = obj.Transformer.forward(E, 'KeyMask', keyMask);
            H = OceanDL.Layout.SequenceMask.apply(H, valid);
        end
    end

    methods (Static, Access = private)
        function P = positions(T, D)
            persistent cache
            if isempty(cache) || size(cache, 1) ~= D || size(cache, 3) < T
                cache = OceanDL.Transformer.PositionalEncoding.sincos1d(max(T, 512), D);
            end
            P = cache(:, :, 1:T);
        end
    end
end

function v = valOrNaN(L)
if isempty(L)
    v = NaN;
else
    v = double(OceanDL.Autograd.Tensor.raw(L));
end
end
