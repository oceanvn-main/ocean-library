classdef EncoderDecoderRecognizer < OceanNLP.Speech.Models.ISpeechRecognizer
    % ENCODERDECODERRECOGNIZER  Attention encoder-decoder ASR ("Listen, Attend and Spell"
    %   family; Chan et al. 2016) with a Transformer decoder (Vaswani et al. 2017), optional
    %   joint CTC auxiliary objective (Kim, Hori & Watanabe 2017), trainable end to end.
    %
    %   m   = OceanNLP.Speech.Models.EncoderDecoderRecognizer(features, vocab, encoders, ...
    %           'DecoderDim', [], 'DecoderLayers', 2, 'DecoderHeads', 4, 'DecoderFF', [], ...
    %           'Activation', "relu", 'Dropout', 0.1, 'LabelSmoothing', 0, 'CTCWeight', 0, ...
    %           'InputNorm', "utterance", 'Seed', 0, 'ModelId', "")
    %   out = m.forward(batch)       teacher forcing (batch needs Targets)
    %         out.Logits      K x N x U Tensor, U = max_n |y_n| + 1 decoder steps
    %         out.Lengths     1 x N ENCODER output lengths T'(n) (memory valid frames)
    %         out.LogitLengths 1 x N valid decoder steps |y_n| + 1;  out.TargetMask N x U
    %         out.Targets     N x U ids (y_n then Eos, Pad after);  out.EncoderOutput
    %         out.CTCLogits   K x N x T' (when CTCWeight > 0)
    %         out.CrossAttention  1 x layers cell of (U x T' x h x N) (when StoreAttention)
    %   L   = m.loss(batch);   s = m.losses(batch)  (Total, Attention, CTC Tensors)
    %   [tr, info] = m.transcribe(input, 'Decoder', "greedy"|"beam", 'BeamWidth', 8, ...
    %           'NBest', [], 'LengthPenalty', 1, 'MaxLength', 200, 'MaxLengthRatio', 1.5, ...
    %           'ReturnAttention', false)
    %   cfg = m.config();   m2 = RecognizerFactory.fromConfig(cfg)
    %
    %   COMPUTATION for utterance n (features x_1..x_T, target units y_1..y_L from
    %   Vocabulary.encode):
    %     CMVN (InputNorm "utterance", valid frames only) as in CTCRecognizer;
    %     [h, T'] = E_J(...E_1(x~, T))            encoder chain (lengths propagated)
    %     m_t = P h_t                             (P = identity unless DecoderDim ~= encoder dim)
    %     decoder input  d = (Bos, y_1, ..., y_L),  target  z = (y_1, ..., y_L, Eos)
    %     e_u = Emb(d_u) + PE(u - 1)              embedding (N(0,1) init) + sinusoidal position
    %     g   = DecoderBlock_D(...DecoderBlock_1(Drop(e); memory m, key mask t <= T'))
    %     o_u = W LN(g_u) + b                     K logits per decoder step
    %   ATTENTION LOSS (masked sequence cross entropy, token-level mean over the batch,
    %   Pad positions excluded; label smoothing eps spreads eps uniformly over all K
    %   classes, PyTorch convention):
    %     L_att = -(1 / sum_n (L_n + 1)) sum_n sum_{u <= L_n + 1}
    %               [(1 - eps) log p(z_u | z_<u, x) + eps mean_k log p_k]
    %   JOINT CTC (CTCWeight lambda > 0): a linear CTC head on the encoder output,
    %     L = (1 - lambda) L_att + lambda L_ctc,  L_ctc = OceanDL CTC "mean" (per-utterance
    %     loss / target length, averaged). CTCZeroInfinity (default true, PyTorch zero_infinity /
    %     ESPnet convention): an utterance whose encoder output is too short to emit its
    %     labels under CTC (T' < |y| + repeats, e.g. a very short clip after 4x subsampling)
    %     contributes 0 CTC loss and gradient instead of aborting training; its attention
    %     loss is unaffected. false = raise OceanDL's ':Infeasible' error.  The CTC head only shapes training (monotonic
    %     alignment prior); transcribe decodes with the attention decoder only.
    %   PADDING: encoder frames > T'(n) are masked keys of the cross-attention; decoder
    %   positions > L_n + 1 are excluded from the loss; the causal mask makes position u
    %   depend on d_1..d_u only. Valid logits and the loss are therefore those of each
    %   utterance alone (tested).
    %
    %   DECODING: Decoding.AttentionBeamDecoder against the encoder memory computed once.
    %   'UseCache' true (default): INCREMENTAL step with a key/value cache -- per decoder
    %   layer the cross-attention keys/values of the memory are computed once per batch
    %   (DecoderBlock.memoryKeyValues) and the self-attention keys/values of the prefix
    %   are cached per live hypothesis, reordered by the decoder's parent indices (beam
    %   reordering / pruning), so each step runs only the new position (O(u) per step).
    %   'UseCache' false: the reference step that re-runs the whole prefix (O(u^2) per
    %   step); both give the same log-probabilities to ~1e-12 and identical transcripts
    %   (tested). Max length per utterance = min(MaxLength,
    %   ceil(MaxLengthRatio * T'(n))). 'ReturnAttention' re-runs the teacher-forced
    %   decoder on (Bos, hypothesis) -- by causality identical to the attention seen
    %   during search -- and returns info(n).CrossAttention = (L+1) x T'(n) x h weights of
    %   the LAST decoder layer. Transcripts carry no Segments (attention is not timing).
    %   JOINT CTC N-BEST RESCORING ('CTCRescoreWeight' lambda > 0, beam only, needs the CTC
    %   head): every N-best attention hypothesis y is re-ranked by
    %     logP_joint(y) = (1 - lambda) logP_att(y) + lambda logP_ctc(y | x),
    %     Score = logP_joint / (|y| + 1)^LengthPenalty,
    %   logP_ctc = -CTC loss of y on the CTC head (sum over all alignments; -Inf when y is
    %   infeasible for T'). This is N-best rescoring (Kim, Hori & Watanabe 2017 use the CTC
    %   prefix score inside the beam; rescoring only reorders what the attention beam found).
    %   Candidates then also carry AttentionLogProb and CTCLogProb; Decoder
    %   "attention-beam+ctc-rescore".

    properties (SetAccess = protected)
        Encoders cell = {}
        MemoryProj = []
        Embed
        Blocks cell = {}
        FinalNorm
        Head
        CTCHead = []
        Drop
        InputDim (1,1) double
        InputNorm (1,1) string
        DecoderDim (1,1) double
        DecoderLayers (1,1) double
        DecoderHeads (1,1) double
        DecoderFF (1,1) double
        Activation (1,1) string
        DropoutP (1,1) double
        LabelSmoothing (1,1) double
        CTCWeight (1,1) double
        CTCZeroInfinity (1,1) logical
        Seed (1,1) double
    end

    properties
        StoreAttention (1,1) logical = false
    end

    methods
        function obj = EncoderDecoderRecognizer(features, vocabulary, encoders, opts)
            arguments
                features (1,1) OceanNLP.Speech.Core.IFeatureTransform
                vocabulary (1,1) OceanNLP.Speech.Data.SpeechVocabulary
                encoders {mustBeNonempty}
                opts.DecoderDim double {mustBeScalarOrEmpty} = []
                opts.DecoderLayers (1,1) double {mustBePositive, mustBeInteger} = 2
                opts.DecoderHeads (1,1) double {mustBePositive, mustBeInteger} = 4
                opts.DecoderFF double {mustBeScalarOrEmpty} = []
                opts.Activation (1,1) string {mustBeMember(opts.Activation, ["relu", "gelu"])} = "relu"
                opts.Dropout (1,1) double {mustBeGreaterThanOrEqual(opts.Dropout, 0), mustBeLessThan(opts.Dropout, 1)} = 0.1
                opts.LabelSmoothing (1,1) double {mustBeGreaterThanOrEqual(opts.LabelSmoothing, 0), mustBeLessThan(opts.LabelSmoothing, 1)} = 0
                opts.CTCWeight (1,1) double {mustBeGreaterThanOrEqual(opts.CTCWeight, 0), mustBeLessThan(opts.CTCWeight, 1)} = 0
                opts.CTCZeroInfinity (1,1) logical = true
                opts.InputNorm (1,1) string {mustBeMember(opts.InputNorm, ["utterance", "none"])} = "utterance"
                opts.Seed (1,1) double {mustBeInteger, mustBeNonnegative} = 0
                opts.ModelId (1,1) string = ""
            end
            if ~iscell(encoders), encoders = {encoders}; end
            encoders = reshape(encoders, 1, []);
            fdim = features.outputDim([]);
            d = fdim;
            for j = 1:numel(encoders)
                e = encoders{j};
                if ~isa(e, 'OceanDL.nn.Module') || ~ismethod(e, 'outputLength') || ~isprop(e, 'OutputDim') ...
                        || ~isprop(e, 'InputDim') || ~ismethod(e, 'config')
                    error('OceanNLP:Speech:Models:EncoderDecoderRecognizer:EncoderContract', ...
                        'Encoder %d (%s) must be an OceanDL.nn.Module with InputDim, OutputDim, forward(X, lens), outputLength, config.', j, class(e));
                end
                if e.InputDim ~= d
                    error('OceanNLP:Speech:Models:EncoderDecoderRecognizer:DimMismatch', ...
                        'Encoder %d expects %d inputs but receives %d.', j, e.InputDim, d);
                end
                d = e.OutputDim;
            end
            D = opts.DecoderDim;
            if isempty(D), D = d; end
            if mod(D, 2) || mod(D, opts.DecoderHeads)
                error('OceanNLP:Speech:Models:EncoderDecoderRecognizer:DecoderDim', ...
                    'DecoderDim %d must be even and divisible by DecoderHeads %d.', D, opts.DecoderHeads);
            end
            ff = opts.DecoderFF;
            if isempty(ff), ff = 4 * D; end
            K = vocabulary.numClasses();
            obj.Features = features;
            obj.Vocabulary = vocabulary;
            obj.ModelId = opts.ModelId;
            obj.Encoders = encoders;
            obj.InputDim = fdim;
            obj.InputNorm = opts.InputNorm;
            obj.DecoderDim = D;
            obj.DecoderLayers = opts.DecoderLayers;
            obj.DecoderHeads = opts.DecoderHeads;
            obj.DecoderFF = ff;
            obj.Activation = opts.Activation;
            obj.DropoutP = opts.Dropout;
            obj.LabelSmoothing = opts.LabelSmoothing;
            obj.CTCWeight = opts.CTCWeight;
            obj.CTCZeroInfinity = opts.CTCZeroInfinity;
            obj.Seed = opts.Seed;
            s = opts.Seed;
            if D ~= d
                obj.MemoryProj = OceanDL.nn.Linear(d, D, 'Seed', s + 1);
            end
            obj.Embed = OceanDL.nn.Embedding(K, D, 'Seed', s + 2);
            obj.Blocks = cell(1, opts.DecoderLayers);
            for l = 1:opts.DecoderLayers
                obj.Blocks{l} = OceanDL.Transformer.DecoderBlock(D, opts.DecoderHeads, ff, 'Activation', opts.Activation, ...
                    'Dropout', opts.Dropout, 'Seed', s + 100 * l);
            end
            obj.FinalNorm = OceanDL.nn.LayerNorm(D);
            obj.Head = OceanDL.nn.Linear(D, K, 'Seed', s + 3);
            if opts.CTCWeight > 0
                obj.CTCHead = OceanDL.nn.Linear(d, K, 'Seed', s + 4);
            end
            obj.Drop = OceanDL.nn.Dropout(opts.Dropout);
            obj.Vocabulary.assertCompatible(obj.Head.OutFeatures);
        end

        function out = forward(obj, batch)
            arguments
                obj
                batch (1,1) struct
            end
            if ~isfield(batch, 'Targets')
                error('OceanNLP:Speech:Models:EncoderDecoderRecognizer:NoTargets', ...
                    'forward / loss need batch.Targets (teacher forcing); use transcribe for inference.');
            end
            [H, lens] = obj.encode(batch);
            [din, tgt, tmask] = obj.teacherForcing(batch.Targets);
            [Z, xatt] = obj.decodeLogits(din, obj.memory(H), lens);
            out = struct('Logits', Z, 'Lengths', lens, 'LogitLengths', sum(tmask, 2).', 'TargetMask', tmask, ...
                'Targets', tgt, 'EncoderOutput', H);
            if ~isempty(obj.CTCHead)
                C = obj.CTCHead.forward(H);
                out.CTCLogits = OceanDL.Layout.SequenceMask.apply(C, OceanDL.Layout.SequenceMask.fromLengths(lens, size(C, 3)));
            end
            if obj.StoreAttention
                out.CrossAttention = xatt;
            end
        end

        function s = losses(obj, batch)
            % LOSSES  struct(Total, Attention, CTC) of scalar Tensors (CTC = [] when off).
            arguments
                obj
                batch (1,1) struct
            end
            out = obj.forward(batch);
            Latt = OceanDL.Autograd.F.crossEntropy(out.Logits, out.Targets, 'Mask', out.TargetMask, ...
                'LabelSmoothing', obj.LabelSmoothing);
            s = struct('Total', Latt, 'Attention', Latt, 'CTC', []);
            if obj.CTCWeight > 0
                Lc = OceanDL.Autograd.F.ctcLoss(out.CTCLogits, reshape(batch.Targets, 1, []), 'InputLengths', out.Lengths, ...
                    'Blank', obj.Vocabulary.Blank, 'Reduction', "mean", 'ZeroInfinity', obj.CTCZeroInfinity);
                s.CTC = Lc;
                s.Total = (1 - obj.CTCWeight) * Latt + obj.CTCWeight * Lc;
            end
        end

        function L = loss(obj, batch)
            arguments
                obj
                batch (1,1) struct
            end
            s = obj.losses(batch);
            L = s.Total;
        end

        function [tr, info] = transcribe(obj, input, opts)
            arguments
                obj
                input
                opts.Decoder (1,1) string {mustBeMember(opts.Decoder, ["greedy", "beam"])} = "greedy"
                opts.BeamWidth (1,1) double {mustBePositive, mustBeInteger} = 8
                opts.NBest double {mustBeScalarOrEmpty} = []
                opts.LengthPenalty (1,1) double {mustBeNonnegative} = 1
                opts.MaxLength (1,1) double {mustBePositive, mustBeInteger} = 200
                opts.MaxLengthRatio (1,1) double {mustBePositive} = 1.5
                opts.BatchSize (1,1) double {mustBePositive, mustBeInteger} = 32
                opts.ReturnAttention (1,1) logical = false
                opts.UseCache (1,1) logical = true
                opts.CTCRescoreWeight (1,1) double {mustBeGreaterThanOrEqual(opts.CTCRescoreWeight, 0), mustBeLessThanOrEqual(opts.CTCRescoreWeight, 1)} = 0
            end
            obj.Vocabulary.assertCompatible(obj.Head.OutFeatures);
            if opts.CTCRescoreWeight > 0 && (isempty(obj.CTCHead) || opts.Decoder ~= "beam")
                error('OceanNLP:Speech:Models:EncoderDecoderRecognizer:CTCRescore', ...
                    'CTCRescoreWeight > 0 needs a model with a CTC head (CTCWeight > 0) and Decoder "beam".');
            end
            dec = OceanNLP.Speech.Decoding.AttentionBeamDecoder(obj.Vocabulary, 'Method', opts.Decoder, ...
                'BeamWidth', opts.BeamWidth, 'NBest', opts.NBest, 'LengthPenalty', opts.LengthPenalty, ...
                'MaxLength', opts.MaxLength, 'MaxLengthRatio', opts.MaxLengthRatio, 'ModelId', obj.ModelId);
            wasTraining = obj.Training;
            obj.eval();
            restore = onCleanup(@() obj.train(wasTraining));
            grad = OceanDL.Autograd.GradMode.disable(); %#ok<NASGU>
            if isstruct(input)
                batches = {input};
            else
                seqs = obj.toFeatureSequences(input);
                batches = {};
                for s = 1:opts.BatchSize:numel(seqs)
                    batches{end+1} = OceanNLP.Speech.Models.CTCRecognizer.padSequences( ...
                        seqs(s:min(s + opts.BatchSize - 1, numel(seqs)))); %#ok<AGROW>
                end
            end
            tr = OceanNLP.Speech.Core.Transcript.empty(1, 0);
            info = struct('CrossAttention', {}, 'EncoderLength', {});
            for b = 1:numel(batches)
                [H, lens] = obj.encode(batches{b});
                Mem = OceanDL.Autograd.Tensor.raw(obj.memory(H));
                if opts.UseCache
                    step = @(prefixes, src, st, parent) obj.nextLogProbsCached(prefixes, src, st, parent);
                    t = dec.decode(step, lens, 'Incremental', true, 'State', obj.initDecodeState(Mem, lens));
                else
                    step = @(prefixes, src) obj.nextLogProbs(prefixes, Mem(:, src, :), lens(src));
                    t = dec.decode(step, lens);
                end
                if opts.CTCRescoreWeight > 0
                    t = obj.ctcRescore(t, H, lens, opts.CTCRescoreWeight, opts.LengthPenalty);
                end
                tr = [tr, t]; %#ok<AGROW>
                if nargout > 1
                    for n = 1:numel(t)
                        a = [];
                        if opts.ReturnAttention
                            a = obj.crossAttention(Mem(:, n, 1:max(lens(n), 1)), lens(n), t(n).TokenIds);
                        end
                        info(end + 1) = struct('CrossAttention', a, 'EncoderLength', lens(n)); %#ok<AGROW>
                    end
                end
            end
            clear restore
        end

        function lens = outputLength(obj, lens)
            % OUTPUTLENGTH  Encoder (memory) frames for feature-frame counts LENS.
            for j = 1:numel(obj.Encoders)
                lens = obj.Encoders{j}.outputLength(lens);
            end
        end

        function cfg = config(obj)
            enc = cellfun(@(e) e.config(), obj.Encoders, 'UniformOutput', false);
            cfg = struct('Type', "EncoderDecoderRecognizer", 'Version', 1, 'InputDim', obj.InputDim, ...
                'InputNorm', obj.InputNorm, 'DecoderDim', obj.DecoderDim, 'DecoderLayers', obj.DecoderLayers, ...
                'DecoderHeads', obj.DecoderHeads, 'DecoderFF', obj.DecoderFF, 'Activation', obj.Activation, ...
                'Dropout', obj.DropoutP, 'LabelSmoothing', obj.LabelSmoothing, 'CTCWeight', obj.CTCWeight, 'CTCZeroInfinity', obj.CTCZeroInfinity, ...
                'Seed', obj.Seed, 'NumClasses', obj.Head.OutFeatures, 'Encoders', {enc}, ...
                'Vocabulary', obj.Vocabulary.stateDict(), 'Features', obj.Features.config(), 'ModelId', obj.ModelId);
        end

        function st = initDecodeState(obj, Mem, lens)
            % INITDECODESTATE  Incremental-decoding state for memory Mem (D x N x T, raw) with
            %   valid lengths lens: per layer cross-attention keys/values (once per batch) and
            %   empty self-attention caches for the N first-step rows (row n = utterance n).
            Mem = OceanDL.Autograd.Tensor.raw(Mem);
            [~, N, T] = size(Mem, 1:3);
            L = numel(obj.Blocks);
            dk = obj.DecoderDim / obj.DecoderHeads;
            st = struct('MemKV', {cell(1, L)}, 'Self', {cell(1, L)}, ...
                'MemMask', OceanDL.Layout.SequenceMask.fromLengths(lens, T));
            for l = 1:L
                st.MemKV{l} = obj.Blocks{l}.memoryKeyValues(Mem);
                st.Self{l} = OceanDL.Transformer.DecoderBlock.emptyCache(dk, obj.DecoderHeads, N, Mem);
            end
        end

        function [lp, st] = nextLogProbsCached(obj, prefixes, src, st, parent)
            % NEXTLOGPROBSCACHED  Incremental step function (AttentionBeamDecoder protocol):
            %   prefixes (B x u, Bos first), src (B) utterance of each row, st from
            %   initDecodeState / the previous step, parent (B) previous row each row extends
            %   -> K x B log-probabilities of token u+1 and the updated state. Only position u
            %   (token prefixes(:, end)) is computed; earlier positions come from the cache.
            u = size(prefixes, 2);
            B = size(prefixes, 1);
            if size(st.Self{1}.K, 2) ~= u - 1
                error('OceanNLP:Speech:Models:EncoderDecoderRecognizer:CacheLength', ...
                    'Cache holds %d positions but the prefixes have %d tokens.', size(st.Self{1}.K, 2), u);
            end
            parent = reshape(parent, [], 1);
            src = reshape(src, [], 1);
            e = reshape(OceanDL.Autograd.Tensor.raw(obj.Embed.forward(prefixes(:, end))), obj.DecoderDim, B, 1);
            pe = OceanDL.Transformer.PositionalEncoding.sincos1d(u, obj.DecoderDim);
            x = e + pe(:, :, u);
            mm = st.MemMask(src, :);
            for l = 1:numel(obj.Blocks)
                kv = struct('K', st.Self{l}.K(:, :, :, parent), 'V', st.Self{l}.V(:, :, :, parent));
                mkv = struct('K', st.MemKV{l}.K(:, :, :, src), 'V', st.MemKV{l}.V(:, :, :, src));
                [x, st.Self{l}] = obj.Blocks{l}.forwardStep(x, kv, mkv, 'MemoryKeyMask', mm);
            end
            z = OceanDL.Autograd.Tensor.raw(obj.Head.forward(OceanDL.Autograd.Tensor.raw(obj.FinalNorm.forward(x))));
            z = reshape(z, size(z, 1), B);
            m = max(z, [], 1);
            lp = z - (m + log(sum(exp(z - m), 1)));
        end

        function lp = nextLogProbs(obj, prefixes, mem, lens)
            % NEXTLOGPROBS  Step function core: prefixes (B x u ids, Bos first), mem (D x B x T)
            %   memory rows, lens (B) valid frames -> K x B log-probabilities of the next token.
            [Z, ~] = obj.decodeLogits(prefixes, mem, lens);
            z = OceanDL.Autograd.Tensor.raw(Z);
            z = reshape(z(:, :, end), size(z, 1), []);
            m = max(z, [], 1);
            lp = z - (m + log(sum(exp(z - m), 1)));
        end
    end

    methods (Static)
        function obj = fromConfig(cfg)
            arguments
                cfg (1,1) struct
            end
            if string(cfg.Type) ~= "EncoderDecoderRecognizer"
                error('OceanNLP:Speech:Models:EncoderDecoderRecognizer:ConfigType', ...
                    'Config Type is "%s", not "EncoderDecoderRecognizer".', string(cfg.Type));
            end
            features = OceanNLP.Speech.Models.RecognizerFactory.featuresFromConfig(cfg.Features);
            vocab = OceanNLP.Speech.Data.SpeechVocabulary.fromStateDict(cfg.Vocabulary);
            encs = cfg.Encoders;
            if isstruct(encs), encs = num2cell(encs); end
            encoders = cellfun(@(c) OceanNLP.Speech.Models.RecognizerFactory.encoderFromConfig(c), reshape(encs, 1, []), 'UniformOutput', false);
            obj = OceanNLP.Speech.Models.EncoderDecoderRecognizer(features, vocab, encoders, 'DecoderDim', cfg.DecoderDim, ...
                'DecoderLayers', cfg.DecoderLayers, 'DecoderHeads', cfg.DecoderHeads, 'DecoderFF', cfg.DecoderFF, ...
                'Activation', cfg.Activation, 'Dropout', cfg.Dropout, 'LabelSmoothing', cfg.LabelSmoothing, ...
                'CTCWeight', cfg.CTCWeight, 'CTCZeroInfinity', cfg.CTCZeroInfinity, 'InputNorm', cfg.InputNorm, 'Seed', cfg.Seed, 'ModelId', cfg.ModelId);
            if obj.Head.OutFeatures ~= cfg.NumClasses
                error('OceanNLP:Speech:Models:EncoderDecoderRecognizer:ConfigNumClasses', ...
                    'Config says %d classes, vocabulary gives %d.', cfg.NumClasses, obj.Head.OutFeatures);
            end
        end
    end

    methods (Access = protected)
        function [H, lens] = encode(obj, batch)
            if ~isfield(batch, 'X') || ~isfield(batch, 'Lengths')
                error('OceanNLP:Speech:Models:EncoderDecoderRecognizer:Batch', 'batch needs fields X (F x N x T) and Lengths (1 x N).');
            end
            X = double(batch.X);
            lens = double(reshape(batch.Lengths, 1, []));
            [F, N, T] = size(X, 1:3);
            if F ~= obj.InputDim
                error('OceanNLP:Speech:Models:EncoderDecoderRecognizer:InputDim', ...
                    'Model expects %d feature bins, batch has %d.', obj.InputDim, F);
            end
            if numel(lens) ~= N || any(lens < 0 | lens > T | lens ~= round(lens))
                error('OceanNLP:Speech:Models:EncoderDecoderRecognizer:Lengths', 'Lengths must be N = %d integers in [0, %d].', N, T);
            end
            if obj.InputNorm == "utterance"
                for n = 1:N
                    if lens(n) == 0, continue; end
                    v = X(:, n, 1:lens(n));
                    mu = mean(v, 3);
                    s2 = mean((v - mu).^2, 3);
                    X(:, n, 1:lens(n)) = (v - mu) ./ sqrt(s2 + 1e-8);
                end
                X = X .* reshape((1:T) <= lens(:), 1, N, T);
            end
            X = obj.padToMinimumFrames(X);
            H = X;
            for j = 1:numel(obj.Encoders)
                [H, lens] = obj.Encoders{j}.forward(H, lens);
            end
        end

        function M = memory(obj, H)
            M = H;
            if ~isempty(obj.MemoryProj)
                M = obj.MemoryProj.forward(H);
            end
        end

        function [din, tgt, mask] = teacherForcing(obj, targets)
            % Bos-prefixed inputs, Eos-terminated targets, Pad elsewhere (N x U).
            v = obj.Vocabulary;
            targets = reshape(targets, 1, []);
            N = numel(targets);
            L = cellfun(@numel, targets);
            U = max(L) + 1;
            din = v.Pad * ones(N, U);
            tgt = v.Pad * ones(N, U);
            for n = 1:N
                y = double(reshape(targets{n}, 1, []));
                if any(y <= v.Unk & y ~= v.Unk)
                    error('OceanNLP:Speech:Models:EncoderDecoderRecognizer:SpecialInTarget', ...
                        'Target %d contains a special token (Blank/Pad/Bos/Eos) -- targets are unit ids only.', n);
                end
                din(n, 1:L(n) + 1) = [v.Bos, y];
                tgt(n, 1:L(n) + 1) = [y, v.Eos];
            end
            mask = (1:U) <= (L(:) + 1);
        end

        function [Z, xatt] = decodeLogits(obj, din, mem, lens)
            % din (N x U) ids, mem (D x N x T) -> logits K x N x U.
            [N, U] = size(din);
            T = size(mem, 3);
            e = obj.Embed.forward(din);                                  % D x N x U
            x = obj.Drop.forward(e + OceanDL.Transformer.PositionalEncoding.sincos1d(U, obj.DecoderDim));
            mm = OceanDL.Layout.SequenceMask.fromLengths(lens, T);
            xatt = cell(1, numel(obj.Blocks));
            for l = 1:numel(obj.Blocks)
                obj.Blocks{l}.StoreAttention = obj.StoreAttention;
                x = obj.Blocks{l}.forward(x, mem, 'MemoryKeyMask', mm);
                if obj.StoreAttention
                    xatt{l} = obj.Blocks{l}.LastCrossAttention;
                end
            end
            Z = obj.Head.forward(obj.FinalNorm.forward(x));
            assert(size(Z, 2) == N);
        end

        function t = ctcRescore(obj, t, H, lens, lambda, alpha)
            % N-best rescoring with the joint CTC head (see class help).
            C = OceanDL.Autograd.Tensor.raw(obj.CTCHead.forward(H));          % K x N x T'
            blank = obj.Vocabulary.Blank;
            for n = 1:numel(t)
                c = t(n).Candidates;
                Cn = C(:, n, 1:max(lens(n), 1));
                for h = 1:numel(c)
                    lc = -Inf;
                    if lens(n) > 0
                        try
                            lc = -double(OceanDL.Autograd.Tensor.raw(OceanDL.Autograd.F.ctcLoss(Cn, {c(h).TokenIds}, ...
                                'InputLengths', lens(n), 'Blank', blank, 'Reduction', "sum")));
                        catch err
                            if ~strcmp(err.identifier, 'OceanDL:Autograd:CTCLoss:Infeasible'), rethrow(err); end
                        end
                    end
                    c(h).AttentionLogProb = c(h).LogProb;
                    c(h).CTCLogProb = lc;
                    c(h).LogProb = (1 - lambda) * c(h).AttentionLogProb + lambda * lc;
                    c(h).Score = c(h).LogProb / c(h).Length ^ alpha;
                end
                if ~any(isfinite([c.Score]))          % nothing CTC-feasible: keep the attention ranking
                    for h = 1:numel(c)
                        c(h).LogProb = c(h).AttentionLogProb;
                        c(h).Score = c(h).LogProb / c(h).Length ^ alpha;
                    end
                end
                [~, order] = sort([c.Score], 'descend');
                c = c(order);
                o = t(n).DecoderOptions;
                o.CTCRescoreWeight = lambda;
                t(n) = OceanNLP.Speech.Core.Transcript(c(1).Text, 'TokenIds', c(1).TokenIds, ...
                    'Decoder', "attention-beam+ctc-rescore", 'DecoderOptions', o, 'ModelId', t(n).ModelId, ...
                    'Score', c(1).Score, 'Candidates', c);
            end
        end

        function A = crossAttention(obj, mem, len, ids)
            % Last-layer cross-attention of the teacher-forced hypothesis (L+1) x T' x h.
            prev = obj.StoreAttention;
            obj.StoreAttention = true;
            c = onCleanup(@() restoreStore(obj, prev));
            din = [obj.Vocabulary.Bos, double(ids)];
            [~, xatt] = obj.decodeLogits(din, mem, len);
            A = xatt{end};
            A = A(:, 1:len, :, 1);
            clear c
        end

        function seqs = toFeatureSequences(obj, input)
            if iscell(input)
                items = input;
            else
                items = num2cell(input);
            end
            seqs = cell(1, numel(items));
            ref = stripHistory(obj.Features.config());
            for i = 1:numel(items)
                it = items{i};
                if isa(it, 'OceanNLP.Speech.Core.Waveform')
                    it = obj.Features.forward(it);
                elseif isa(it, 'OceanNLP.Speech.Core.FeatureSequence')
                    if it.numBins() ~= obj.InputDim
                        error('OceanNLP:Speech:Models:EncoderDecoderRecognizer:InputDim', ...
                            'Input %d has %d bins; the model expects %d.', i, it.numBins(), obj.InputDim);
                    end
                    if ~isempty(fieldnames(it.Config)) && ~isequaln(stripHistory(it.Config), ref)
                        error('OceanNLP:Speech:Models:EncoderDecoderRecognizer:FeatureMismatch', ...
                            'Input %d was computed with a different feature configuration than the model''s.', i);
                    end
                else
                    error('OceanNLP:Speech:Models:EncoderDecoderRecognizer:InputType', ...
                        'transcribe takes a Waveform, FeatureSequence, an array/cell of them, or a batch struct (got %s).', class(it));
                end
                if it.numBins() ~= obj.InputDim
                    error('OceanNLP:Speech:Models:EncoderDecoderRecognizer:InputDim', ...
                        'Input %d has %d bins; the model expects %d.', i, it.numBins(), obj.InputDim);
                end
                seqs{i} = it;
            end
        end
    end
end

function c = stripHistory(c)
if isfield(c, 'InputHistory')
    c = rmfield(c, 'InputHistory');
end
end

function restoreStore(obj, v)
obj.StoreAttention = v;
end
