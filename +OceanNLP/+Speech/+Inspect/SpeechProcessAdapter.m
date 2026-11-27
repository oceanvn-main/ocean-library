classdef SpeechProcessAdapter
    % SPEECHPROCESSADAPTER  Run a real speech recogniser stage by stage and capture every intermediate.
    %
    %   [tr, reader] = OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(recognizer, waveform)
    %   [tr, reader] = OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(recognizer, waveform, ...
    %                    'Trace', SpeechTrace(...), 'Decoder', "greedy"|"beam", 'BeamWidth', 8, ...
    %                    'Prune', -inf, 'NBest', [], 'LanguageModel', [], 'LMWeight', 0, 'Bonus', 0)
    %
    %   tr      1 x 1 OceanNLP.Speech.Core.Transcript -- identical (Text, TokenIds, Score,
    %           Candidates, Decoder, DecoderOptions, ModelId) to
    %           recognizer.transcribe(waveform, <same decoder options>) (tested).
    %   reader  OceanNLP.Inspect.MemoryReader over the trace (empty when the trace is
    %           disabled). Omitting 'Trace' uses a fresh SpeechTrace() with default budgets.
    %
    %   STAGES (CTCRecognizer; each is one trace step, Stage name in brackets, artifact
    %   values are plain structs whose fields are the data contracts of the
    %   OceanViz.Strategies.Speech.* strategies named on the right):
    %     [input]     Waveform           Samples, SampleRate, Source, History        SpeechWaveform
    %     [frontend]  Frames + window    Frames (L x T raw), Window, FrameStarts (0-based s_t), ...
    %     [frontend]  Power spectrum     Values = |DFT(w .* x_t)|^2, FrequencyHz, FrameTimes  Spectrogram
    %     [frontend]  Mel filter bank    Weights (M x K), BinFrequencyHz, CentersHz       MelFilterBank
    %     [frontend]  Log-mel            Values = ln(max(W P, floor)), FrequencyHz, ...   SpeechFeatures
    %     [features]  Model features     recognizer.Features.forward(waveform)            SpeechFeatures
    %     [model]     Normalised input   per-utterance CMVN (CTCRecognizer InputNorm)      SpeechFeatures
    %     [model]     Encoder j          [H, lens] = Encoders{j}.forward(H, lens)          SpeechFeatures
    %     [model]     Logits             u_t = W h_t + b (padding masked)                  SpeechFeatures
    %     [decode]    Posteriors         y_k(t) = softmax(u_t)_k, best path argmax         CTCPosterior
    %     [decode]    Beam candidates    decoder N-best (Text, TokenIds, Score)            BeamCandidates
    %     [decode]    Transcript         Text, TokenIds, Score, Decoder
    %   The four [frontend] stages come from the front end's own intermediates()
    %   method (LogMel; MFCC's LogMelTransform; the first stage of a FeaturePipeline
    %   when that is LogMel/MFCC) -- they are inspection captures and are skipped when
    %   the trace is disabled. Everything the transcript depends on is computed exactly
    %   as recognizer.transcribe computes it: features by recognizer.Features.forward,
    %   batch by CTCRecognizer.padSequences, then (eval mode, gradients off) the same
    %   CMVN, the model's own encoder / dropout / head / sequence-mask modules in the
    %   order of CTCRecognizer.forward, and the same CTCPrefixBeamDecoder. The only
    %   restated formula is the 6-line CMVN of CTCRecognizer.normalizeInput (protected);
    %   the test suite asserts the stage-by-stage logits equal recognizer.forward(batch)
    %   bit for bit, so any drift in CTCRecognizer is caught.
    %
    %   Logit frame times: when every encoder exposes its convolutions (a Convs cell of
    %   OceanDL.nn.Conv1d, e.g. ConvEncoder) or keeps the length unchanged
    %   (outputLength(T) = T, e.g. RecurrentEncoder), output frame t' is placed at
    %   the centre of its receptive field on the feature-frame grid:
    %     c' = (t' - 1) s - p + 1 + d (k - 1) / 2   (per Conv1d layer, composed),
    %     time(c) = c_1 + (c - 1) H / f_s           (FrameGrid centre times).
    %   Otherwise FrameTimes is [] and plots use the output-frame index.
    %
    %   INSPECTION NEVER CHANGES RESULTS: nothing here draws random numbers; the model's
    %   train/eval mode and the autograd mode are restored on exit (onCleanup).
    %
    %   ENCODER-DECODER BRANCH (Models.EncoderDecoderRecognizer), replaying
    %   EncoderDecoderRecognizer.transcribe; decode options as transcribe: 'Decoder',
    %   'BeamWidth', 'NBest', 'LengthPenalty' (1), 'MaxLength' (200), 'MaxLengthRatio' (1.5),
    %   'UseCache' (true), 'CTCRescoreWeight' (0). Stages after [features]:
    %     [model]   Normalised input   CMVN as EncoderDecoderRecognizer.encode   SpeechFeatures
    %     [model]   Encoder j          [H, lens] = Encoders{j}.forward(H, lens)   SpeechFeatures
    %     [model]   Decoder memory     m_t = P h_t (only when MemoryProj exists)  SpeechFeatures
    %     [decode]  CTC head posteriors softmax(CTCHead(h_t)) (only with a CTC head) CTCPosterior
    %     [decode]  Decoder search steps  every step-function call of the search:
    %               Steps(i).Prefixes (B x u ids, Bos first), Source, Parent (cache mode),
    %               LogProbs (K x B model log-probabilities of the next token, before the
    %               decoder masks Blank/Pad/Bos)
    %     [decode]  Decoder posteriors  teacher-forced p(z_u | z_<u, x) of the chosen
    %               hypothesis z = (y, Eos): Posteriors K x U, TokenIds, StepLabels,
    %               LogProb = sum_u ln p(z_u | ...) (= the attention log-prob of the search
    %               by causality; tested)                                     SpeechFeatures
    %     [decode]  Cross-attention    last decoder layer, Values U x T' (mean over heads),
    %               PerHead U x T' x h, FrameTimes of the encoder frames          SpeechFeatures
    %     [decode]  Beam candidates    decoder N-best (Text, TokenIds, Score, LogProb, ...)
    %     [decode]  Transcript
    %   The search runs the model's own public step functions (nextLogProbsCached with
    %   initDecodeState, or nextLogProbs when UseCache = false) inside the model's own
    %   Decoding.AttentionBeamDecoder; the recorded step wrapper only copies what they
    %   return. Restated (protected in the model): the CMVN + padToMinimumFrames of encode,
    %   the optional memory projection and the 25-line joint-CTC N-best rescoring; the
    %   posteriors / cross-attention come from the PUBLIC teacher-forced
    %   recognizer.forward(batch with Targets = {chosen ids}) with StoreAttention on (an
    %   inspection-only extra pass, skipped when the trace is disabled). Tests assert the
    %   transcript equals transcribe and the encoder output equals forward bit for bit.
    %   Options that do not apply to the recogniser type (e.g. LMWeight for an
    %   encoder-decoder, LengthPenalty for CTC) must stay at their defaults
    %   ('OceanNLP:Speech:Inspect:SpeechProcessAdapter:OptionNotApplicable');
    %   'BatchSize' and 'ReturnAttention' are accepted and ignored (one utterance; the
    %   attention is always recorded when the trace is on).
    %   Any other recogniser raises
    %   'OceanNLP:Speech:Inspect:SpeechProcessAdapter:UnsupportedRecognizer'.

    methods (Static)
        function [tr, reader] = run(recognizer, waveform, opts)
            arguments
                recognizer (1,1) OceanNLP.Speech.Models.ISpeechRecognizer
                waveform (1,1) OceanNLP.Speech.Core.Waveform
                opts.Trace = []
                opts.Decoder (1,1) string {mustBeMember(opts.Decoder, ["greedy", "beam"])} = "greedy"
                opts.BeamWidth (1,1) double {mustBePositive, mustBeInteger} = 8
                opts.Prune (1,1) double = -inf
                opts.NBest double {mustBeScalarOrEmpty} = []
                opts.LanguageModel = []
                opts.LMWeight (1,1) double = 0
                opts.Bonus (1,1) double = 0
                opts.LengthPenalty (1,1) double {mustBeNonnegative} = 1
                opts.MaxLength (1,1) double {mustBePositive, mustBeInteger} = 200
                opts.MaxLengthRatio (1,1) double {mustBePositive} = 1.5
                opts.UseCache (1,1) logical = true
                opts.CTCRescoreWeight (1,1) double {mustBeGreaterThanOrEqual(opts.CTCRescoreWeight, 0), mustBeLessThanOrEqual(opts.CTCRescoreWeight, 1)} = 0
                opts.BatchSize (1,1) double {mustBePositive, mustBeInteger} = 32
                opts.ReturnAttention (1,1) logical = false
            end
            trace = opts.Trace;
            if isempty(trace)
                trace = OceanNLP.Speech.Inspect.SpeechTrace();
            elseif ~isa(trace, 'OceanNLP.Speech.Inspect.SpeechTrace')
                error('OceanNLP:Speech:Inspect:SpeechProcessAdapter:TraceType', ...
                    'Trace must be an OceanNLP.Speech.Inspect.SpeechTrace (got %s).', class(trace));
            end
            isCTC = isa(recognizer, 'OceanNLP.Speech.Models.CTCRecognizer');
            isED = isa(recognizer, 'OceanNLP.Speech.Models.EncoderDecoderRecognizer');
            if ~isCTC && ~isED
                error('OceanNLP:Speech:Inspect:SpeechProcessAdapter:UnsupportedRecognizer', ...
                    'No stage-by-stage replay for %s (supported: CTCRecognizer, EncoderDecoderRecognizer).', ...
                    class(recognizer));
            end
            decOpts = rmfield(opts, 'Trace');
            if isCTC
                foreign = struct('LengthPenalty', 1, 'MaxLength', 200, 'MaxLengthRatio', 1.5, ...
                    'UseCache', true, 'CTCRescoreWeight', 0, 'ReturnAttention', false);
            else
                foreign = struct('Prune', -inf, 'LanguageModel', [], 'LMWeight', 0, 'Bonus', 0);
            end
            for f = string(fieldnames(foreign))'
                if ~isequal(decOpts.(f), foreign.(f))
                    error('OceanNLP:Speech:Inspect:SpeechProcessAdapter:OptionNotApplicable', ...
                        'Option %s does not apply to %s.', f, class(recognizer));
                end
            end

            % ---- [input] ----------------------------------------------------------
            waveId = trace.record("Waveform", "input", struct('Samples', waveform.Samples, ...
                'SampleRate', waveform.SampleRate, 'Source', waveform.Source, ...
                'History', waveform.History), strings(1, 0), ...
                'Summary', sprintf("%d samples x %d channel(s) at %g Hz (%.3f s)", ...
                waveform.numSamples(), waveform.numChannels(), waveform.SampleRate, waveform.duration()));

            % ---- [frontend] inspection captures (skipped when disabled) -----------
            frontId = waveId;
            if trace.Enabled
                frontId = OceanNLP.Speech.Inspect.SpeechProcessAdapter.recordFrontEnd_( ...
                    recognizer.Features, waveform, waveId, trace);
            end

            % ---- [features] exactly what transcribe computes ----------------------
            feats = recognizer.Features.forward(waveform);
            featId = trace.record("Model features", "features", ...
                OceanNLP.Speech.Inspect.SpeechProcessAdapter.featureData_(feats, ...
                "Model input features (" + string(feats.Config.Type) + ")"), frontId, ...
                'Summary', "recognizer.Features.forward(waveform): the features the model consumes");

            if isCTC
                tr = OceanNLP.Speech.Inspect.SpeechProcessAdapter.runCTC_(recognizer, feats, featId, trace, decOpts);
            else
                tr = OceanNLP.Speech.Inspect.SpeechProcessAdapter.runEncoderDecoder_(recognizer, feats, featId, trace, decOpts);
            end
            reader = trace.reader();
        end

        function times = outputFrameTimes(encoders, featTimes, hopSeconds, numOut)
            % OUTPUTFRAMETIMES  Receptive-field centre time of each of NUMOUT encoder output
            %   frames (see class help); [] when an encoder's time mapping is not known.
            arguments
                encoders (1,:) cell
                featTimes (1,:) double
                hopSeconds (1,1) double
                numOut (1,1) double {mustBeInteger, mustBeNonnegative}
            end
            c = (1:numOut);                       % output frame index on the last grid
            % walk the encoders backwards, mapping output index -> input-grid position
            for j = numel(encoders):-1:1
                e = encoders{j};
                if isprop(e, 'Convs') && iscell(e.Convs) && ~isempty(e.Convs) && ...
                        all(cellfun(@(cv) isa(cv, 'OceanDL.nn.Conv1d'), e.Convs))
                    for l = numel(e.Convs):-1:1
                        cv = e.Convs{l};
                        c = (c - 1) * cv.Stride - cv.Padding + 1 + cv.Dilation * (cv.KernelSize - 1) / 2;
                    end
                elseif isprop(e, 'Subsample') && iscell(e.Subsample) && ~isempty(e.Subsample) && ...
                        all(cellfun(@(cv) isa(cv, 'OceanDL.nn.Conv1d'), e.Subsample))
                    % conv subsampling followed by length-preserving blocks (ConformerEncoder)
                    for l = numel(e.Subsample):-1:1
                        cv = e.Subsample{l};
                        c = (c - 1) * cv.Stride - cv.Padding + 1 + cv.Dilation * (cv.KernelSize - 1) / 2;
                    end
                elseif ismethod(e, 'outputLength') && isequal(e.outputLength([1 7 64]), [1 7 64])
                    % length-preserving encoder: identity time mapping
                else
                    times = [];
                    return;
                end
            end
            if isempty(featTimes)
                times = [];
                return;
            end
            times = featTimes(1) + (c - 1) * hopSeconds;
        end
    end

    methods (Static, Access = private)
        function tr = runCTC_(m, feats, featId, trace, o)
            % Replays CTCRecognizer.transcribe -> forward -> CTCPrefixBeamDecoder.decode.
            m.Vocabulary.assertCompatible(m.Head.OutFeatures);
            dec = OceanNLP.Speech.Decoding.CTCPrefixBeamDecoder(m.Vocabulary, 'Method', o.Decoder, ...
                'BeamWidth', o.BeamWidth, 'Prune', o.Prune, 'NBest', o.NBest, ...
                'LanguageModel', o.LanguageModel, 'LMWeight', o.LMWeight, 'Bonus', o.Bonus, ...
                'ModelId', m.ModelId);
            wasTraining = m.Training;
            m.eval();
            restore = onCleanup(@() m.train(wasTraining));
            grad = OceanDL.Autograd.GradMode.disable(); %#ok<NASGU>

            batch = OceanNLP.Speech.Models.CTCRecognizer.padSequences({feats});
            X = double(batch.X);
            lens = double(reshape(batch.Lengths, 1, []));

            X = OceanNLP.Speech.Inspect.SpeechProcessAdapter.padToMinimumFrames_(m, X);
            % CMVN, CTCRecognizer.normalizeInput: x~ = (x - mu) ./ sqrt(sigma^2 + 1e-8)
            if m.InputNorm ~= "none"
                for n = 1:size(X, 2)
                    if lens(n) == 0, continue; end
                    v = X(:, n, 1:lens(n));
                    mu = mean(v, 3);
                    s2 = mean((v - mu).^2, 3);
                    X(:, n, 1:lens(n)) = (v - mu) ./ sqrt(s2 + 1e-8);
                end
                X(:, :, :) = X .* reshape((1:size(X, 3)) <= lens(:), 1, size(X, 2), size(X, 3));
            end
            hopSec = OceanNLP.Speech.Inspect.SpeechProcessAdapter.hopSeconds_(feats);
            prevId = trace.record("Normalised input", "model", struct('Values', reshape(X(:, 1, :), size(X, 1), []), ...
                'FrameTimes', feats.FrameTimes, 'FrequencyHz', feats.FrequencyHz, ...
                'AxisNames', feats.AxisNames, 'Units', "standardised (z-score)", ...
                'Title', "Normalised input (InputNorm = " + m.InputNorm + ")"), featId, ...
                'Summary', "Per-utterance mean/variance normalisation over the valid frames");

            H = X;
            for j = 1:numel(m.Encoders)
                [H, lens] = m.Encoders{j}.forward(H, lens);
                if trace.Enabled
                    Hr = double(gather(OceanDL.Autograd.Tensor.raw(H)));
                    t = OceanNLP.Speech.Inspect.SpeechProcessAdapter.outputFrameTimes( ...
                        m.Encoders(1:j), feats.FrameTimes, hopSec, size(Hr, 3));
                    name = sprintf("Encoder %d (%s)", j, OceanNLP.Speech.Inspect.SpeechProcessAdapter.shortClass_(m.Encoders{j}));
                    prevId = trace.record(name, "model", struct('Values', reshape(Hr(:, 1, :), size(Hr, 1), []), ...
                        'FrameTimes', t, 'FrequencyHz', [], 'AxisNames', ["hidden unit", "time"], ...
                        'Units', "activation", 'Lengths', lens, 'Title', name), prevId, ...
                        'Summary', sprintf("[H, lens] = Encoders{%d}.forward(H, lens); %d valid output frames", j, lens(1)));
                end
            end
            Z = m.Head.forward(m.Drop.forward(H));
            Z = OceanDL.Layout.SequenceMask.apply(Z, OceanDL.Layout.SequenceMask.fromLengths(lens, size(Z, 3)));

            Zr = double(gather(OceanDL.Autograd.Tensor.raw(Z)));
            K = size(Zr, 1);
            T1 = lens(1);
            logits = reshape(Zr(:, 1, 1:T1), K, T1);
            times = OceanNLP.Speech.Inspect.SpeechProcessAdapter.outputFrameTimes( ...
                m.Encoders, feats.FrameTimes, hopSec, T1);
            tokens = m.Vocabulary.Tokens;
            logitId = trace.record("Logits", "model", struct('Values', logits, 'FrameTimes', times, ...
                'FrequencyHz', [], 'AxisNames', ["class", "time"], 'Units', "logit (nats)", ...
                'Tokens', tokens, 'Title', "CTC logits"), prevId, ...
                'Summary', sprintf("Head: u_t = W h_t + b over %d classes x %d frames", K, T1));

            if trace.Enabled
                % posteriors y_k(t) = softmax(u_t)_k (stable); best path = argmax_k
                mx = max(logits, [], 1);
                ex = exp(logits - mx);
                post = ex ./ sum(ex, 1);
                [~, best] = max(logits, [], 1);
                postId = trace.record("Posteriors", "decode", struct('Posteriors', post, ...
                    'Tokens', tokens, 'FrameTimes', times, 'BestPath', best, ...
                    'Blank', m.Vocabulary.Blank), logitId, ...
                    'Summary', "Frame-wise class posteriors softmax(u_t) and the best path argmax_k");
            else
                postId = "";
            end

            tr = dec.decode(Z, lens);
            clear restore

            candId = trace.record("Beam candidates", "decode", struct('Candidates', tr.Candidates, ...
                'Method', dec.Method, 'BeamWidth', dec.BeamWidth, 'Chosen', 1, ...
                'ScoreLabel', OceanNLP.Speech.Inspect.SpeechProcessAdapter.scoreLabel_(dec)), ...
                [logitId, postId], 'Summary', sprintf("%s decoding: %d candidate(s)", tr.Decoder, numel(tr.Candidates)));
            trace.record("Transcript", "decode", struct('Text', tr.Text, 'TokenIds', tr.TokenIds, ...
                'Score', tr.Score, 'Decoder', tr.Decoder, 'ModelId', tr.ModelId), candId, ...
                'Summary', "Final transcript: """ + tr.Text + """");
        end

        function tr = runEncoderDecoder_(m, feats, featId, trace, o)
            % Replays EncoderDecoderRecognizer.transcribe -> encode -> AttentionBeamDecoder.
            m.Vocabulary.assertCompatible(m.Head.OutFeatures);
            if o.CTCRescoreWeight > 0 && (isempty(m.CTCHead) || o.Decoder ~= "beam")
                error('OceanNLP:Speech:Models:EncoderDecoderRecognizer:CTCRescore', ...
                    'CTCRescoreWeight > 0 needs a model with a CTC head (CTCWeight > 0) and Decoder "beam".');
            end
            dec = OceanNLP.Speech.Decoding.AttentionBeamDecoder(m.Vocabulary, 'Method', o.Decoder, ...
                'BeamWidth', o.BeamWidth, 'NBest', o.NBest, 'LengthPenalty', o.LengthPenalty, ...
                'MaxLength', o.MaxLength, 'MaxLengthRatio', o.MaxLengthRatio, 'ModelId', m.ModelId);
            wasTraining = m.Training;
            m.eval();
            restore = onCleanup(@() m.train(wasTraining));
            grad = OceanDL.Autograd.GradMode.disable(); %#ok<NASGU>

            batch = OceanNLP.Speech.Models.CTCRecognizer.padSequences({feats});
            X = double(batch.X);
            lens = double(reshape(batch.Lengths, 1, []));
            [~, N, T] = size(X, 1:3);
            % CMVN exactly as EncoderDecoderRecognizer.encode (protected there)
            if m.InputNorm == "utterance"
                for n = 1:N
                    if lens(n) == 0, continue; end
                    v = X(:, n, 1:lens(n));
                    mu = mean(v, 3);
                    s2 = mean((v - mu).^2, 3);
                    X(:, n, 1:lens(n)) = (v - mu) ./ sqrt(s2 + 1e-8);
                end
                X = X .* reshape((1:T) <= lens(:), 1, N, T);
            end
            X = OceanNLP.Speech.Inspect.SpeechProcessAdapter.padToMinimumFrames_(m, X);
            hopSec = OceanNLP.Speech.Inspect.SpeechProcessAdapter.hopSeconds_(feats);
            prevId = trace.record("Normalised input", "model", struct('Values', reshape(X(:, 1, :), size(X, 1), []), ...
                'FrameTimes', feats.FrameTimes, 'FrequencyHz', feats.FrequencyHz, ...
                'AxisNames', feats.AxisNames, 'Units', "standardised (z-score)", ...
                'Title', "Normalised input (InputNorm = " + m.InputNorm + ")"), featId, ...
                'Summary', "Per-utterance mean/variance normalisation over the valid frames");

            H = X;
            for j = 1:numel(m.Encoders)
                [H, lens] = m.Encoders{j}.forward(H, lens);
                if trace.Enabled
                    Hr = double(gather(OceanDL.Autograd.Tensor.raw(H)));
                    t = OceanNLP.Speech.Inspect.SpeechProcessAdapter.outputFrameTimes( ...
                        m.Encoders(1:j), feats.FrameTimes, hopSec, size(Hr, 3));
                    name = sprintf("Encoder %d (%s)", j, OceanNLP.Speech.Inspect.SpeechProcessAdapter.shortClass_(m.Encoders{j}));
                    prevId = trace.record(name, "model", struct('Values', reshape(Hr(:, 1, :), size(Hr, 1), []), ...
                        'FrameTimes', t, 'FrequencyHz', [], 'AxisNames', ["hidden unit", "time"], ...
                        'Units', "activation", 'Lengths', lens, 'Title', name), prevId, ...
                        'Summary', sprintf("[H, lens] = Encoders{%d}.forward(H, lens); %d valid output frames", j, lens(1)));
                end
            end
            T1 = lens(1);
            encTimes = OceanNLP.Speech.Inspect.SpeechProcessAdapter.outputFrameTimes( ...
                m.Encoders, feats.FrameTimes, hopSec, T1);
            M = H;
            if ~isempty(m.MemoryProj)
                M = m.MemoryProj.forward(H);
            end
            Mem = OceanDL.Autograd.Tensor.raw(M);
            memId = prevId;
            if trace.Enabled && ~isempty(m.MemoryProj)
                Mr = double(gather(Mem));
                memId = trace.record("Decoder memory", "model", struct('Values', reshape(Mr(:, 1, 1:T1), size(Mr, 1), []), ...
                    'FrameTimes', encTimes, 'FrequencyHz', [], 'AxisNames', ["hidden unit", "time"], ...
                    'Units', "activation", 'Title', "Decoder memory m_t = P h_t"), prevId, ...
                    'Summary', sprintf("Linear projection %d -> %d of the encoder output", size(OceanDL.Autograd.Tensor.raw(H), 1), size(Mr, 1)));
            end
            tokens = m.Vocabulary.Tokens;
            ctcId = strings(1, 0);
            if trace.Enabled && ~isempty(m.CTCHead) && T1 > 0
                C = double(gather(OceanDL.Autograd.Tensor.raw(m.CTCHead.forward(H))));
                cl = reshape(C(:, 1, 1:T1), size(C, 1), T1);
                mx = max(cl, [], 1);
                ex = exp(cl - mx);
                [~, best] = max(cl, [], 1);
                ctcId = trace.record("CTC head posteriors", "decode", struct('Posteriors', ex ./ sum(ex, 1), ...
                    'Tokens', tokens, 'FrameTimes', encTimes, 'BestPath', best, 'Blank', m.Vocabulary.Blank), prevId, ...
                    'Summary', "Auxiliary joint-CTC head softmax(W_ctc h_t) (training prior; used only for CTC rescoring)");
            end

            % ---- search with the model's own step functions ----------------------
            acc = [];
            if trace.Enabled
                acc = containers.Map('KeyType', 'double', 'ValueType', 'any');
            end
            if o.UseCache
                base = @(prefixes, src, st, parent) m.nextLogProbsCached(prefixes, src, st, parent);
                if trace.Enabled
                    step = @(prefixes, src, st, parent) recordingStepCached(base, acc, prefixes, src, st, parent);
                else
                    step = base;
                end
                tr = dec.decode(step, lens, 'Incremental', true, 'State', m.initDecodeState(Mem, lens));
            else
                base = @(prefixes, src) m.nextLogProbs(prefixes, Mem(:, src, :), lens(src));
                if trace.Enabled
                    step = @(prefixes, src) recordingStep(base, acc, prefixes, src);
                else
                    step = base;
                end
                tr = dec.decode(step, lens);
            end
            if o.CTCRescoreWeight > 0
                tr = OceanNLP.Speech.Inspect.SpeechProcessAdapter.ctcRescore_(m, tr, H, lens, ...
                    o.CTCRescoreWeight, o.LengthPenalty);
            end

            searchId = memId;
            postId = strings(1, 0);
            attId = strings(1, 0);
            if trace.Enabled
                n = acc.Count;
                stepsRec = struct('Step', cell(1, n), 'Prefixes', [], 'Source', [], 'Parent', [], 'LogProbs', []);
                for i = 1:n
                    stepsRec(i) = acc(i);
                end
                searchId = trace.record("Decoder search steps", "decode", struct('Steps', stepsRec, ...
                    'Tokens', tokens, 'Method', dec.Method, 'BeamWidth', dec.BeamWidth, 'UseCache', o.UseCache, ...
                    'MaxLength', dec.maxLengths(lens(1))), memId, ...
                    'Summary', sprintf("%d step-function calls of the %s search (K x B log-probabilities per call)", n, dec.Method));
                if T1 > 0
                    % teacher-forced replay of the chosen hypothesis (public forward)
                    prevStore = m.StoreAttention;
                    m.StoreAttention = true;
                    storeGuard = onCleanup(@() restoreStore(m, prevStore));
                    tf = batch;
                    tf.Targets = {double(tr.TokenIds)};
                    out = m.forward(tf);
                    clear storeGuard
                    Z = double(gather(OceanDL.Autograd.Tensor.raw(out.Logits)));
                    U = size(Z, 3);
                    z = reshape(Z(:, 1, 1:U), size(Z, 1), U);
                    mx = max(z, [], 1);
                    lse = mx + log(sum(exp(z - mx), 1));
                    post = exp(z - lse);
                    ids = [double(tr.TokenIds), m.Vocabulary.Eos];
                    lp = sum(z(sub2ind(size(z), ids, 1:U)) - lse);
                    labels = tokens(ids);
                    postId = trace.record("Decoder posteriors", "decode", struct('Posteriors', post, ...
                        'Values', post, 'Tokens', tokens, 'TokenIds', ids, 'StepLabels', labels, 'LogProb', lp, ...
                        'FrameTimes', [], 'FrequencyHz', [], 'AxisNames', ["class", "decoder step"], ...
                        'Units', "probability", 'Title', "Decoder token posteriors (teacher-forced on the hypothesis)"), ...
                        searchId, 'Summary', sprintf("p(z_u | z_<u, x) for %d decoder steps; sum ln p(z_u) = %.4f", U, lp));
                    A = double(gather(OceanDL.Autograd.Tensor.raw(out.CrossAttention{end})));
                    A = A(:, 1:T1, :, 1);
                    attId = trace.record("Cross-attention", "decode", struct('Values', mean(A, 3), 'PerHead', A, ...
                        'FrameTimes', encTimes, 'FrequencyHz', [], 'StepLabels', labels, ...
                        'AxisNames', ["decoder step", "time"], 'Units', "attention weight", ...
                        'Layer', numel(m.Blocks), 'Title', sprintf("Cross-attention, decoder layer %d (mean of %d heads)", ...
                        numel(m.Blocks), size(A, 3))), searchId, ...
                        'Summary', sprintf("%d decoder steps x %d encoder frames x %d heads", U, T1, size(A, 3)));
                end
            end
            clear restore

            candId = trace.record("Beam candidates", "decode", struct('Candidates', tr.Candidates, ...
                'Method', dec.Method, 'BeamWidth', dec.BeamWidth, 'Chosen', 1, ...
                'ScoreLabel', OceanNLP.Speech.Inspect.SpeechProcessAdapter.attentionScoreLabel_(dec, o.CTCRescoreWeight)), ...
                [searchId, postId, attId, ctcId], 'Summary', sprintf("%s decoding: %d candidate(s)", tr.Decoder, numel(tr.Candidates)));
            trace.record("Transcript", "decode", struct('Text', tr.Text, 'TokenIds', tr.TokenIds, ...
                'Score', tr.Score, 'Decoder', tr.Decoder, 'ModelId', tr.ModelId), candId, ...
                'Summary', "Final transcript: """ + tr.Text + """");
        end

        function t = ctcRescore_(m, t, H, lens, lambda, alpha)
            % Restated EncoderDecoderRecognizer.ctcRescore (protected): joint CTC N-best rescoring.
            C = OceanDL.Autograd.Tensor.raw(m.CTCHead.forward(H));
            blank = m.Vocabulary.Blank;
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
                if ~any(isfinite([c.Score]))
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

        function X = padToMinimumFrames_(m, X)
            % Restated ISpeechRecognizer.padToMinimumFrames (protected): right-pad with zeros
            % until the encoders produce at least one output frame.
            T = size(X, 3);
            if m.outputLength(T) >= 1
                return;
            end
            Tmin = T + 1;
            while m.outputLength(Tmin) < 1
                Tmin = Tmin + 1;
                if Tmin > 4096
                    error('OceanNLP:Speech:Models:ISpeechRecognizer:ReceptiveField', ...
                        'Encoders produce no output frame for any input up to 4096 frames.');
                end
            end
            [F, N, ~] = size(X, 1:3);
            X = cat(3, X, zeros(F, N, Tmin - T, 'like', X));
        end

        function s = attentionScoreLabel_(dec, lambda)
            if lambda > 0
                s = sprintf("[(1 - %g) ln p_att(y|x) + %g ln p_ctc(y|x)] / (|y| + 1)^%g (nats)", lambda, lambda, dec.LengthPenalty);
            else
                s = sprintf("ln p_att(y, Eos | x) / (|y| + 1)^%g (nats)", dec.LengthPenalty);
            end
        end

        function lastId = recordFrontEnd_(features, waveform, waveId, trace)
            % Record the spectral front end through its own intermediates() method.
            lm = OceanNLP.Speech.Inspect.SpeechProcessAdapter.logMelOf_(features);
            lastId = waveId;
            if isempty(lm)
                return;
            end
            st = lm.intermediates(waveform);
            g = lm.Grid;
            S = st.Waveform.numSamples();
            frameId = trace.record("Frames + window", "frontend", struct('Frames', st.Frames, ...
                'Window', st.Window, 'WindowType', g.Window, 'FrameStarts', g.frameStarts(S), ...
                'FrameTimes', st.FrameTimes, 'WindowLength', g.WindowLength, 'HopLength', g.HopLength, ...
                'NFFT', g.NFFT, 'SampleRate', g.SampleRate), waveId, ...
                'Summary', sprintf("FrameGrid: L = %d, H = %d samples, %s window, %d frames", ...
                g.WindowLength, g.HopLength, g.Window, size(st.Frames, 2)));
            powUnits = "power |X[k]|^2";
            if lm.Power ~= 2
                powUnits = sprintf("|X[k]|^%g", lm.Power);
            end
            powId = trace.record("Power spectrum", "frontend", struct('Values', st.PowerSpectrum, ...
                'FrequencyHz', st.BinFrequencies, 'FrameTimes', st.FrameTimes, 'Units', powUnits, ...
                'SampleRate', g.SampleRate, 'Title', "Power spectrum (NFFT = " + g.NFFT + ")"), frameId, ...
                'Summary', "P_t[k] = |sum_n w[n] x_t[n] e^{-j2 pi kn/NFFT}|^2");
            bankId = trace.record("Mel filter bank", "frontend", struct('Weights', lm.Filters, ...
                'BinFrequencyHz', st.BinFrequencies, 'CentersHz', lm.CentersHz, 'Scale', lm.MelScale, ...
                'Norm', lm.MelNorm, 'FMin', lm.FMin, 'FMax', lm.FMax), strings(1, 0), ...
                'Summary', sprintf("%d triangular %s filters on %d bins, %g-%g Hz", lm.NumMels, ...
                lm.MelScale, numel(st.BinFrequencies), lm.FMin, lm.FMax));
            lastId = trace.record("Log-mel", "frontend", struct('Values', st.LogMel, ...
                'FrameTimes', st.FrameTimes, 'FrequencyHz', lm.CentersHz, 'AxisNames', ["mel", "time"], ...
                'Units', "ln(" + powUnits + ")", 'Title', "Log-mel spectrogram"), [powId, bankId], ...
                'Summary', sprintf("Y = ln(max(W P, %g))", lm.LogFloor));
        end

        function lm = logMelOf_(features)
            lm = [];
            if isa(features, 'OceanNLP.Speech.Features.LogMel')
                lm = features;
            elseif isa(features, 'OceanNLP.Speech.Features.MFCC')
                lm = features.LogMelTransform;
            elseif isa(features, 'OceanNLP.Speech.Features.FeaturePipeline')
                lm = OceanNLP.Speech.Inspect.SpeechProcessAdapter.logMelOf_(features.Stages{1});
            end
        end

        function d = featureData_(fs, titleText)
            d = struct('Values', fs.Values, 'FrameTimes', fs.FrameTimes, 'FrequencyHz', fs.FrequencyHz, ...
                'AxisNames', fs.AxisNames, 'Units', fs.Units, 'Title', titleText, 'Config', fs.Config);
        end

        function h = hopSeconds_(fs)
            % Frame hop in seconds from the recorded FrameGrid config (or the frame times).
            h = NaN;
            c = fs.Config;
            if isfield(c, 'Grid')
                h = c.Grid.HopLength / c.Grid.SampleRate;
            elseif isfield(c, 'LogMel') && isfield(c.LogMel, 'Grid')
                h = c.LogMel.Grid.HopLength / c.LogMel.Grid.SampleRate;
            elseif numel(fs.FrameTimes) > 1
                h = fs.FrameTimes(2) - fs.FrameTimes(1);
            end
        end

        function s = shortClass_(obj)
            s = string(class(obj));
            s = extractAfter(s, asManyOfPattern(wildcardPattern + ".", 1));
        end

        function s = scoreLabel_(dec)
            if dec.Method == "greedy"
                s = "Best-path log probability sum_t max_k ln y_k(t) (nats)";
            else
                s = "Beam score ln p_ctc(l|x) + LMWeight ln p_lm(l) + Bonus |l| (nats)";
            end
        end
    end
end

function [lp, st] = recordingStepCached(base, acc, prefixes, src, st, parent)
% Calls the model's incremental step unchanged and keeps a copy of what it returned.
[lp, st] = base(prefixes, src, st, parent);
k = acc.Count + 1;
acc(k) = struct('Step', k, 'Prefixes', prefixes, 'Source', src(:), 'Parent', parent(:), ...
    'LogProbs', double(gather(OceanDL.Autograd.Tensor.raw(lp))));
end

function lp = recordingStep(base, acc, prefixes, src)
% Calls the model's (non-incremental) step unchanged and keeps a copy of what it returned.
lp = base(prefixes, src);
k = acc.Count + 1;
acc(k) = struct('Step', k, 'Prefixes', prefixes, 'Source', src(:), 'Parent', [], ...
    'LogProbs', double(gather(OceanDL.Autograd.Tensor.raw(lp))));
end

function restoreStore(m, v)
m.StoreAttention = v;
end
