classdef CTCRecognizer < OceanNLP.Speech.Models.ISpeechRecognizer
    % CTCRECOGNIZER  Encoder chain + linear CTC head (Graves et al. 2006), trainable end to end.
    %
    %   m   = OceanNLP.Speech.Models.CTCRecognizer(features, vocab, encoders, ...
    %           'InputNorm', "utterance", 'Dropout', 0, 'Seed', 0, 'ModelId', "", 'ZeroInfinity', false)
    %   out = m.forward(batch)      batch: SpeechBatcher struct (X F x N x T, Lengths 1 x N)
    %                               out: Logits (K x N x T' Tensor), Lengths (1 x N), EncoderOutput
    %   L   = m.loss(batch)         F.ctcLoss(Logits, batch.Targets, InputLengths = Lengths, Blank = 1)
    %   tr  = m.transcribe(input, 'Decoder', "greedy"|"beam", 'BeamWidth', 8, ...)
    %   cfg = m.config();   m2 = OceanNLP.Speech.Models.RecognizerFactory.fromConfig(cfg)
    %
    %   COMPUTATION, for utterance n with T(n) valid feature frames x_1..x_T(n) (F x 1):
    %     InputNorm "utterance" (per-utterance mean/variance normalisation, CMVN):
    %       x~_t = (x_t - mu) ./ sqrt(sigma^2 + 1e-8),
    %       mu = (1/T) sum_{t<=T} x_t,  sigma^2 = (1/T) sum_{t<=T} (x_t - mu).^2   (valid frames only)
    %     ("none": x~ = x). Stateless, so it needs no fitted statistics.
    %     [h, T'] = E_J( ... E_1(x~, T) )         encoder chain (each propagates lengths)
    %     u_t = W Dropout(h_t) + b                 K logits per output frame, t <= T'(n)
    %     y_k(t) = softmax(u_t)_k,  L_n = -log sum_{pi in B^{-1}(l_n)} prod_t y_{pi_t}(t)
    %   loss = mean_n L_n / |l_n| (OceanDL CTCLoss "mean", PyTorch default reduction).
    %   Output frames t > T'(n) are zeroed in Logits and excluded from the loss
    %   (InputLengths) and from decoding (lengths) -- padding never reaches outputs.
    %
    %   transcribe(input): input is a Core.Waveform (features computed by m.Features),
    %   a Core.FeatureSequence (must have InputDim bins and, when it records a
    %   Config, the same feature configuration as m.Features), an array/cell of
    %   either, or a SpeechBatcher batch struct. Runs in eval mode without a graph,
    %   decodes with Decoding.CTCPrefixBeamDecoder, returns 1 x N Core.Transcript.
    %   No timestamps are produced (CTC spikes are not an alignment).
    %   Vocabulary/model compatibility (vocab.assertCompatible(K)) is checked at
    %   construction and before every transcribe.

    properties (SetAccess = protected)
        Encoders cell = {}
        Head
        Drop
        InputDim (1,1) double
        InputNorm (1,1) string
        DropoutP (1,1) double
        Seed (1,1) double
        ZeroInfinity (1,1) logical
    end

    methods
        function obj = CTCRecognizer(features, vocabulary, encoders, opts)
            arguments
                features (1,1) OceanNLP.Speech.Core.IFeatureTransform
                vocabulary (1,1) OceanNLP.Speech.Data.SpeechVocabulary
                encoders (1,:) cell {mustBeNonempty}
                opts.InputNorm (1,1) string {mustBeMember(opts.InputNorm, ["utterance", "none"])} = "utterance"
                opts.Dropout (1,1) double {mustBeGreaterThanOrEqual(opts.Dropout, 0), mustBeLessThan(opts.Dropout, 1)} = 0
                opts.Seed (1,1) double {mustBeInteger, mustBeNonnegative} = 0
                opts.ModelId (1,1) string = ""
                opts.ZeroInfinity (1,1) logical = false
            end
            fdim = features.outputDim([]);
            d = fdim;
            for j = 1:numel(encoders)
                e = encoders{j};
                if ~isa(e, 'OceanDL.nn.Module') || ~ismethod(e, 'outputLength') || ~isprop(e, 'OutputDim') ...
                        || ~isprop(e, 'InputDim') || ~ismethod(e, 'config')
                    error('OceanNLP:Speech:Models:CTCRecognizer:EncoderContract', ...
                        'Encoder %d (%s) must be an OceanDL.nn.Module with InputDim, OutputDim, forward(X, lens), outputLength, config.', ...
                        j, class(e));
                end
                if e.InputDim ~= d
                    error('OceanNLP:Speech:Models:CTCRecognizer:DimMismatch', ...
                        'Encoder %d expects %d inputs but receives %d.', j, e.InputDim, d);
                end
                d = e.OutputDim;
            end
            obj.Features = features;
            obj.Vocabulary = vocabulary;
            obj.ModelId = opts.ModelId;
            obj.Encoders = encoders;
            obj.InputDim = fdim;
            obj.InputNorm = opts.InputNorm;
            obj.DropoutP = opts.Dropout;
            obj.Seed = opts.Seed;
            obj.ZeroInfinity = opts.ZeroInfinity;
            obj.Drop = OceanDL.nn.Dropout(opts.Dropout);
            obj.Head = OceanDL.nn.Linear(d, vocabulary.numClasses(), 'Seed', opts.Seed);
            obj.Vocabulary.assertCompatible(obj.Head.OutFeatures);
        end

        function out = forward(obj, batch)
            arguments
                obj
                batch (1,1) struct
            end
            [X, lens] = obj.checkBatch(batch);
            % X is F x N x T.  LENS is the only authority on which right-padded
            % frames are real; every encoder updates it if it subsamples time.
            X = obj.normalizeInput(X, lens);
            H = X;
            for j = 1:numel(obj.Encoders)
                [H, lens] = obj.Encoders{j}.forward(H, lens);
            end
            % The linear head maps encoder channels to vocabulary classes at each
            % remaining time step.  Mask again so padded logits cannot leak into
            % loss, decoding, inspection, or later numerical comparisons.
            Z = obj.Head.forward(obj.Drop.forward(H));
            Z = OceanDL.Layout.SequenceMask.apply(Z, OceanDL.Layout.SequenceMask.fromLengths(lens, size(Z, 3)));
            out = struct('Logits', Z, 'Lengths', lens, 'EncoderOutput', H);
        end

        function L = loss(obj, batch)
            arguments
                obj
                batch (1,1) struct
            end
            if ~isfield(batch, 'Targets')
                error('OceanNLP:Speech:Models:CTCRecognizer:NoTargets', 'loss needs batch.Targets (cell of token-id rows).');
            end
            out = obj.forward(batch);
            L = OceanDL.Autograd.F.ctcLoss(out.Logits, batch.Targets, 'InputLengths', out.Lengths, ...
                'Blank', obj.Vocabulary.Blank, 'Reduction', "mean", 'ZeroInfinity', obj.ZeroInfinity);
        end

        function tr = transcribe(obj, input, opts)
            arguments
                obj
                input
                opts.Decoder (1,1) string {mustBeMember(opts.Decoder, ["greedy", "beam"])} = "greedy"
                opts.BeamWidth (1,1) double {mustBePositive, mustBeInteger} = 8
                opts.Prune (1,1) double = -inf
                opts.NBest double {mustBeScalarOrEmpty} = []
                opts.LanguageModel = []
                opts.LMWeight (1,1) double = 0
                opts.Bonus (1,1) double = 0
                opts.BatchSize (1,1) double {mustBePositive, mustBeInteger} = 32
            end
            obj.Vocabulary.assertCompatible(obj.Head.OutFeatures);
            dec = OceanNLP.Speech.Decoding.CTCPrefixBeamDecoder(obj.Vocabulary, 'Method', opts.Decoder, ...
                'BeamWidth', opts.BeamWidth, 'Prune', opts.Prune, 'NBest', opts.NBest, ...
                'LanguageModel', opts.LanguageModel, 'LMWeight', opts.LMWeight, 'Bonus', opts.Bonus, ...
                'ModelId', obj.ModelId);
            wasTraining = obj.Training;
            obj.eval();
            % Restore the caller's train/eval state even if feature extraction or
            % decoding throws.  Inference also disables graph construction because
            % no backward pass will consume these logits.
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
            for b = 1:numel(batches)
                out = obj.forward(batches{b});
                tr = [tr, dec.decode(out.Logits, out.Lengths)]; %#ok<AGROW>
            end
            clear restore
        end

        function lens = outputLength(obj, lens)
            % OUTPUTLENGTH  Number of logit frames for feature-frame counts LENS.
            for j = 1:numel(obj.Encoders)
                lens = obj.Encoders{j}.outputLength(lens);
            end
        end

        function cfg = config(obj)
            enc = cellfun(@(e) e.config(), obj.Encoders, 'UniformOutput', false);
            cfg = struct('Type', "CTCRecognizer", 'Version', 1, 'InputDim', obj.InputDim, ...
                'InputNorm', obj.InputNorm, 'Dropout', obj.DropoutP, 'Seed', obj.Seed, ...
                'ZeroInfinity', obj.ZeroInfinity, 'NumClasses', obj.Head.OutFeatures, ...
                'Encoders', {enc}, 'Vocabulary', obj.Vocabulary.stateDict(), ...
                'Features', obj.Features.config(), 'ModelId', obj.ModelId);
        end
    end

    methods (Static)
        function obj = fromConfig(cfg)
            % FROMCONFIG  Same architecture, features and vocabulary (fresh weights).
            arguments
                cfg (1,1) struct
            end
            if string(cfg.Type) ~= "CTCRecognizer"
                error('OceanNLP:Speech:Models:CTCRecognizer:ConfigType', 'Config Type is "%s", not "CTCRecognizer".', string(cfg.Type));
            end
            features = OceanNLP.Speech.Models.RecognizerFactory.featuresFromConfig(cfg.Features);
            vocab = OceanNLP.Speech.Data.SpeechVocabulary.fromStateDict(cfg.Vocabulary);
            encs = cfg.Encoders;
            if isstruct(encs), encs = num2cell(encs); end
            encoders = cellfun(@(c) OceanNLP.Speech.Models.RecognizerFactory.encoderFromConfig(c), reshape(encs, 1, []), 'UniformOutput', false);
            obj = OceanNLP.Speech.Models.CTCRecognizer(features, vocab, encoders, 'InputNorm', cfg.InputNorm, ...
                'Dropout', cfg.Dropout, 'Seed', cfg.Seed, 'ModelId', cfg.ModelId, 'ZeroInfinity', cfg.ZeroInfinity);
            if obj.Head.OutFeatures ~= cfg.NumClasses
                error('OceanNLP:Speech:Models:CTCRecognizer:ConfigNumClasses', ...
                    'Config says %d classes, vocabulary gives %d.', cfg.NumClasses, obj.Head.OutFeatures);
            end
        end

        function batch = padSequences(seqs)
            % PADSEQUENCES  Cell of FeatureSequence -> right-zero-padded CBT batch struct.
            N = numel(seqs);
            L = cellfun(@(s) s.numFrames(), seqs);
            F = seqs{1}.numBins();
            T = max([L, 0]);
            X = zeros(F, N, T);
            for n = 1:N
                % Right padding keeps all utterances aligned at frame 1.  Lengths
                % and Mask, rather than the zero value itself, identify padding.
                X(:, n, 1:L(n)) = reshape(seqs{n}.Values, F, 1, L(n));
            end
            batch = struct('X', X, 'Lengths', L, 'Mask', (1:T) <= L(:));
        end
    end

    methods (Access = protected)
        function [X, lens] = checkBatch(obj, batch)
            if ~isfield(batch, 'X') || ~isfield(batch, 'Lengths')
                error('OceanNLP:Speech:Models:CTCRecognizer:Batch', 'batch needs fields X (F x N x T) and Lengths (1 x N).');
            end
            X = double(batch.X);
            lens = double(reshape(batch.Lengths, 1, []));
            [F, N, T] = size(X, 1:3);
            if F ~= obj.InputDim
                error('OceanNLP:Speech:Models:CTCRecognizer:InputDim', ...
                    'Model expects %d feature bins, batch has %d.', obj.InputDim, F);
            end
            if numel(lens) ~= N || any(lens < 0 | lens > T | lens ~= round(lens))
                error('OceanNLP:Speech:Models:CTCRecognizer:Lengths', 'Lengths must be N = %d integers in [0, %d].', N, T);
            end
            X = obj.padToMinimumFrames(X);
        end

        function X = normalizeInput(obj, X, lens)
            if obj.InputNorm == "none"
                return;
            end
            for n = 1:size(X, 2)
                if lens(n) == 0, continue; end
                v = X(:, n, 1:lens(n));
                mu = mean(v, 3);
                s2 = mean((v - mu).^2, 3);
                X(:, n, 1:lens(n)) = (v - mu) ./ sqrt(s2 + 1e-8);
            end
            X(:, :, :) = X .* reshape((1:size(X, 3)) <= lens(:), 1, size(X, 2), size(X, 3));
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
                        error('OceanNLP:Speech:Models:CTCRecognizer:InputDim', ...
                            'Input %d has %d bins; the model expects %d.', i, it.numBins(), obj.InputDim);
                    end
                    if ~isempty(fieldnames(it.Config)) && ~isequaln(stripHistory(it.Config), ref)
                        error('OceanNLP:Speech:Models:CTCRecognizer:FeatureMismatch', ...
                            'Input %d was computed with a different feature configuration than the model''s.', i);
                    end
                else
                    error('OceanNLP:Speech:Models:CTCRecognizer:InputType', ...
                        'transcribe takes a Waveform, FeatureSequence, an array/cell of them, or a batch struct (got %s).', class(it));
                end
                if it.numBins() ~= obj.InputDim
                    error('OceanNLP:Speech:Models:CTCRecognizer:InputDim', ...
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
