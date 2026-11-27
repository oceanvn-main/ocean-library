classdef AudioDataset < handle
    % AUDIODATASET  Lazy (waveform, features, transcript, token ids) items from a manifest.
    %
    %   ds = OceanNLP.Speech.Data.AudioDataset(manifest, features, 'Vocabulary', v, ...
    %          'Normalizer', n, 'SampleRate', fs, 'Cache', true, 'Augment', a)
    %   it = ds.item(i)    struct Id, Speaker, Waveform (Core.Waveform),
    %                      Features (Core.FeatureSequence), Text (normalized), TokenIds
    %   N  = ds.numel() / ds.length();   L = ds.lengths()   (1 x N frame counts)
    %   ds.setEpoch(e)     forwarded to the augmenter (reproducible per epoch)
    %
    %   features: an OceanNLP.Speech.Core.IFeatureTransform taking a Waveform.
    %   Loading: Core.Waveform.read of each component (root-relative Source); a
    %   composed entry ("a.wav+b.wav") is the vertical concatenation of its
    %   components, which must share sample rate and channel count.
    %   Resampling: if 'SampleRate' is given and differs from the file rate, the
    %   waveform is resampled with OceanMath.Signal.ResampleSignal.resample and the
    %   step is recorded in Waveform.History; if that class is unavailable the item
    %   fails with ':ResamplerUnavailable' (never a silent fallback).
    %   Text: Normalizer.forward(Transcript). Normalizer defaults to the vocabulary's
    %   normalizer (or TranscriptNormalizer() without a vocabulary); a Normalizer whose
    %   config differs from the vocabulary's is rejected (':NormalizerMismatch').
    %   TokenIds: Vocabulary.encode(Transcript) ([] without a vocabulary).
    %   Augment (AudioAugment): waveform augmentation before the features, SpecAugment
    %   after. Cache=true keeps waveforms, and features when no augmentation is active.
    %
    %   lengths(): frame counts. Without loading audio when the transform exposes
    %   numFrames(S) (or a Grid property with numFrames(S)): S = round(Duration * rate)
    %   at the file rate, S' = ceil(S p / q) with p/q = rat(fs / rate) after
    %   resampling (the ResampleSignal contract). Otherwise each item's features are
    %   computed (un-augmented) and counted.

    properties (SetAccess = private)
        Manifest
        Features
        Vocabulary
        Normalizer
        SampleRate
        Cache (1,1) logical
        Augment
    end

    properties (Access = private)
        WaveCache cell
        FeatCache cell
    end

    methods
        function obj = AudioDataset(manifest, features, opts)
            arguments
                manifest (1,1) OceanNLP.Speech.Data.SpeechManifest
                features (1,1) OceanNLP.Speech.Core.IFeatureTransform
                opts.Vocabulary = []
                opts.Normalizer = []
                opts.SampleRate double = []
                opts.Cache (1,1) logical = true
                opts.Augment = []
            end
            if ~isempty(opts.Vocabulary) && ~isa(opts.Vocabulary, 'OceanNLP.Speech.Data.SpeechVocabulary')
                error('OceanNLP:Speech:Data:AudioDataset:Vocabulary', 'Vocabulary must be a SpeechVocabulary.');
            end
            if ~isempty(opts.Augment) && ~isa(opts.Augment, 'OceanNLP.Speech.Data.AudioAugment')
                error('OceanNLP:Speech:Data:AudioDataset:Augment', 'Augment must be an AudioAugment.');
            end
            if ~isempty(opts.SampleRate) && ~(isscalar(opts.SampleRate) && isfinite(opts.SampleRate) && opts.SampleRate > 0)
                error('OceanNLP:Speech:Data:AudioDataset:SampleRate', 'SampleRate must be [] or a positive finite scalar.');
            end
            norm = opts.Normalizer;
            if isempty(norm)
                if ~isempty(opts.Vocabulary) && ~isempty(opts.Vocabulary.Normalizer)
                    norm = opts.Vocabulary.Normalizer;
                else
                    norm = OceanNLP.Speech.Data.TranscriptNormalizer();
                end
            elseif ~isempty(opts.Vocabulary)
                vn = opts.Vocabulary.Normalizer;
                if isempty(vn) || ~isequal(vn.config(), norm.config())
                    error('OceanNLP:Speech:Data:AudioDataset:NormalizerMismatch', ...
                        'Normalizer differs from the vocabulary''s; Text and TokenIds would disagree.');
                end
            end
            obj.Manifest = manifest;
            obj.Features = features;
            obj.Vocabulary = opts.Vocabulary;
            obj.Normalizer = norm;
            obj.SampleRate = opts.SampleRate;
            obj.Cache = opts.Cache;
            obj.Augment = opts.Augment;
            obj.WaveCache = cell(1, manifest.height());
            obj.FeatCache = cell(1, manifest.height());
        end

        function n = numel(obj, varargin)
            n = obj.Manifest.height();
        end

        function n = length(obj)
            n = obj.Manifest.height();
        end

        function it = item(obj, i)
            arguments
                obj
                i (1,1) double {mustBeInteger, mustBePositive}
            end
            N = obj.Manifest.height();
            if i > N
                error('OceanNLP:Speech:Data:AudioDataset:Index', 'Index %d exceeds %d items.', i, N);
            end
            w = obj.waveform(i);
            augmenting = ~isempty(obj.Augment) && obj.Augment.isEnabled();
            if augmenting
                w = obj.Augment.forwardWaveform(w);          % item reports the augmented audio
                f = obj.Augment.forwardFeatures(obj.Features.forward(w));
            elseif obj.Cache && ~isempty(obj.FeatCache{i})
                f = obj.FeatCache{i};
            else
                f = obj.Features.forward(w);
                if obj.Cache, obj.FeatCache{i} = f; end
            end
            E = obj.Manifest.Entries;
            ids = zeros(1, 0);
            if ~isempty(obj.Vocabulary)
                ids = obj.Vocabulary.encode(E.Transcript(i));
            end
            it = struct('Id', E.Id(i), 'Speaker', E.Speaker(i), 'Waveform', w, ...
                'Features', f, 'Text', obj.Normalizer.forward(E.Transcript(i)), 'TokenIds', ids);
        end

        function L = lengths(obj)
            N = obj.Manifest.height();
            L = zeros(1, N);
            counter = obj.frameCounter();
            E = obj.Manifest.Entries;
            for i = 1:N
                if ~isempty(counter)
                    S = round(E.Duration(i) * E.SampleRate(i));
                    if ~isempty(obj.SampleRate) && obj.SampleRate ~= E.SampleRate(i)
                        [p, q] = rat(obj.SampleRate / E.SampleRate(i));
                        S = ceil(S * p / q);
                    end
                    L(i) = counter(S);
                else
                    if obj.Cache && ~isempty(obj.FeatCache{i})
                        f = obj.FeatCache{i};
                    else
                        f = obj.Features.forward(obj.waveform(i));
                        if obj.Cache, obj.FeatCache{i} = f; end
                    end
                    L(i) = f.numFrames();
                end
            end
        end

        function setEpoch(obj, epoch)
            arguments
                obj
                epoch (1,1) double {mustBeInteger, mustBeNonnegative}
            end
            if ~isempty(obj.Augment)
                obj.Augment.setEpoch(epoch);
            end
        end

        function w = waveform(obj, i)
            % WAVEFORM  Un-augmented (resampled, concatenated) waveform of item i.
            if obj.Cache && ~isempty(obj.WaveCache{i})
                w = obj.WaveCache{i};
                return;
            end
            m = obj.Manifest;
            files = m.Entries.AudioPath(i);
            comps = split(files, "+");
            parts = cell(1, numel(comps));
            for k = 1:numel(comps)
                parts{k} = OceanNLP.Speech.Core.Waveform.read(comps(k), 'Root', m.Root);
            end
            if isscalar(parts)
                w = parts{1};
            else
                rates = cellfun(@(p) p.SampleRate, parts);
                chans = cellfun(@(p) p.numChannels(), parts);
                if any(rates ~= rates(1)) || any(chans ~= chans(1))
                    error('OceanNLP:Speech:Data:AudioDataset:ComponentMismatch', ...
                        'Components of %s differ in sample rate or channel count.', m.Entries.Id(i));
                end
                x = cell2mat(cellfun(@(p) p.Samples, parts, 'UniformOutput', false)');
                w = OceanNLP.Speech.Core.Waveform(x, rates(1), 'Source', files, ...
                    'History', sprintf("concatenated %d recordings", numel(parts)));
            end
            if ~isempty(obj.SampleRate) && w.SampleRate ~= obj.SampleRate
                if exist('OceanMath.Signal.ResampleSignal', 'class') ~= 8
                    error('OceanNLP:Speech:Data:AudioDataset:ResamplerUnavailable', ...
                        ['%s is at %g Hz but %g Hz was requested, and OceanMath.Signal.ResampleSignal ', ...
                        'is not available.'], m.Entries.Id(i), w.SampleRate, obj.SampleRate);
                end
                x = w.Samples;
                y = [];
                for c = 1:size(x, 2)
                    yc = OceanMath.Signal.ResampleSignal.resample(x(:, c), w.SampleRate, obj.SampleRate);
                    y = [y, yc(:)]; %#ok<AGROW>
                end
                w = w.derive(y, obj.SampleRate, sprintf("resampled %g -> %g Hz (OceanMath.Signal.ResampleSignal)", ...
                    w.SampleRate, obj.SampleRate));
            end
            if obj.Cache, obj.WaveCache{i} = w; end
        end
    end

    methods (Access = private)
        function counter = frameCounter(obj)
            % Function handle S -> number of frames, or [] when the transform has none.
            counter = [];
            f = obj.Features;
            if ismethod(f, 'numFrames')
                counter = @(S) f.numFrames(S);
            elseif isprop(f, 'Grid') && ~isempty(f.Grid) && ismethod(f.Grid, 'numFrames')
                g = f.Grid;
                counter = @(S) g.numFrames(S);
            end
        end
    end
end
