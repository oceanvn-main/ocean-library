classdef StudioSession < handle
    % STUDIOSESSION  Model layer of the Speech Pipeline Studio: input audio, the chosen
    %   recogniser and the last pipeline run. Thin -- every computation is a library call.
    %
    %   s = OceanViz.App.NLP.SpeechStudio.Models.StudioSession('Root', repoRoot)
    %   T = s.models()                        ModelCatalog.list() rows that are recognisers
    %   s.selectModel(name)                   ModelCatalog.load (cached)
    %   m = s.manifest()                      SpeechManifest.fromFSDD().canonicalFSDDSplit() (lazy)
    %   s.useFSDD(id) | s.useWav(file) | s.useSamples(x, fs, label)   set the input
    %   ctx = s.run(settings, progressFcn)    Inference.RecognitionPipeline.run -> Stages.StageContext
    %
    %   SETTINGS (struct; see defaultSettings): decoder (Decoder, BeamWidth, LMWeight,
    %   Bonus, LMOrder, CTCRescoreWeight) and robustness (UseDetector, Sensitivity, MaxSegment,
    %   Segmentation, Normalize, TargetRMSdBFS, MaxGainDB, RemoveDC). Robustness settings map 1:1
    %   onto Inference.AudioConditioner / Inference.SpeechActivityDetector options.
    %   The CTC language model is Decoding.TokenNGramLM fitted on the canonical FSDD
    %   TRAIN transcripts (never validation/test), built once per order.
    %   Runs are cached per (model, input, settings, reference).

    properties (SetObservable, SetAccess = private)
        ModelName (1,1) string = ""
        Recognizer = []
        Input = []                      % OceanNLP.Speech.Core.Waveform as captured / read
        InputLabel (1,1) string = ""
        InputKind (1,1) string = ""     % "fsdd" | "wav" | "microphone"
        Reference (1,1) string = ""     % raw reference text ("" when unknown)
        Context = []                    % Stages.StageContext of the last run
    end

    properties (SetAccess = private)
        Root (1,1) string
        CheckpointDir (1,1) string
        Catalog = []
    end

    properties (Access = private)
        Manifest_ = []
        LMs = struct()
        RunCache = []                   % containers.Map, created per session (never a class default)
        RunOrder (1,:) string = strings(1, 0)   % cache keys oldest -> newest (FIFO eviction)
        InputKey (1,1) string = ""
    end

    properties (Constant)
        MaxCachedRuns = 8
    end

    methods
        function obj = StudioSession(opts)
            arguments
                opts.Root (1,1) string = OceanViz.App.NLP.SpeechStudio.Models.StudioSession.repoRoot()
                opts.CheckpointDir (1,1) string = ""
                opts.Manifest = []
            end
            obj.Root = opts.Root;
            obj.CheckpointDir = opts.CheckpointDir;
            if obj.CheckpointDir == ""
                obj.CheckpointDir = fullfile(obj.Root, "Results", "Speech", "checkpoints");
            end
            obj.Manifest_ = opts.Manifest;
            obj.RunCache = containers.Map('KeyType', 'char', 'ValueType', 'any');
        end

        function n = numCachedRuns(obj)
            n = double(obj.RunCache.Count);
        end

        function releaseRuns(obj)
            % RELEASERUNS  Drop every cached run (traces) and the current context.
            remove(obj.RunCache, keys(obj.RunCache));
            obj.RunOrder = strings(1, 0);
            obj.Context = [];
        end

        function delete(obj)
            if ~isempty(obj.RunCache) && isvalid(obj.RunCache), remove(obj.RunCache, keys(obj.RunCache)); end
        end

        function c = catalog(obj)
            if isempty(obj.Catalog)
                obj.Catalog = OceanNLP.Speech.Inference.ModelCatalog('Root', obj.CheckpointDir);
            end
            c = obj.Catalog;
        end

        function T = models(obj)
            % MODELS  Catalog rows of the recognisers (CTC / encoder-decoder) only.
            T = obj.catalog().list();
            keep = ismember(string(T.Type), ["CTCRecognizer", "EncoderDecoderRecognizer"]);
            T = T(keep, :);
        end

        function selectModel(obj, name)
            arguments
                obj
                name (1,1) string
            end
            obj.Recognizer = obj.catalog().load(name);
            obj.ModelName = name;
        end

        function row = modelRow(obj)
            row = table();
            if obj.ModelName == "", return; end
            T = obj.models();
            row = T(string(T.Name) == obj.ModelName, :);
        end

        function caps = capabilities(obj)
            % CAPABILITIES  What the selected recogniser supports (drives stages and controls).
            caps = strings(1, 0);
            m = obj.Recognizer;
            if isempty(m), return; end
            if isa(m, 'OceanNLP.Speech.Models.CTCRecognizer')
                caps = [caps, "ctc", "lm"];
            elseif isa(m, 'OceanNLP.Speech.Models.EncoderDecoderRecognizer')
                caps = [caps, "attention"];
                if ~isempty(m.CTCHead), caps = [caps, "ctc-rescore"]; end
            end
            f = m.Features;
            if isa(f, 'OceanNLP.Speech.Features.LogMel') || isa(f, 'OceanNLP.Speech.Features.MFCC') || ...
                    (isa(f, 'OceanNLP.Speech.Features.FeaturePipeline') && ...
                    (isa(f.Stages{1}, 'OceanNLP.Speech.Features.LogMel') || isa(f.Stages{1}, 'OceanNLP.Speech.Features.MFCC')))
                caps = [caps, "frontend"];
            end
        end

        function m = manifest(obj)
            if isempty(obj.Manifest_)
                obj.Manifest_ = OceanNLP.Speech.Data.SpeechManifest.fromFSDD().canonicalFSDDSplit();
            end
            m = obj.Manifest_;
        end

        function useFSDD(obj, id)
            % USEFSDD  Manifest entry ID: the raw recording and its transcript.
            arguments
                obj
                id (1,1) string
            end
            m = obj.manifest();
            i = find(m.Entries.Id == id, 1);
            if isempty(i)
                error('OceanViz:App:NLP:SpeechStudio:StudioSession:UnknownUtterance', 'No FSDD entry "%s".', id);
            end
            e = m.Entries(i, :);
            w = OceanNLP.Speech.Core.Waveform.read(e.AudioPath, 'Root', m.Root);
            obj.InputLabel = sprintf("FSDD %s (%s, speaker %s)", id, e.Split, e.Speaker);
            obj.InputKind = "fsdd";
            obj.Reference = e.Transcript;
            obj.InputKey = "fsdd:" + id;
            obj.Input = w;                  % last: listeners see a complete input
        end

        function useWav(obj, file, reference)
            arguments
                obj
                file (1,1) string {mustBeFile}
                reference (1,1) string = ""
            end
            w = OceanNLP.Speech.Core.Waveform.read(file);
            [~, n, ext] = fileparts(file);
            obj.InputLabel = "WAV " + n + ext;
            obj.InputKind = "wav";
            obj.Reference = reference;
            d = dir(file);
            obj.InputKey = "wav:" + file + ":" + string(d.datenum);
            obj.Input = w;
        end

        function useSamples(obj, x, fs, label)
            % USESAMPLES  Microphone capture (samples + rate); Source "microphone".
            arguments
                obj
                x {mustBeNumeric, mustBeNonempty}
                fs (1,1) double {mustBePositive}
                label (1,1) string = "Microphone"
            end
            w = OceanNLP.Speech.Core.Waveform(double(x(:, 1)), fs, 'Source', "microphone");
            obj.InputLabel = label;
            obj.InputKind = "microphone";
            obj.Reference = "";
            obj.InputKey = "mic:" + string(numel(x)) + ":" + string(sum(double(x(:)).^2, 'all'));
            obj.Input = w;
        end

        function setReference(obj, text)
            arguments
                obj
                text (1,1) string
            end
            obj.Reference = text;
        end

        function ctx = run(obj, settings, progressFcn)
            % RUN  Whole pipeline once (condition -> detect -> features -> model -> decode).
            arguments
                obj
                settings struct = OceanViz.App.NLP.SpeechStudio.Models.StudioSession.defaultSettings()
                progressFcn = []
            end
            if isempty(obj.Recognizer)
                error('OceanViz:App:NLP:SpeechStudio:StudioSession:NoModel', 'Select a model first.');
            end
            if isempty(obj.Input)
                error('OceanViz:App:NLP:SpeechStudio:StudioSession:NoInput', 'Choose an input first.');
            end
            settings = OceanViz.App.NLP.SpeechStudio.Models.StudioSession.completeSettings(settings);
            key = char(jsonencode(struct('Model', obj.ModelName, 'Input', obj.InputKey, ...
                'Reference', obj.Reference, 'Settings', settings)));
            if isKey(obj.RunCache, key)
                ctx = obj.RunCache(key);
                ctx.SegmentIndex = 1;
                obj.Context = ctx;
                obj.report(progressFcn, 1, "Cached result (same input and settings)");
                return;
            end
            t0 = tic;
            I = 'OceanNLP.Speech.Inference.';
            caps = obj.capabilities();
            row = obj.modelRow();
            fs = obj.modelRate();
            obj.report(progressFcn, 0.05, "Configuring conditioner and speech-activity detector");
            cond = feval([I 'AudioConditioner'], 'TargetRate', fs, 'RemoveDC', settings.RemoveDC, ...
                'Normalize', settings.Normalize, 'TargetRMSdBFS', settings.TargetRMSdBFS, ...
                'MaxGainDB', settings.MaxGainDB);
            det = feval([I 'SpeechActivityDetector'], 'Sensitivity', settings.Sensitivity, ...
                'MaxSegment', settings.MaxSegment);
            pipe = feval([I 'RecognitionPipeline'], obj.Recognizer, 'Conditioner', cond, 'Detector', det, ...
                'UseDetector', settings.UseDetector, 'Segmentation', settings.Segmentation);
            decodeArgs = {'Decoder', settings.Decoder, 'BeamWidth', settings.BeamWidth};
            lm = [];
            if any(caps == "ctc")
                if settings.Decoder == "beam" && (settings.LMWeight ~= 0 || settings.Bonus ~= 0)
                    obj.report(progressFcn, 0.1, "Fitting the token n-gram LM on FSDD train transcripts");
                    lm = obj.languageModel(settings.LMOrder);
                    decodeArgs = [decodeArgs, {'LanguageModel', lm, 'LMWeight', settings.LMWeight, 'Bonus', settings.Bonus}];
                end
            elseif any(caps == "ctc-rescore") && settings.Decoder == "beam" && settings.CTCRescoreWeight > 0
                decodeArgs = [decodeArgs, {'CTCRescoreWeight', settings.CTCRescoreWeight}];
            end
            ref = obj.normalizedReference();
            obj.report(progressFcn, 0.2, "RecognitionPipeline.run: condition, detect, features, encoder, decode");
            r = pipe.run(obj.Input, 'Trace', true, 'Reference', ref, decodeArgs{:});
            info = r.DetectorInfo;
            if isempty(info)          % detector off: run it for the diagnostic plot only
                obj.report(progressFcn, 0.85, "Speech-activity diagnostics (SpeechActivityDetector.detect)");
                [~, info] = det.detect(r.Conditioned, 'GainDB', r.Report.GainDB);
            end
            ctx = OceanViz.App.NLP.SpeechStudio.Stages.StageContext(r, 'Recognizer', obj.Recognizer, ...
                'ModelName', obj.ModelName, 'ModelRow', row, 'Capabilities', caps, 'LanguageModel', lm, ...
                'Settings', settings, 'Reference', ref, 'InputLabel', obj.InputLabel, ...
                'InputKind', obj.InputKind, 'DetectorInfo', info, 'Detector', det, 'Conditioner', cond, ...
                'RunSeconds', toc(t0), 'Notes', obj.domainNotes(r));
            if obj.RunCache.Count >= obj.MaxCachedRuns && ~isempty(obj.RunOrder)
                remove(obj.RunCache, char(obj.RunOrder(1)));   % evict the OLDEST run
                obj.RunOrder(1) = [];
            end
            obj.RunCache(key) = ctx;
            obj.RunOrder(end + 1) = string(key);
            obj.Context = ctx;
            obj.report(progressFcn, 1, "Done");
        end

        function s = normalizedReference(obj)
            s = "";
            if obj.Reference == "" || isempty(obj.Recognizer), return; end
            n = obj.Recognizer.Vocabulary.Normalizer;
            if isempty(n), s = obj.Reference; else, s = n.forward(obj.Reference); end
        end

        function fs = modelRate(obj)
            % MODELRATE  Sample rate of the model's front end (catalog SampleRate first).
            row = obj.modelRow();
            if ~isempty(row) && ismember('SampleRate', row.Properties.VariableNames) && isfinite(row.SampleRate(1))
                fs = row.SampleRate(1);
                return;
            end
            f = obj.Recognizer.Features;
            if isprop(f, 'SampleRate')
                fs = f.SampleRate;
            elseif isa(f, 'OceanNLP.Speech.Features.MFCC')
                fs = f.LogMelTransform.SampleRate;
            elseif isa(f, 'OceanNLP.Speech.Features.FeaturePipeline') && isprop(f.Stages{1}, 'SampleRate')
                fs = f.Stages{1}.SampleRate;
            else
                error('OceanViz:App:NLP:SpeechStudio:StudioSession:SampleRate', 'Cannot determine the model sample rate.');
            end
        end

        function lm = languageModel(obj, order)
            % LANGUAGEMODEL  TokenNGramLM of ORDER fitted on the FSDD train transcripts
            %   (vocabulary-encoded), as in Experiments/Scripts/Speech/exp_speech_ctc_fsdd.m.
            key = sprintf('o%d_%s', order, matlab.lang.makeValidName(obj.ModelName));
            if isfield(obj.LMs, key), lm = obj.LMs.(key); return; end
            m = obj.manifest();
            train = m.Entries.Transcript(m.Entries.Split == "train");
            v = obj.Recognizer.Vocabulary;
            lm = OceanNLP.Speech.Decoding.TokenNGramLM.fit(v.encode(train), v, 'Order', order);
            obj.LMs.(key) = lm;
        end

        function n = domainNotes(obj, r)
            % Honest scope notes: the training vocabulary (from the manifest) and the input domain.
            n = strings(1, 0);
            if obj.InputKind ~= "fsdd"
                words = unique(obj.manifest().Entries.Transcript(obj.manifest().Entries.Split == "train"))';
                n(end+1) = "Model trained on FSDD (" + numel(words) + " words: " + strjoin(words, ", ") + ...
                    "): other words cannot be recognised.";
            end
            if obj.InputKind == "fsdd"
                e = obj.manifest().Entries;
                spk = unique(e.Speaker(e.Split == "train"))';
                if contains(obj.InputLabel, "train")
                    n(end+1) = "This utterance is from the TRAINING split (speakers " + strjoin(spk, ", ") + "): not a held-out result.";
                end
            end
            if ~isempty(r.Warnings), n = [n, reshape(string(r.Warnings), 1, [])]; end
        end
    end

    methods (Static)
        function s = defaultSettings()
            % DEFAULTSETTINGS  Robustness defaults read from the library objects' own configs
            %   (AudioConditioner / SpeechActivityDetector constructor defaults); decoder:
            %   beam width 8 and LM order 3 as in exp_speech_ctc_fsdd.m, LM / rescoring off.
            c = OceanNLP.Speech.Inference.AudioConditioner().config();
            d = OceanNLP.Speech.Inference.SpeechActivityDetector().config();
            s = struct('Decoder', "beam", 'BeamWidth', 8, 'LMWeight', 0, 'Bonus', 0, 'LMOrder', 3, ...
                'CTCRescoreWeight', 0, 'UseDetector', true, 'Sensitivity', d.Sensitivity, ...
                'MaxSegment', d.MaxSegment, 'Segmentation', "auto", 'Normalize', c.Normalize, ...
                'TargetRMSdBFS', c.TargetRMSdBFS, 'MaxGainDB', c.MaxGainDB, 'RemoveDC', c.RemoveDC);
        end

        function s = completeSettings(s)
            d = OceanViz.App.NLP.SpeechStudio.Models.StudioSession.defaultSettings();
            f = fieldnames(d);
            for i = 1:numel(f)
                if ~isfield(s, f{i}), s.(f{i}) = d.(f{i}); end
            end
            s = orderfields(s, d);
        end

        function r = repoRoot()
            % <repo> from this file: <repo>/+OceanViz/+App/+NLP/+SpeechStudio/+Models/StudioSession.m
            here = mfilename('fullpath');
            r = string(here);
            for k = 1:6, r = string(fileparts(r)); end
        end
    end

    methods (Static, Access = private)
        function report(fcn, frac, msg)
            if ~isempty(fcn), fcn(frac, msg); end
        end
    end
end
