classdef StudioViewModel < handle
    % STUDIOVIEWMODEL  Bindable UI state + commands of the Speech Pipeline Studio (no UI code).
    %
    %   vm = OceanViz.App.NLP.SpeechStudio.ViewModels.StudioViewModel(session, 'Recorder', rec)
    %   Commands: setMode, setSource, setSplit, selectUtterance, randomHeldOut, setWavFile,
    %   setReference, startRecording, pollLevel, stopRecording, selectModel, setSetting,
    %   run(progressFcn), selectStage, next, back, selectSegment, setCursor.
    %
    %   STALE RULE: any change of input, model or setting after a run sets IsStale; the
    %   stepper is enabled only when HasResult && ~IsStale (StageEnabled), and Run
    %   recomputes everything once. Stages whose capability the model lacks stay disabled.

    properties (SetObservable)
        Mode (1,1) string = "Pipeline"            % "Pipeline" | "Workflow"
        Source (1,1) string = "FSDD"              % "FSDD" | "WAV" | "Microphone"
        Splits (1,:) string = strings(1, 0)
        Split (1,1) string = ""
        Utterances (1,:) string = strings(1, 0)
        UtteranceId (1,1) string = ""
        WavFile (1,1) string = ""
        ReferenceText (1,1) string = ""
        Devices (1,:) string = strings(1, 0)
        Device (1,1) string = ""
        MaxRecordSeconds (1,1) double = 5
        IsRecording (1,1) logical = false
        InputLevelDBFS (1,1) double = -120
        ModelNames (1,:) string = strings(1, 0)
        ModelName (1,1) string = ""
        ModelCard (1,1) string = ""
        Capabilities (1,:) string = strings(1, 0)
        Settings struct = struct()
        StageTitles (1,:) string = strings(1, 0)
        StageRequires cell = {}
        StageNeedsSpeech (1,:) logical = false(1, 0)
        InputReady (1,1) logical = false
        NoSpeech (1,1) logical = false
        CurrentStage (1,1) double = 1
        StageEnabled (1,:) logical = false(1, 0)
        HasResult (1,1) logical = false
        IsStale (1,1) logical = false
        IsBusy (1,1) logical = false
        Context = []
        SegmentIndex (1,1) double = 1
        NumSegments (1,1) double = 1
        CursorTime (1,1) double = NaN
        Warnings (1,:) string = strings(1, 0)
        StatusText (1,1) string = "Choose an input and a model, then press Run."
        InputSummary (1,1) string = "No input"
        RunSummary (1,1) string = ""
    end

    properties (SetAccess = private)
        Session
        Recorder
    end

    properties (Access = private)
        Listeners = event.listener.empty
        Rng = []
    end

    methods
        function obj = StudioViewModel(session, opts)
            arguments
                session (1,1) OceanViz.App.NLP.SpeechStudio.Models.StudioSession
                opts.Recorder = []
                opts.Stages = []
            end
            obj.Session = session;
            if isempty(opts.Recorder)
                obj.Recorder = OceanViz.App.NLP.SpeechStudio.Models.MicrophoneRecorder();
            else
                obj.Recorder = opts.Recorder;
            end
            stages = opts.Stages;
            if isempty(stages), stages = OceanViz.App.NLP.SpeechStudio.Stages.StageRegistry.create(); end
            obj.StageTitles = cellfun(@(s) string(s.title()), stages);
            obj.StageRequires = cellfun(@(s) s.requires(), stages, 'UniformOutput', false);
            obj.StageNeedsSpeech = cellfun(@(s) s.needsRecognition(), stages);
            obj.StageEnabled = false(1, numel(stages));
            obj.Settings = OceanViz.App.NLP.SpeechStudio.Models.StudioSession.defaultSettings();
            obj.Listeners = [addlistener(session, 'Context', 'PostSet', @(~, ~) obj.onContext()), ...
                addlistener(session, 'Input', 'PostSet', @(~, ~) obj.onInput())];
        end

        function delete(obj)
            delete(obj.Listeners);
            try
                if ~isempty(obj.Recorder) && isvalid(obj.Recorder) && obj.Recorder.isRecording()
                    obj.Recorder.stop();
                end
            catch
            end
        end

        %% ---- initial population ---------------------------------------------
        function initialize(obj)
            % INITIALIZE  Fill the model list (ModelCatalog) and FSDD splits (manifest).
            T = obj.Session.models();
            obj.ModelNames = reshape(string(T.Name), 1, []);
            obj.Devices = reshape(obj.Recorder.devices(), 1, []);
            if ~isempty(obj.Devices), obj.Device = obj.Devices(1); end
            m = obj.Session.manifest();
            sp = unique(m.Entries.Split, 'stable');
            obj.Splits = reshape(sp(sp ~= ""), 1, []);
            if any(obj.Splits == "test"), obj.setSplit("test"); else, obj.setSplit(obj.Splits(1)); end
            if ~isempty(obj.ModelNames)
                pref = obj.ModelNames(startsWith(obj.ModelNames, "ctc"));
                if isempty(pref), pref = obj.ModelNames(1); end
                obj.selectModel(pref(1));
            end
            if ~isempty(obj.Utterances), obj.selectUtterance(obj.Utterances(1)); end
        end

        %% ---- commands --------------------------------------------------------
        function setMode(obj, mode)
            arguments
                obj
                mode (1,1) string {mustBeMember(mode, ["Pipeline", "Workflow"])}
            end
            obj.Mode = mode;
        end

        function setSource(obj, source)
            arguments
                obj
                source (1,1) string {mustBeMember(source, ["FSDD", "WAV", "Microphone"])}
            end
            if obj.Source == source, return; end
            obj.Source = source;
            % Switching back to a source restores its remembered input: re-picking the
            % already-selected dropdown item fires no ValueChanged event, so without this the
            % user would be stuck with "No FSDD utterance selected" (critic round 2, N2).
            kinds = struct('FSDD', "fsdd", 'WAV', "wav", 'Microphone', "microphone");
            if obj.Session.InputKind ~= kinds.(source)
                if source == "FSDD" && strlength(obj.UtteranceId) > 0
                    obj.guard(@() obj.Session.useFSDD(obj.UtteranceId));
                    obj.ReferenceText = obj.Session.Reference;
                elseif source == "WAV" && strlength(obj.WavFile) > 0 && isfile(obj.WavFile)
                    obj.guard(@() obj.Session.useWav(obj.WavFile, ""));
                end
            end
            obj.updateInputReady();
            obj.markStale();
        end

        function setSplit(obj, split)
            arguments
                obj
                split (1,1) string
            end
            m = obj.Session.manifest();
            obj.Split = split;
            obj.Utterances = reshape(m.Entries.Id(m.Entries.Split == split), 1, []);
        end

        function selectUtterance(obj, id)
            arguments
                obj
                id (1,1) string
            end
            obj.guard(@() obj.Session.useFSDD(id));
            obj.UtteranceId = id;
            obj.Source = "FSDD";
            obj.updateInputReady();
            obj.ReferenceText = obj.Session.Reference;
        end

        function id = randomHeldOut(obj, seed)
            % RANDOMHELDOUT  A random test-split (held-out speaker) utterance; SEED optional.
            arguments
                obj
                seed double = []
            end
            m = obj.Session.manifest();
            ids = m.Entries.Id(m.Entries.Split == "test");
            if isempty(seed)
                if isempty(obj.Rng), obj.Rng = RandStream('mt19937ar', 'Seed', 'shuffle'); end
                s = obj.Rng;                     % private stream: global RNG state untouched
            else
                s = RandStream('mt19937ar', 'Seed', seed);
            end
            k = randi(s, numel(ids));
            obj.setSplit("test");
            id = ids(k);
            obj.selectUtterance(id);
        end

        function setWavFile(obj, file)
            arguments
                obj
                file (1,1) string
            end
            obj.guard(@() obj.Session.useWav(file, ""));
            obj.WavFile = file;
            obj.Source = "WAV";
            obj.updateInputReady();
            obj.ReferenceText = "";
        end

        function setReference(obj, text)
            % SETREFERENCE  Typed reference for WAV / microphone input (FSDD has its own).
            arguments
                obj
                text (1,1) string
            end
            obj.Session.setReference(text);
            obj.ReferenceText = text;
            obj.markStale();
        end

        function startRecording(obj)
            obj.guard(@() obj.Recorder.start(obj.Device, obj.MaxRecordSeconds));
            obj.IsRecording = obj.Recorder.isRecording();
            if obj.IsRecording, obj.StatusText = "Recording... press Stop when done."; end
        end

        function pollLevel(obj)
            if obj.Recorder.isRecording()
                obj.InputLevelDBFS = obj.Recorder.level();
            elseif obj.IsRecording
                obj.stopRecording();            % reached the maximum duration
            end
        end

        function stopRecording(obj)
            if ~obj.IsRecording && ~obj.Recorder.isRecording(), return; end
            [x, fs] = obj.Recorder.stop();
            obj.IsRecording = false;
            obj.InputLevelDBFS = -120;
            obj.guard(@() obj.Session.useSamples(x, fs, sprintf("Microphone (%.2f s at %g Hz)", numel(x) / fs, fs)));
            obj.Source = "Microphone";
            obj.updateInputReady();
            obj.ReferenceText = "";
        end

        function setDevice(obj, name)
            obj.Device = string(name);
        end

        function setMaxRecordSeconds(obj, s)
            arguments
                obj
                s (1,1) double {mustBePositive}
            end
            obj.MaxRecordSeconds = s;
        end

        function selectModel(obj, name)
            arguments
                obj
                name (1,1) string
            end
            obj.IsBusy = true;
            c = onCleanup(@() obj.clearBusy());
            obj.guard(@() obj.Session.selectModel(name));
            obj.ModelName = obj.Session.ModelName;
            obj.Capabilities = obj.Session.capabilities();
            obj.ModelCard = obj.modelCard();
            if any(obj.Capabilities == "attention")      % CTC-only controls do not apply
                s = obj.Settings; s.LMWeight = 0; s.Bonus = 0; obj.Settings = s;
            end
            if obj.Session.Reference ~= "", obj.ReferenceText = obj.Session.Reference; end
            obj.markStale();
        end

        function setSetting(obj, name, value)
            % SETSETTING  One decoder / robustness setting (see StudioSession.defaultSettings).
            arguments
                obj
                name (1,1) string
                value
            end
            s = obj.Settings;
            if ~isfield(s, name)
                error('OceanViz:App:NLP:SpeechStudio:StudioViewModel:UnknownSetting', 'Unknown setting "%s".', name);
            end
            if isstring(s.(name)) || ischar(value), value = string(value); end
            if isequal(s.(name), value), return; end
            s.(name) = value;
            obj.Settings = s;
            obj.markStale();
        end

        function resetSettings(obj)
            % RESETSETTINGS  Back to StudioSession.defaultSettings (library defaults).
            obj.Settings = OceanViz.App.NLP.SpeechStudio.Models.StudioSession.defaultSettings();
            obj.markStale();
        end

        function ok = run(obj, progressFcn)
            % RUN  Whole pipeline once; on success every applicable stage is enabled.
            arguments
                obj
                progressFcn = []
            end
            ok = false;
            if obj.IsRecording, obj.stopRecording(); end
            if ~obj.InputReady
                obj.StatusText = "Cannot run: " + obj.missingInputText();
                return;
            end
            obj.IsBusy = true;
            c = onCleanup(@() obj.clearBusy());
            obj.StatusText = "Running the pipeline...";
            try
                obj.Session.run(obj.Settings, progressFcn);
                ok = true;
            catch err
                obj.StatusText = "Run failed: " + err.message;
                obj.Warnings = "Run failed: " + string(err.message);
            end
        end

        function selectStage(obj, k)
            arguments
                obj
                k (1,1) double {mustBeInteger, mustBePositive}
            end
            if k > numel(obj.StageTitles) || ~obj.StageEnabled(k), return; end
            obj.CurrentStage = k;
        end

        function next(obj)
            k = find(obj.StageEnabled & (1:numel(obj.StageEnabled)) > obj.CurrentStage, 1);
            if ~isempty(k), obj.CurrentStage = k; end
        end

        function back(obj)
            k = find(obj.StageEnabled & (1:numel(obj.StageEnabled)) < obj.CurrentStage, 1, 'last');
            if ~isempty(k), obj.CurrentStage = k; end
        end

        function selectSegment(obj, k)
            arguments
                obj
                k (1,1) double {mustBeInteger, mustBePositive}
            end
            if isempty(obj.Context) || k > obj.NumSegments, return; end
            obj.Context.SegmentIndex = k;
            obj.SegmentIndex = k;
        end

        function setCursor(obj, t)
            obj.CursorTime = t;
        end

        function tf = canGoNext(obj)
            tf = any(obj.StageEnabled & (1:numel(obj.StageEnabled)) > obj.CurrentStage);
        end

        function tf = canGoBack(obj)
            tf = any(obj.StageEnabled & (1:numel(obj.StageEnabled)) < obj.CurrentStage);
        end

        function tf = showControl(obj, name)
            % SHOWCONTROL  Whether a decoder control applies to the selected model / decoder.
            caps = obj.Capabilities;
            beam = obj.Settings.Decoder == "beam";
            switch string(name)
                case "BeamWidth", tf = beam;
                case {"LMWeight", "Bonus", "LMOrder"}, tf = any(caps == "lm") && beam;
                case "CTCRescoreWeight", tf = any(caps == "ctc-rescore") && beam;
                otherwise, tf = true;
            end
        end
    end

    methods (Access = private)
        function clearBusy(obj)
            if isvalid(obj), obj.IsBusy = false; end
        end

        function markStale(obj)
            if obj.HasResult
                obj.IsStale = true;
                obj.StatusText = "Settings changed: press Run to recompute (stages show the previous run).";
            end
            obj.updateStageEnabled();
        end

        function onInput(obj)
            obj.updateInputReady();
            obj.markStale();
        end

        function updateInputReady(obj)
            % The input is usable only when it came from the selected source.
            kind = struct('FSDD', "fsdd", 'WAV', "wav", 'Microphone', "microphone");
            in = obj.Session.Input;
            obj.InputReady = ~isempty(in) && obj.Session.InputKind == kind.(obj.Source);
            if obj.InputReady
                obj.InputSummary = sprintf("%s: %.3f s, %g Hz, %d ch", obj.Session.InputLabel, ...
                    in.duration(), in.SampleRate, in.numChannels());
            else
                obj.InputSummary = obj.missingInputText();
            end
        end

        function s = missingInputText(obj)
            switch obj.Source
                case "FSDD", s = "No FSDD utterance selected: choose one.";
                case "WAV", s = "No WAV file chosen for this source: press Browse...";
                otherwise, s = "Nothing recorded yet: press Record, then Stop.";
            end
        end

        function onContext(obj)
            ctx = obj.Session.Context;
            obj.Context = ctx;
            if isempty(ctx), return; end
            obj.HasResult = true;
            obj.IsStale = false;
            obj.NoSpeech = ~ctx.hasPieces();
            obj.NumSegments = ctx.numSegments();
            obj.SegmentIndex = ctx.SegmentIndex;
            r = ctx.Result;
            obj.Warnings = ctx.Notes;
            tm = r.Timings;
            total = ctx.RunSeconds;
            obj.RunSummary = sprintf("%s | %s | %g Hz | %.2f s | run %.2f s", ctx.ModelName, ...
                ctx.InputLabel, r.Conditioned.SampleRate, r.Conditioned.duration(), total);
            if isstruct(tm) && isscalar(tm) && ~isempty(fieldnames(tm))
                f = string(fieldnames(tm))';
                parts = arrayfun(@(n) sprintf("%s %.2f s", n, tm.(n)), f(structfun(@(x) isnumeric(x) && isscalar(x), tm)'));
                if ~isempty(parts), obj.RunSummary = obj.RunSummary + " (" + strjoin(parts, ", ") + ")"; end
            end
            if obj.NoSpeech
                obj.StatusText = "No speech detected: nothing was recognised (see stage 1 for the detector evidence).";
            else
                obj.StatusText = "Transcript: """ + r.Transcript.Text + """";
            end
            obj.updateStageEnabled();
            if ~obj.StageEnabled(obj.CurrentStage), obj.CurrentStage = 1; end
        end

        function updateStageEnabled(obj)
            n = numel(obj.StageTitles);
            en = false(1, n);
            if obj.HasResult && ~obj.IsStale
                for k = 1:n
                    req = obj.StageRequires{k};
                    en(k) = all(arrayfun(@(c) any(obj.Capabilities == c), req));
                end
                if obj.NoSpeech, en = en & ~obj.StageNeedsSpeech; end
            end
            obj.StageEnabled = en;
        end

        function s = modelCard(obj)
            row = obj.Session.modelRow();
            if isempty(row), s = ""; return; end
            v = @(n) obj.cardValue(row, n);
            s = sprintf("%s | %s params | best epoch %s | val WER %s | %s Hz | units %s", v("Type"), ...
                v("NumParams"), v("BestEpoch"), v("BestValWER"), v("SampleRate"), v("Units"));
        end

        function guard(obj, fn)
            try
                fn();
            catch err
                obj.StatusText = "Error: " + err.message;
                rethrow(err);
            end
        end
    end

    methods (Static, Access = private)
        function s = cardValue(row, name)
            s = "?";
            if ~ismember(name, row.Properties.VariableNames), return; end
            x = row.(name)(1);
            if iscell(x), x = x{1}; end
            if isnumeric(x)
                if name == "NumParams"
                    if x >= 1e6, s = sprintf("%.2f M", x / 1e6); else, s = sprintf("%.1f k", x / 1e3); end
                else
                    s = sprintf("%.4g", x);
                end
            else
                s = string(x);
            end
        end
    end
end
