classdef ConsoleViewModel < handle
    % CONSOLEVIEWMODEL  Bindable state + commands of the ASR Console (no UI code).
    %
    %   vm = OceanViz.App.NLP.ASRConsole.ViewModels.ConsoleViewModel( ...
    %            'Client', client, 'RecorderFactory', @(fs) rec, 'Player', @(x, fs) [], 'SaveFolder', dir)
    %   Commands: connect, setServiceUrl, refreshDevices, setDevice, setSampleRate, setThreads,
    %   setMaxSeconds, setUse, setBackend, startRecording, pollLevel, stopRecording, loadWav, setAudio, play,
    %   transcribe, save.
    %
    %   Every command catches its own failures: the message goes to StatusText and Log and the
    %   console stays usable (a model that fails reports its error in its own Results row).
    %   The recorder is created by RecorderFactory(sampleRate) whenever the rate changes.

    properties (SetObservable)
        ServiceUrl (1,1) string = "http://127.0.0.1:8009"
        ServiceOnline (1,1) logical = false
        ServiceText (1,1) string = "Not connected"
        HardwareText (1,1) string = "Connect to the service to see its hardware."
        ModelTable table = OceanViz.App.NLP.ASRConsole.ViewModels.ConsoleViewModel.emptyModelTable()
        Devices (1,:) string = strings(1, 0)
        Device (1,1) string = ""
        SampleRates (1,:) double = [8000 16000 22050 44100 48000]
        SampleRate (1,1) double = 16000
        Threads (1,1) double = 4
        MaxThreads (1,1) double = 4
        MaxRecordSeconds (1,1) double = 15
        IsRecording (1,1) logical = false
        InputLevelDBFS (1,1) double = -120
        Samples (:,1) double = zeros(0, 1)
        AudioRate (1,1) double = 16000
        AudioLabel (1,1) string = "No audio yet: record or load a WAV file."
        PeakDBFS (1,1) double = -120
        SpeechOk (1,1) logical = false     % gate 2: the audio contains real speech (Models.SpeechCheck)
        SpeechText (1,1) string = ""       % why not, when SpeechOk is false
        Results table = OceanViz.App.NLP.ASRConsole.ViewModels.ConsoleViewModel.emptyResults()
        History table = OceanViz.App.NLP.ASRConsole.ViewModels.ConsoleViewModel.emptyHistory()
        Log (:,1) string = strings(0, 1)
        IsBusy (1,1) logical = false
        StatusText (1,1) string = "Press Connect to reach the recognition service."
        LastSaved (1,1) string = ""
    end

    properties (SetAccess = private)
        Client
        Recorder
        RecorderFactory
        Player
        Store
    end

    methods
        function obj = ConsoleViewModel(opts)
            arguments
                opts.Client = []
                opts.RecorderFactory = []
                opts.Player = []
                opts.SaveFolder (1,1) string = OceanViz.App.NLP.ASRConsole.ViewModels.ConsoleViewModel.defaultSaveFolder()
            end
            obj.Client = opts.Client;
            if isempty(obj.Client)
                obj.Client = OceanViz.App.NLP.ASRConsole.Models.RecognitionClient();
            end
            obj.ServiceUrl = obj.Client.BaseUrl;
            obj.RecorderFactory = opts.RecorderFactory;
            if isempty(obj.RecorderFactory)
                obj.RecorderFactory = @(fs) OceanViz.App.NLP.SpeechStudio.Models.MicrophoneRecorder('SampleRate', fs);
            end
            obj.Player = opts.Player;
            if isempty(obj.Player), obj.Player = @(x, fs) sound(x, fs); end
            obj.Store = OceanViz.App.NLP.ASRConsole.Models.RecordingStore(opts.SaveFolder);
            obj.MaxThreads = max(1, feature('numcores'));
            obj.Threads = min(4, obj.MaxThreads);
            obj.Recorder = obj.RecorderFactory(obj.SampleRate);
            obj.refreshDevices();
        end

        function delete(obj)
            try
                if ~isempty(obj.Recorder) && isvalid(obj.Recorder) && obj.Recorder.isRecording()
                    obj.Recorder.stop();
                end
            catch
            end
        end

        % ---- service -------------------------------------------------------------------------
        function setServiceUrl(obj, url)
            obj.ServiceUrl = strip(string(url));
            obj.Client.BaseUrl = obj.ServiceUrl;
            obj.ServiceOnline = false;
            obj.ServiceText = "Not connected";
        end

        function ok = connect(obj)
            % CONNECT  health -> models -> hardware; keeps the user's model choices when possible.
            obj.Client.BaseUrl = obj.ServiceUrl;
            ok = false;
            try
                h = obj.Client.health();
                obj.ModelTable = obj.mergeModels(obj.Client.models());
                obj.HardwareText = obj.formatHardware(obj.Client.hardware());
                obj.ServiceOnline = true;
                obj.ServiceText = sprintf("Connected: service %s", h.service_version);
                nAvail = sum(obj.ModelTable.Available);
                obj.say(sprintf("Connected to %s: %d of %d models available.", obj.ServiceUrl, ...
                    nAvail, height(obj.ModelTable)));
                ok = true;
            catch err
                obj.ServiceOnline = false;
                obj.ServiceText = "Offline";
                obj.say("Service not reachable. Start it with `uv run asr serve` (or `just up a10`). " + ...
                    string(err.message), true);
            end
        end

        % ---- microphone / settings -----------------------------------------------------------
        function refreshDevices(obj)
            obj.Devices = obj.Recorder.devices();
            if ~any(obj.Devices == obj.Device)
                obj.Device = "";
                if ~isempty(obj.Devices), obj.Device = obj.Devices(1); end
            end
        end

        function setDevice(obj, name)
            obj.Device = string(name);
        end

        function setSampleRate(obj, fs)
            if obj.IsRecording, return; end
            obj.SampleRate = fs;
            obj.Recorder = obj.RecorderFactory(fs);
            obj.refreshDevices();
        end

        function setMaxSeconds(obj, s)
            obj.MaxRecordSeconds = max(1, min(60, s));
        end

        function setThreads(obj, n)
            obj.Threads = max(1, min(obj.MaxThreads, round(n)));
        end

        function setUse(obj, model, tf)
            k = obj.row(model);
            t = obj.ModelTable;
            t.Use(k) = logical(tf) && t.Available(k);
            obj.ModelTable = t;
        end

        function setBackend(obj, model, backend)
            k = obj.row(model);
            t = obj.ModelTable;
            if ~any(t.Backends{k} == string(backend))
                error('OceanViz:App:NLP:ASRConsole:Backend', 'Model "%s" has no backend "%s".', model, backend);
            end
            t.Backend(k) = string(backend);
            obj.ModelTable = t;
        end

        % ---- audio -----------------------------------------------------------------------------
        function startRecording(obj)
            if obj.IsRecording, return; end
            obj.refreshDevices();              % gate 1: never start (or send anything) without an input device
            if isempty(obj.Devices)
                obj.say("No microphone found: connect one (or allow microphone access in Windows Settings), " + ...
                    "then press Record again.", true);
                return
            end
            try
                obj.Recorder.start(obj.Device, obj.MaxRecordSeconds);
                obj.IsRecording = true;
                obj.say(sprintf("Recording at %g Hz (stops by itself after %g s)...", ...
                    obj.SampleRate, obj.MaxRecordSeconds));
            catch err
                obj.say("Could not start the microphone: " + string(err.message), true);
            end
        end

        function pollLevel(obj)
            % POLLLEVEL  Called ~10 times a second by the view while recording.
            if ~obj.IsRecording, return; end
            obj.InputLevelDBFS = obj.Recorder.level();
            if ~obj.Recorder.isRecording()      % the recorder reached MaxRecordSeconds
                obj.stopRecording();
            end
        end

        function stopRecording(obj)
            if ~obj.IsRecording, return; end
            obj.IsRecording = false;
            obj.InputLevelDBFS = -120;
            try
                [x, fs] = obj.Recorder.stop();
                obj.setAudio(x, fs, "Microphone: " + obj.Device);
            catch err
                obj.say("Recording failed: " + string(err.message), true);
            end
        end

        function loadWav(obj, file)
            try
                [x, fs] = audioread(file);
                [~, name, ext] = fileparts(file);
                obj.setAudio(mean(x, 2), fs, "File: " + name + ext);
            catch err
                obj.say("Could not read " + string(file) + ": " + string(err.message), true);
            end
        end

        function setAudio(obj, x, fs, label)
            arguments
                obj
                x {mustBeNumeric}
                fs (1,1) double {mustBePositive}
                label (1,1) string = "Audio"
            end
            obj.AudioRate = fs;
            obj.Samples = double(x(:));
            obj.PeakDBFS = 20 * log10(max(max(abs(obj.Samples), [], 'all'), 1e-6));
            obj.AudioLabel = sprintf("%s  (%.2f s, %g Hz, peak %.1f dBFS)", label, ...
                numel(obj.Samples) / fs, fs, obj.PeakDBFS);
            obj.Results = obj.emptyResults();
            check = OceanViz.App.NLP.ASRConsole.Models.SpeechCheck.assess(obj.Samples, fs);   % gate 2
            obj.SpeechText = check.Reason;
            obj.SpeechOk = check.Usable;
            if ~check.Usable
                obj.say(obj.AudioLabel + ". " + check.Reason, true);
                return
            end
            obj.say(obj.AudioLabel + sprintf(". %.1f s of speech. Press Transcribe.", check.SpeechSeconds));
            if obj.PeakDBFS > -0.1
                obj.say("The recording clips (peak at full scale): speak further from the microphone.", true);
            end
        end

        function play(obj)
            if isempty(obj.Samples), obj.say("Nothing to play yet.", true); return; end
            obj.Player(obj.Samples, obj.AudioRate);
        end

        % ---- recognition -----------------------------------------------------------------------
        function transcribe(obj)
            if isempty(obj.Samples), obj.say("Record or load audio first.", true); return; end
            if ~obj.SpeechOk                   % gate 3: never call the service on audio without speech
                obj.say(obj.SpeechText, true);
                return
            end
            t = obj.ModelTable;
            use = t.Use & t.Available;
            if ~any(use), obj.say("Tick at least one available model.", true); return; end
            choices = struct('model', cellstr(t.Model(use))', 'backend', cellstr(t.Backend(use))');
            obj.IsBusy = true;
            busy = onCleanup(@() obj.setIdle());
            obj.StatusText = sprintf("Transcribing with %d model(s)...", nnz(use));
            try
                r = obj.Client.recognize(obj.Samples, obj.AudioRate, choices, obj.Threads);
            catch err
                obj.ServiceOnline = false;
                obj.ServiceText = "Offline";
                obj.say("Transcription failed: " + string(err.message), true);
                return;
            end
            obj.Results = obj.resultsTable(r.results);
            obj.History = [obj.History; obj.historyRows(r.results)];
            bad = obj.Results.Error ~= "";
            msg = sprintf("Done: %d model(s) answered", nnz(~bad));
            if any(bad), msg = msg + sprintf(", %d failed (see their rows)", nnz(bad)); end
            obj.say(msg + ".", any(bad));
        end

        function file = save(obj)
            file = "";
            if isempty(obj.Samples), obj.say("Nothing to save yet.", true); return; end
            try
                res = table2struct(obj.Results);
                meta = struct('source', char(obj.AudioLabel), 'service', char(obj.ServiceUrl));
                file = obj.Store.save(obj.Samples, obj.AudioRate, res, meta);
                obj.LastSaved = file;
                obj.say("Saved " + file + " (+ .json with the transcripts).");
            catch err
                obj.say("Save failed: " + string(err.message), true);
            end
        end
    end

    methods (Static)
        function t = emptyModelTable()
            t = table('Size', [0 9], 'VariableTypes', ["string", "string", "logical", "string", "cell", ...
                "logical", "string", "string", "string"], 'VariableNames', ["Model", "Title", "Use", ...
                "Backend", "Backends", "Available", "Version", "ModelName", "Reason"]);
        end

        function t = emptyResults()
            t = table('Size', [0 7], 'VariableTypes', ["string", "string", "string", "string", "double", ...
                "double", "string"], 'VariableNames', ["Model", "Backend", "Version", "Text", "LatencyMs", ...
                "RTF", "Error"]);
        end

        function t = emptyHistory()
            t = table('Size', [0 7], 'VariableTypes', ["datetime", "string", "double", "string", "string", ...
                "string", "double"], 'VariableNames', ["Time", "Audio", "Seconds", "Model", "Backend", ...
                "Text", "LatencyMs"]);
        end

        function folder = defaultSaveFolder()
            here = fileparts(mfilename('fullpath'));
            root = fileparts(fileparts(fileparts(fileparts(fileparts(here)))));   % ...\Matlab code practice
            folder = string(fullfile(root, 'Results', 'ASRConsole', 'recordings'));
        end

        function s = formatHardware(hw)
            % FORMATHARDWARE  The service's probe facts as readable lines.
            lines = strings(0, 1);
            lines(end + 1) = sprintf("Host: %s (%s)", string(hw.hostname), string(hw.platform));
            lines(end + 1) = sprintf("CPU: %d logical cores;  RAM: %.1f GB", hw.cpu_logical, hw.ram_gb);
            lines(end + 1) = sprintf("PyTorch: %s", string(hw.torch_version));
            gpus = hw.gpus;
            if iscell(gpus), gpus = [gpus{:}]; end
            if isempty(gpus)
                lines(end + 1) = "GPU: none visible to the service";
            end
            for k = 1:numel(gpus)
                g = gpus(k);
                if g.torch_has_kernels
                    lines(end + 1) = sprintf("GPU %d: %s - usable by PyTorch", k - 1, string(g.name)); %#ok<AGROW>
                else
                    lines(end + 1) = sprintf("GPU %d: %s - NOT usable by PyTorch (%s)", k - 1, ...
                        string(g.name), string(g.kernel_note)); %#ok<AGROW>
                end
            end
            if isfield(hw, 'onnxruntime_providers')
                lines(end + 1) = sprintf("ONNX Runtime %s providers: %s", string(hw.onnxruntime_version), ...
                    strjoin(string(hw.onnxruntime_providers), ", "));
            end
            s = strjoin(lines, newline);
        end
    end

    methods (Access = private)
        function t = mergeModels(obj, m)
            t = obj.emptyModelTable();
            old = obj.ModelTable;
            for k = 1:numel(m)
                use = m(k).available;
                backend = m(k).default_backend;
                j = find(old.Model == m(k).model, 1);
                if ~isempty(j)                                   % keep the user's choices
                    use = old.Use(j) && m(k).available;
                    if any(m(k).backends == old.Backend(j)), backend = old.Backend(j); end
                end
                t(end + 1, :) = {m(k).model, m(k).title, use, backend, {m(k).backends}, m(k).available, ...
                    m(k).model_version, m(k).model_name, m(k).reason}; %#ok<AGROW>
            end
        end

        function k = row(obj, model)
            k = find(obj.ModelTable.Model == string(model), 1);
            if isempty(k)
                error('OceanViz:App:NLP:ASRConsole:UnknownModel', 'Unknown model "%s".', model);
            end
        end

        function t = resultsTable(obj, r)
            t = obj.emptyResults();
            for k = 1:numel(r)
                ver = r(k).model_name;
                if r(k).model_version ~= "", ver = ver + " v" + r(k).model_version; end
                t(end + 1, :) = {r(k).model, r(k).backend, strip(ver), r(k).text, r(k).latency_ms, ...
                    r(k).rtf, r(k).error}; %#ok<AGROW>
            end
        end

        function t = historyRows(obj, r)
            t = obj.emptyHistory();
            stamp = datetime('now');
            for k = 1:numel(r)
                txt = r(k).text;
                if r(k).error ~= "", txt = "ERROR: " + r(k).error; end
                t(end + 1, :) = {stamp, obj.AudioLabel, numel(obj.Samples) / obj.AudioRate, r(k).model, ...
                    r(k).backend, txt, r(k).latency_ms}; %#ok<AGROW>
            end
        end

        function setIdle(obj)
            if isvalid(obj), obj.IsBusy = false; end
        end

        function say(obj, msg, isProblem)
            if nargin < 3, isProblem = false; end
            msg = string(msg);
            obj.StatusText = msg;
            prefix = string(datetime('now', 'Format', 'HH:mm:ss'));
            if isProblem, prefix = prefix + " !"; end
            obj.Log = [obj.Log; prefix + "  " + msg];
        end
    end
end
