classdef ConsoleView < handle
    % CONSOLEVIEW  ASR Console window (Explorer layout): fixed sidebar of controls on the left,
    %   waveform + level meter, per-model results and History / Hardware / Log tabs on the right.
    %   All structure is uigridlayout; the view only builds components and binds them to the
    %   ConsoleViewModel (listeners on its SetObservable properties).
    %
    %   v = OceanViz.App.NLP.ASRConsole.Views.ConsoleView(viewModel, 'Visible', true)
    %   v.close()     deletes the window, timers, listeners and the ViewModel

    properties (SetAccess = private)
        Figure
        ViewModel
        Viz
        UrlField
        ServiceLamp
        ServiceLabel
        ConnectButton
        DeviceDropDown
        RateDropDown
        MaxSecondsSpinner
        ModelsGrid
        ThreadsSpinner
        RecordButton
        StopButton
        PlayButton
        LoadButton
        TranscribeButton
        SaveButton
        AudioLabel
        WaveAxes
        LevelGauge
        ResultsTable
        HistoryTable
        HardwareArea
        LogArea
        StatusLabel
        ModelControls struct = struct('Model', {}, 'Check', {}, 'Backend', {})
    end

    properties (Access = private)
        Listeners = event.listener.empty
        LevelTimer = []
        ModelsKey (1,1) string = "<none>"
    end

    methods
        function obj = ConsoleView(viewModel, opts)
            arguments
                viewModel (1,1) OceanViz.App.NLP.ASRConsole.ViewModels.ConsoleViewModel
                opts.Visible (1,1) logical = true
                opts.Theme (1,1) string = "Light"
            end
            obj.ViewModel = viewModel;
            obj.Viz = OceanViz.Core('Theme', opts.Theme);
            obj.buildUi();
            obj.bind();
            obj.syncAll();
            obj.Figure.Visible = matlab.lang.OnOffSwitchState(opts.Visible);
        end

        function close(obj)
            obj.stopLevelTimer();
            delete(obj.Listeners);
            obj.Listeners = event.listener.empty;
            if ~isempty(obj.Figure) && isvalid(obj.Figure), delete(obj.Figure); end
            if ~isempty(obj.ViewModel) && isvalid(obj.ViewModel), delete(obj.ViewModel); end
        end

        function delete(obj)
            obj.stopLevelTimer();
            delete(obj.Listeners);
        end

        function transcribe(obj)
            % TRANSCRIBE  Busy pointer + disabled buttons while the service answers.
            obj.Figure.Pointer = 'watch';
            drawnow;
            restore = onCleanup(@() obj.setPointer('arrow'));
            obj.ViewModel.transcribe();
        end
    end

    methods (Access = private)
        % ---- layout --------------------------------------------------------------------------------
        function buildUi(obj)
            obj.Figure = uifigure('Name', 'ASR Console', 'Position', [60 60 1280 820], 'Visible', 'off', ...
                'CloseRequestFcn', @(~, ~) obj.close());
            root = uigridlayout(obj.Figure, [2 2], 'ColumnWidth', {270, '1x'}, 'RowHeight', {'1x', 22}, ...
                'Padding', [0 0 0 0], 'ColumnSpacing', 0, 'RowSpacing', 0);
            obj.buildSidebar(root);
            obj.buildMain(root);
            obj.StatusLabel = uilabel(root, 'Text', '', 'FontAngle', 'italic');
            obj.StatusLabel.Layout.Row = 2; obj.StatusLabel.Layout.Column = [1 2];
        end

        function buildSidebar(obj, root)
            vm = obj.ViewModel;
            p = uipanel(root, 'BorderType', 'none');
            p.Layout.Row = 1; p.Layout.Column = 1;
            g = uigridlayout(p, [17 2], 'ColumnWidth', {'1x', '1x'}, 'Padding', [10 10 10 10], ...
                'RowSpacing', 6, 'ColumnSpacing', 6, 'Scrollable', 'on', ...
                'RowHeight', {'fit', 'fit', 'fit', 'fit', 'fit', 'fit', 'fit', 'fit', 'fit', 'fit', 'fit', ...
                'fit', 'fit', 'fit', 'fit', 'fit', '1x'});
            r = 1;
            obj.header(g, "SERVICE", r); r = r + 1;
            obj.UrlField = uieditfield(g, 'text', 'Value', char(vm.ServiceUrl), ...
                'Tooltip', 'Address of the local recognition service (asr serve / just up a10)', ...
                'ValueChangedFcn', @(f, ~) vm.setServiceUrl(f.Value));
            obj.place(obj.UrlField, r, [1 2]); r = r + 1;
            sg = uigridlayout(g, [1 3], 'ColumnWidth', {16, '1x', 80}, 'Padding', [0 0 0 0], 'ColumnSpacing', 6);
            obj.place(sg, r, [1 2]); r = r + 1;
            obj.ServiceLamp = uilamp(sg, 'Color', [0.6 0.6 0.6]);
            obj.ServiceLabel = uilabel(sg, 'Text', 'Not connected');
            obj.ConnectButton = uibutton(sg, 'Text', 'Connect', 'Tooltip', 'Check the service, list models and hardware', ...
                'ButtonPushedFcn', @(~, ~) obj.withPointer(@() vm.connect()));

            obj.header(g, "MICROPHONE", r); r = r + 1;
            obj.label(g, "Device", r, 1);
            obj.DeviceDropDown = uidropdown(g, 'Items', {''}, 'ValueChangedFcn', @(d, ~) vm.setDevice(d.Value));
            obj.place(obj.DeviceDropDown, r, 2); r = r + 1;
            obj.label(g, "Sample rate (Hz)", r, 1);
            obj.RateDropDown = uidropdown(g, 'Items', cellstr(string(vm.SampleRates)), 'ItemsData', vm.SampleRates, ...
                'Value', vm.SampleRate, 'Tooltip', 'Capture rate; the service resamples to 16 kHz', ...
                'ValueChangedFcn', @(d, ~) vm.setSampleRate(d.Value));
            obj.place(obj.RateDropDown, r, 2); r = r + 1;
            obj.label(g, "Max seconds", r, 1);
            obj.MaxSecondsSpinner = uispinner(g, 'Limits', [1 60], 'Step', 1, 'Value', vm.MaxRecordSeconds, ...
                'ValueChangedFcn', @(s, ~) vm.setMaxSeconds(s.Value));
            obj.place(obj.MaxSecondsSpinner, r, 2); r = r + 1;

            obj.header(g, "MODELS", r); r = r + 1;
            obj.ModelsGrid = uigridlayout(g, [1 2], 'ColumnWidth', {'1x', 96}, 'RowHeight', {'fit'}, ...
                'Padding', [0 0 0 0], 'RowSpacing', 4, 'ColumnSpacing', 6);
            obj.place(obj.ModelsGrid, r, [1 2]); r = r + 1;

            obj.header(g, "HARDWARE", r); r = r + 1;
            obj.label(g, "CPU threads", r, 1);
            obj.ThreadsSpinner = uispinner(g, 'Limits', [1 vm.MaxThreads], 'Step', 1, 'Value', vm.Threads, ...
                'Tooltip', 'CPU threads the service uses for this request', ...
                'ValueChangedFcn', @(s, ~) vm.setThreads(s.Value));
            obj.place(obj.ThreadsSpinner, r, 2); r = r + 1;

            obj.header(g, "ACTIONS", r); r = r + 1;
            obj.RecordButton = obj.button(g, char("● Record"), r, 1, @() vm.startRecording());
            obj.StopButton = obj.button(g, char("■ Stop"), r, 2, @() vm.stopRecording()); r = r + 1;
            obj.PlayButton = obj.button(g, char("▶ Play"), r, 1, @() vm.play());
            obj.LoadButton = obj.button(g, 'Load WAV...', r, 2, @() obj.loadWav()); r = r + 1;
            obj.TranscribeButton = obj.button(g, 'Transcribe', r, 1, @() obj.transcribe());
            obj.TranscribeButton.FontWeight = 'bold';
            obj.SaveButton = obj.button(g, 'Save', r, 2, @() vm.save());
        end

        function buildMain(obj, root)
            m = uigridlayout(root, [3 1], 'RowHeight', {'1x', 170, '1x'}, 'Padding', [8 8 8 8], 'RowSpacing', 8);
            m.Layout.Row = 1; m.Layout.Column = 2;

            w = uigridlayout(m, [2 2], 'RowHeight', {'fit', '1x'}, 'ColumnWidth', {'1x', 70}, ...
                'Padding', [0 0 0 0], 'RowSpacing', 2);
            obj.AudioLabel = uilabel(w, 'Text', '', 'FontWeight', 'bold');
            obj.AudioLabel.Layout.Row = 1; obj.AudioLabel.Layout.Column = [1 2];
            obj.WaveAxes = uiaxes(w);
            obj.WaveAxes.Layout.Row = 2; obj.WaveAxes.Layout.Column = 1;
            lg = uigridlayout(w, [2 1], 'RowHeight', {'fit', '1x'}, 'Padding', [0 0 0 0]);
            lg.Layout.Row = 2; lg.Layout.Column = 2;
            uilabel(lg, 'Text', 'Level dBFS', 'HorizontalAlignment', 'center');
            obj.LevelGauge = uigauge(lg, 'linear', 'Orientation', 'vertical', 'Limits', [-60 0], ...
                'ScaleColors', [0.3 0.7 0.3; 0.95 0.75 0.1; 0.85 0.2 0.2], ...
                'ScaleColorLimits', [-60 -12; -12 -3; -3 0], 'Value', -60);

            obj.ResultsTable = uitable(m, 'ColumnName', {'Model', 'Backend', 'Version', 'Transcript', ...
                'Latency (ms)', 'RTF', 'Error'}, 'ColumnWidth', {90, 80, 150, '1x', 85, 60, 180}, ...
                'RowName', {});

            tg = uitabgroup(m);
            t1 = uitab(tg, 'Title', 'History');
            g1 = uigridlayout(t1, [1 1], 'Padding', [4 4 4 4]);
            obj.HistoryTable = uitable(g1, 'ColumnName', {'Time', 'Audio', 'Seconds', 'Model', 'Backend', ...
                'Transcript', 'Latency (ms)'}, 'ColumnWidth', {70, 220, 60, 80, 80, '1x', 85}, 'RowName', {});
            t2 = uitab(tg, 'Title', 'Service & hardware');
            g2 = uigridlayout(t2, [1 1], 'Padding', [4 4 4 4]);
            obj.HardwareArea = uitextarea(g2, 'Editable', 'off', 'FontName', 'Consolas');
            t3 = uitab(tg, 'Title', 'Log');
            g3 = uigridlayout(t3, [1 1], 'Padding', [4 4 4 4]);
            obj.LogArea = uitextarea(g3, 'Editable', 'off', 'FontName', 'Consolas');
        end

        % ---- binding -------------------------------------------------------------------------------
        function bind(obj)
            vm = obj.ViewModel;
            map = { ...
                {"ServiceOnline", "ServiceText"}, @() obj.syncService(); ...
                {"Devices", "Device"}, @() obj.syncDevices(); ...
                {"SampleRate", "MaxRecordSeconds", "Threads"}, @() obj.syncSettings(); ...
                {"ModelTable"}, @() obj.syncModels(); ...
                {"Samples", "AudioLabel"}, @() obj.syncAudio(); ...
                {"InputLevelDBFS"}, @() obj.syncLevel(); ...
                {"IsRecording", "IsBusy", "SpeechOk", "Devices", "Samples"}, @() obj.syncState(); ...
                {"Results"}, @() obj.syncResults(); ...
                {"History"}, @() obj.syncHistory(); ...
                {"HardwareText"}, @() obj.syncHardware(); ...
                {"Log", "StatusText"}, @() obj.syncLog()};
            for k = 1:size(map, 1)
                props = map{k, 1};
                fn = map{k, 2};
                for p = props
                    obj.Listeners(end + 1) = addlistener(vm, char(p{1}), 'PostSet', @(~, ~) obj.safe(fn));
                end
            end
        end

        function syncAll(obj)
            obj.syncService(); obj.syncDevices(); obj.syncSettings(); obj.syncModels(); obj.syncAudio();
            obj.syncLevel(); obj.syncState(); obj.syncResults(); obj.syncHistory(); obj.syncHardware();
            obj.syncLog();
        end

        function syncService(obj)
            vm = obj.ViewModel;
            obj.ServiceLabel.Text = char(vm.ServiceText);
            if vm.ServiceOnline
                obj.ServiceLamp.Color = [0.2 0.75 0.3];
            elseif vm.ServiceText == "Offline"
                obj.ServiceLamp.Color = [0.85 0.2 0.2];
            else
                obj.ServiceLamp.Color = [0.6 0.6 0.6];
            end
        end

        function syncDevices(obj)
            vm = obj.ViewModel;
            items = cellstr(vm.Devices);
            if isempty(items), items = {'(no input device)'}; end
            obj.DeviceDropDown.Items = items;
            if vm.Device ~= "", obj.DeviceDropDown.Value = char(vm.Device); end
        end

        function syncSettings(obj)
            vm = obj.ViewModel;
            obj.RateDropDown.Value = vm.SampleRate;
            obj.MaxSecondsSpinner.Value = vm.MaxRecordSeconds;
            obj.ThreadsSpinner.Value = vm.Threads;
        end

        function syncModels(obj)
            t = obj.ViewModel.ModelTable;
            key = strjoin(t.Model + "|" + string(t.Available) + "|" + t.Version + "|" + ...
                string(cellfun(@(b) strjoin(b, ","), t.Backends, 'UniformOutput', false)), ";");
            if key ~= obj.ModelsKey                       % structure changed: rebuild the rows
                obj.ModelsKey = key;
                delete(obj.ModelsGrid.Children);
                obj.ModelControls = struct('Model', {}, 'Check', {}, 'Backend', {});
                if height(t) == 0
                    uilabel(obj.ModelsGrid, 'Text', 'Connect to list the models.', 'FontAngle', 'italic');
                    obj.ModelsGrid.RowHeight = {'fit'};
                    return;
                end
                obj.ModelsGrid.RowHeight = repmat({'fit'}, 1, height(t));
                for k = 1:height(t)
                    model = t.Model(k);
                    text = t.Title(k);
                    if t.Version(k) ~= "", text = text + " v" + t.Version(k); end
                    tip = t.ModelName(k);
                    if ~t.Available(k), tip = "Unavailable: " + t.Reason(k); end
                    c = uicheckbox(obj.ModelsGrid, 'Text', char(text), 'Tooltip', char(tip), ...
                        'Enable', matlab.lang.OnOffSwitchState(t.Available(k)), ...
                        'ValueChangedFcn', @(cb, ~) obj.safe(@() obj.ViewModel.setUse(model, cb.Value)));
                    c.Layout.Row = k; c.Layout.Column = 1;
                    d = uidropdown(obj.ModelsGrid, 'Items', cellstr(t.Backends{k}), ...
                        'Enable', matlab.lang.OnOffSwitchState(t.Available(k) && numel(t.Backends{k}) > 1), ...
                        'Tooltip', 'How the service runs this model', ...
                        'ValueChangedFcn', @(dd, ~) obj.safe(@() obj.ViewModel.setBackend(model, dd.Value)));
                    d.Layout.Row = k; d.Layout.Column = 2;
                    obj.ModelControls(k) = struct('Model', model, 'Check', c, 'Backend', d);
                end
            end
            for k = 1:numel(obj.ModelControls)                  % values only
                obj.ModelControls(k).Check.Value = t.Use(k);
                obj.ModelControls(k).Backend.Value = char(t.Backend(k));
            end
        end

        function syncAudio(obj)
            vm = obj.ViewModel;
            obj.AudioLabel.Text = char(vm.AudioLabel);
            ax = obj.WaveAxes;
            cla(ax); legend(ax, 'off');
            if isempty(vm.Samples)
                title(ax, 'Waveform'); xlabel(ax, 'Time (s)'); ylabel(ax, 'Amplitude (full scale)');
                return;
            end
            obj.Viz.plot(ax, "SpeechWaveform", struct('Samples', vm.Samples, 'SampleRate', vm.AudioRate, ...
                'Title', sprintf("Waveform (%g Hz, %.2f s)", vm.AudioRate, numel(vm.Samples) / vm.AudioRate)));
            ylim(ax, [-1 1]);
        end

        function syncLevel(obj)
            obj.LevelGauge.Value = max(-60, min(0, obj.ViewModel.InputLevelDBFS));
        end

        function syncState(obj)
            vm = obj.ViewModel;
            idle = ~vm.IsRecording && ~vm.IsBusy;
            hasAudio = ~isempty(vm.Samples);
            obj.RecordButton.Enable = matlab.lang.OnOffSwitchState(idle);   % stays clickable: a click re-checks devices (gate 1)
            obj.StopButton.Enable = matlab.lang.OnOffSwitchState(vm.IsRecording);
            obj.PlayButton.Enable = matlab.lang.OnOffSwitchState(idle && hasAudio);
            obj.LoadButton.Enable = matlab.lang.OnOffSwitchState(idle);
            obj.TranscribeButton.Enable = matlab.lang.OnOffSwitchState(idle && hasAudio && vm.SpeechOk);   % gate 2
            obj.SaveButton.Enable = matlab.lang.OnOffSwitchState(idle && hasAudio);
            obj.RateDropDown.Enable = matlab.lang.OnOffSwitchState(idle);
            obj.DeviceDropDown.Enable = matlab.lang.OnOffSwitchState(idle);
            if vm.IsRecording
                obj.startLevelTimer();
            else
                obj.stopLevelTimer();
            end
        end

        function syncResults(obj)
            t = obj.ViewModel.Results;
            obj.ResultsTable.Data = t;
            removeStyle(obj.ResultsTable);
            bad = find(t.Error ~= "");
            if ~isempty(bad)
                addStyle(obj.ResultsTable, uistyle('FontColor', [0.8 0.1 0.1]), 'row', bad);
            end
        end

        function syncHistory(obj)
            t = obj.ViewModel.History;
            if ~isempty(t), t.Time.Format = 'HH:mm:ss'; end
            obj.HistoryTable.Data = flipud(t);                 % newest first
        end

        function syncHardware(obj)
            obj.HardwareArea.Value = splitlines(obj.ViewModel.HardwareText);
        end

        function syncLog(obj)
            vm = obj.ViewModel;
            obj.StatusLabel.Text = char(vm.StatusText);
            if isempty(vm.Log), obj.LogArea.Value = {''}; else, obj.LogArea.Value = cellstr(vm.Log); end
            scroll(obj.LogArea, 'bottom');
        end

        % ---- helpers -------------------------------------------------------------------------------
        function loadWav(obj)
            [f, p] = uigetfile({'*.wav;*.flac;*.mp3;*.ogg', 'Audio files'}, 'Load audio');
            figure(obj.Figure);
            if isequal(f, 0), return; end
            obj.ViewModel.loadWav(fullfile(p, f));
        end

        function startLevelTimer(obj)
            if ~isempty(obj.LevelTimer) && isvalid(obj.LevelTimer), return; end
            obj.LevelTimer = timer('ExecutionMode', 'fixedSpacing', 'Period', 0.1, 'BusyMode', 'drop', ...
                'TimerFcn', @(~, ~) obj.safe(@() obj.ViewModel.pollLevel()));
            start(obj.LevelTimer);
        end

        function stopLevelTimer(obj)
            if ~isempty(obj.LevelTimer) && isvalid(obj.LevelTimer)
                stop(obj.LevelTimer);
                delete(obj.LevelTimer);
            end
            obj.LevelTimer = [];
        end

        function withPointer(obj, fn)
            obj.Figure.Pointer = 'watch';
            drawnow;
            restore = onCleanup(@() obj.setPointer('arrow'));
            obj.safe(fn);
        end

        function setPointer(obj, p)
            if ~isempty(obj.Figure) && isvalid(obj.Figure), obj.Figure.Pointer = p; end
        end

        function safe(obj, fn)
            % SAFE  Run a callback; an unexpected error is shown, never left in the command window.
            if isempty(obj.Figure) || ~isvalid(obj.Figure), return; end
            try
                fn();
            catch err
                uialert(obj.Figure, err.message, 'ASR Console');
            end
        end

        function b = button(obj, g, text, r, c, fn)
            b = uibutton(g, 'Text', text, 'ButtonPushedFcn', @(~, ~) obj.safe(fn));
            obj.place(b, r, c);
        end

        function header(obj, g, text, r)
            h = uilabel(g, 'Text', char(text), 'FontWeight', 'bold', 'FontColor', [0.35 0.35 0.45]);
            obj.place(h, r, [1 2]);
        end

        function label(obj, g, text, r, c)
            l = uilabel(g, 'Text', char(text));
            obj.place(l, r, c);
        end

        function place(~, comp, r, c)
            comp.Layout.Row = r;
            comp.Layout.Column = c;
        end
    end
end
