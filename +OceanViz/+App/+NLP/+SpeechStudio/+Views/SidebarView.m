classdef SidebarView < handle
    % SIDEBARVIEW  Settings sidebar of the Speech Pipeline Studio: INPUT (FSDD / WAV /
    %   microphone with level meter), MODEL (checkpoint, decoder, LM / rescoring controls
    %   shown only when the model supports them) and ROBUSTNESS (AudioConditioner /
    %   SpeechActivityDetector options). Every control calls a ViewModel command; every
    %   ViewModel change is reflected through PostSet listeners (deleted in delete()).

    properties (SetAccess = private)
        Panel
        Grid
        SourceButtons struct = struct()
        SplitDropDown
        RandomButton
        UtteranceDropDown
        WavField
        BrowseButton
        DeviceDropDown
        MaxSecondsSpinner
        RecordButton
        StopButton
        LevelGauge
        LevelLabel
        ReferenceField
        InputLabel
        ModelDropDown
        ModelCardLabel
        DecoderDropDown
        Controls struct = struct()       % setting name -> component
        ControlRows struct = struct()    % setting name -> grid row
        SourceRows struct = struct()     % source -> grid rows
        RowLabels struct = struct()      % setting name -> its label
    end

    properties (Access = private)
        ViewModel
        Listeners = event.listener.empty
        OnError
    end

    methods
        function obj = SidebarView(parent, viewModel, onError)
            arguments
                parent
                viewModel (1,1) OceanViz.App.NLP.SpeechStudio.ViewModels.StudioViewModel
                onError = @(err) []
            end
            obj.ViewModel = viewModel;
            obj.OnError = onError;
            obj.build(parent);
            obj.bind();
            obj.sync();
        end

        function delete(obj)
            delete(obj.Listeners);
        end
    end

    methods (Access = private)
        function build(obj, parent)
            obj.Panel = uipanel(parent, 'Title', 'Settings', 'Scrollable', 'on');
            rows = 40;
            g = uigridlayout(obj.Panel, [rows 2], 'ColumnWidth', {'1x', '1x'}, ...
                'RowHeight', repmat({'fit'}, 1, rows), 'Padding', [8 8 8 8], 'RowSpacing', 5);
            obj.Grid = g;
            r = 0;
            r = r + 1; obj.header(g, "INPUT", r);
            r = r + 1;
            sg = uigridlayout(g, [1 3], 'Padding', [0 0 0 0], 'ColumnSpacing', 2);
            sg.Layout.Row = r; sg.Layout.Column = [1 2];
            src = ["FSDD", "WAV", "Microphone"];
            for k = 1:3
                obj.SourceButtons.(src(k)) = uibutton(sg, 'state', 'Text', char(src(k)), ...
                    'ValueChangedFcn', @(~, ~) obj.call(@() obj.ViewModel.setSource(src(k))));
            end
            % FSDD
            r = r + 1;
            obj.SplitDropDown = uidropdown(g, 'Items', {'(loading)'}, 'Tooltip', 'FSDD split (canonical: test = held-out speaker theo)', ...
                'ValueChangedFcn', @(d, ~) obj.call(@() obj.ViewModel.setSplit(string(d.Value))));
            obj.place(obj.SplitDropDown, r, 1);
            obj.RandomButton = uibutton(g, 'Text', 'Random held-out', ...
                'ButtonPushedFcn', @(~, ~) obj.call(@() obj.ViewModel.randomHeldOut()));
            obj.place(obj.RandomButton, r, 2);
            r = r + 1;
            obj.UtteranceDropDown = uidropdown(g, 'Items', {'(none)'}, 'Tooltip', 'FSDD utterance', ...
                'ValueChangedFcn', @(d, ~) obj.call(@() obj.ViewModel.selectUtterance(string(d.Value))));
            obj.place(obj.UtteranceDropDown, r, [1 2]);
            obj.SourceRows.FSDD = [r - 1, r];
            % WAV
            r = r + 1;
            obj.WavField = uieditfield(g, 'text', 'Editable', 'off', 'Placeholder', 'audio file');
            obj.place(obj.WavField, r, 1);
            obj.BrowseButton = uibutton(g, 'Text', 'Browse...', 'ButtonPushedFcn', @(~, ~) obj.onBrowse());
            obj.place(obj.BrowseButton, r, 2);
            obj.SourceRows.WAV = r;
            % Microphone
            r = r + 1;
            obj.DeviceDropDown = uidropdown(g, 'Items', {'(default device)'}, 'Tooltip', 'Input device', ...
                'ValueChangedFcn', @(d, ~) obj.ViewModel.setDevice(string(d.Value)));
            obj.place(obj.DeviceDropDown, r, 1);
            obj.MaxSecondsSpinner = uispinner(g, 'Limits', [0.5 60], 'Step', 0.5, 'Tooltip', 'Maximum recording (s)', ...
                'ValueDisplayFormat', 'max %.1f s', ...
                'ValueChangedFcn', @(s, ~) obj.ViewModel.setMaxRecordSeconds(s.Value));
            obj.place(obj.MaxSecondsSpinner, r, 2);
            r = r + 1;
            obj.RecordButton = uibutton(g, 'Text', char("● Record"), ...
                'ButtonPushedFcn', @(~, ~) obj.call(@() obj.ViewModel.startRecording()));
            obj.place(obj.RecordButton, r, 1);
            obj.StopButton = uibutton(g, 'Text', char("■ Stop"), ...
                'ButtonPushedFcn', @(~, ~) obj.call(@() obj.ViewModel.stopRecording()));
            obj.place(obj.StopButton, r, 2);
            r = r + 1;
            obj.LevelGauge = uigauge(g, 'linear', 'Limits', [-60 0], 'Value', -60, ...
                'Tooltip', 'Input level (dBFS RMS of the last 0.1 s)');
            obj.place(obj.LevelGauge, r, 1);
            obj.LevelLabel = uilabel(g, 'Text', 'level -inf dBFS');
            obj.place(obj.LevelLabel, r, 2);
            obj.SourceRows.Microphone = [r - 2, r - 1, r];
            r = r + 1;
            obj.ReferenceField = uieditfield(g, 'text', 'Placeholder', 'reference transcript (optional)', ...
                'Tooltip', 'Reference transcript: from the manifest for FSDD, typed for WAV / microphone', ...
                'ValueChangedFcn', @(f, ~) obj.call(@() obj.ViewModel.setReference(string(f.Value))));
            obj.place(obj.ReferenceField, r, [1 2]);
            r = r + 1;
            obj.InputLabel = uilabel(g, 'Text', 'No input', 'WordWrap', 'on', 'FontAngle', 'italic');
            obj.place(obj.InputLabel, r, [1 2]);

            r = r + 1; obj.header(g, "MODEL", r);
            r = r + 1;
            obj.ModelDropDown = uidropdown(g, 'Items', {'(loading)'}, 'Tooltip', 'Checkpoints found by ModelCatalog', ...
                'ValueChangedFcn', @(d, ~) obj.call(@() obj.ViewModel.selectModel(string(d.Value))));
            obj.place(obj.ModelDropDown, r, [1 2]);
            r = r + 1;
            obj.ModelCardLabel = uilabel(g, 'Text', '', 'WordWrap', 'on', 'FontSize', 11);
            obj.place(obj.ModelCardLabel, r, [1 2]);
            r = r + 1;
            obj.label(g, "Decoder", r);
            obj.DecoderDropDown = uidropdown(g, 'Items', {'greedy', 'beam'}, ...
                'ValueChangedFcn', @(d, ~) obj.call(@() obj.ViewModel.setSetting("Decoder", string(d.Value))));
            obj.place(obj.DecoderDropDown, r, 2);
            obj.Controls.Decoder = obj.DecoderDropDown;
            r = obj.spinnerRow(g, r + 1, "BeamWidth", "Beam width", [1 64], 1);
            r = obj.spinnerRow(g, r + 1, "LMWeight", "LM weight w", [0 5], 0.1);
            r = obj.spinnerRow(g, r + 1, "Bonus", "Label bonus b", [-5 5], 0.25);
            r = obj.spinnerRow(g, r + 1, "LMOrder", "LM order", [1 6], 1);
            r = obj.spinnerRow(g, r + 1, "CTCRescoreWeight", "CTC rescore lambda", [0 1], 0.1);

            r = r + 1; obj.header(g, "ROBUSTNESS", r);
            r = obj.checkRow(g, r + 1, "UseDetector", "Speech-activity detection");
            r = obj.spinnerRow(g, r + 1, "Sensitivity", "Detector sensitivity", [0 1], 0.05);
            r = r + 1;
            obj.RowLabels.Segmentation = obj.label(g, "Segmentation", r);
            c = uidropdown(g, 'Items', {'auto', 'never', 'always'}, 'Tooltip', ...
                'Split long input at pauses: auto = when longer than the max segment', ...
                'ValueChangedFcn', @(d, ~) obj.call(@() obj.ViewModel.setSetting("Segmentation", string(d.Value))));
            obj.place(c, r, 2);
            obj.Controls.Segmentation = c; obj.ControlRows.Segmentation = r;
            r = r + 1;
            note = uilabel(g, 'Text', ['auto splits only input longer than the max segment, at pauses; ', ...
                'on fast continuous speech a split can fall mid-word and raise the error rate ', ...
                '(compare with "never").'], 'WordWrap', 'on', 'FontSize', 10, 'FontAngle', 'italic');
            obj.place(note, r, [1 2]);
            r = obj.spinnerRow(g, r + 1, "MaxSegment", "Max segment (s)", [0.5 30], 0.5);
            r = obj.checkRow(g, r + 1, "Normalize", "Level normalisation");
            r = obj.spinnerRow(g, r + 1, "TargetRMSdBFS", "Target RMS (dBFS)", [-60 0], 1);
            r = obj.spinnerRow(g, r + 1, "MaxGainDB", "Max gain (dB)", [0 60], 1);
            r = obj.checkRow(g, r + 1, "RemoveDC", "Remove DC offset");
            g.RowHeight = [repmat({'fit'}, 1, r), {'1x'}];
        end

        function r = spinnerRow(obj, g, r, name, text, limits, step)
            obj.RowLabels.(name) = obj.label(g, text, r);
            s = uispinner(g, 'Limits', limits, 'Step', step, ...
                'ValueChangedFcn', @(c, ~) obj.call(@() obj.ViewModel.setSetting(name, c.Value)));
            obj.place(s, r, 2);
            obj.Controls.(name) = s;
            obj.ControlRows.(name) = r;
        end

        function r = checkRow(obj, g, r, name, text)
            c = uicheckbox(g, 'Text', char(text), ...
                'ValueChangedFcn', @(b, ~) obj.call(@() obj.ViewModel.setSetting(name, logical(b.Value))));
            obj.place(c, r, [1 2]);
            obj.Controls.(name) = c;
            obj.ControlRows.(name) = r;
        end

        function header(~, g, text, r)
            l = uilabel(g, 'Text', char(text), 'FontWeight', 'bold');
            l.Layout.Row = r; l.Layout.Column = [1 2];
        end

        function l = label(~, g, text, r)
            l = uilabel(g, 'Text', char(text));
            l.Layout.Row = r; l.Layout.Column = 1;
        end

        function place(~, c, r, col)
            c.Layout.Row = r; c.Layout.Column = col;
        end

        function bind(obj)
            vm = obj.ViewModel;
            props = ["Source", "Splits", "Split", "Utterances", "UtteranceId", "WavFile", "ReferenceText", ...
                "Devices", "Device", "IsRecording", "InputLevelDBFS", "ModelNames", "ModelName", "ModelCard", ...
                "Capabilities", "Settings", "InputSummary", "IsBusy", "MaxRecordSeconds"];
            for p = props
                obj.Listeners(end+1) = addlistener(vm, p, 'PostSet', @(~, ~) obj.sync());
            end
        end

        function sync(obj)
            vm = obj.ViewModel;
            if ~isvalid(obj.Grid), return; end
            src = ["FSDD", "WAV", "Microphone"];
            for s = src
                obj.SourceButtons.(s).Value = vm.Source == s;
                for rr = obj.SourceRows.(s)
                    obj.Grid.RowHeight{rr} = obj.iff(vm.Source == s, 'fit', 0);
                end
            end
            obj.setItems(obj.SplitDropDown, vm.Splits, vm.Split);
            obj.setItems(obj.UtteranceDropDown, vm.Utterances, vm.UtteranceId);
            obj.WavField.Value = char(vm.WavFile);
            obj.setItems(obj.DeviceDropDown, vm.Devices, vm.Device);
            obj.MaxSecondsSpinner.Value = vm.MaxRecordSeconds;
            obj.RecordButton.Enable = ~vm.IsRecording && ~vm.IsBusy;
            obj.StopButton.Enable = vm.IsRecording;
            obj.LevelGauge.Value = max(-60, min(0, vm.InputLevelDBFS));
            obj.LevelLabel.Text = sprintf('level %.1f dBFS', vm.InputLevelDBFS);
            obj.ReferenceField.Value = char(vm.ReferenceText);
            obj.ReferenceField.Editable = vm.Source ~= "FSDD";
            obj.InputLabel.Text = char(vm.InputSummary);
            obj.setItems(obj.ModelDropDown, vm.ModelNames, vm.ModelName);
            obj.ModelCardLabel.Text = char(vm.ModelCard);
            s = vm.Settings;
            f = string(fieldnames(obj.Controls))';
            for name = f
                c = obj.Controls.(name);
                v = s.(name);
                if isa(c, 'matlab.ui.control.DropDown'), c.Value = char(v); else, c.Value = v; end
                if isfield(obj.ControlRows, name)
                    show = vm.showControl(name);
                    obj.Grid.RowHeight{obj.ControlRows.(name)} = obj.iff(show, 'fit', 0);
                    c.Visible = show;
                    if isfield(obj.RowLabels, name), obj.RowLabels.(name).Visible = show; end
                end
            end
            busy = vm.IsBusy;
            set([obj.ModelDropDown, obj.UtteranceDropDown, obj.RandomButton, obj.BrowseButton], 'Enable', ~busy);
        end

        function setItems(~, dd, items, value)
            if isempty(items), items = "(none)"; end
            dd.Items = cellstr(items);
            if any(items == value), dd.Value = char(value); end
        end

        function onBrowse(obj)
            [f, p] = uigetfile({'*.wav;*.flac;*.ogg;*.mp3;*.m4a', 'Audio files'}, 'Choose an audio file');
            if isequal(f, 0), return; end
            obj.call(@() obj.ViewModel.setWavFile(string(fullfile(p, f))));
        end

        function call(obj, fn)
            try
                fn();
            catch err
                obj.OnError(err);
                obj.sync();
            end
        end
    end

    methods (Static, Access = private)
        function v = iff(c, a, b)
            if c, v = a; else, v = b; end
        end
    end
end
