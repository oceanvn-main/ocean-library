classdef SpeechExplorer < handle
    %SPEECHEXPLORER Inspect a trained speech recogniser on one real utterance, stage by stage.
    %
    %   app = OceanViz.App.NLP.SpeechExplorer()
    %   app = OceanViz.App.NLP.SpeechExplorer('Checkpoint', file, 'Manifest', m, ...
    %           'Visible', true, 'Theme', "Dark", 'Decoder', "beam", 'BeamWidth', 8)
    %   app.loadCheckpoint(file)            % Training.SpeechCheckpoint.load
    %   app.useManifest(m)                  % Data.SpeechManifest (splits assigned)
    %   app.selectUtterance(id)             % manifest entry -> waveform + reference
    %   app.loadWav(file, 'Reference', "")  % any WAV (resampled to the model rate if needed)
    %   app.analyze()                       % SpeechProcessAdapter.run + every plot
    %   app.close()
    %
    %   WHAT IT SHOWS (all from the real run; nothing is synthesised):
    %     Waveform, power spectrogram, mel filter bank, model input features
    %     (log-mel/MFCC), pitch contour (Features.PitchContour on the same frame grid),
    %     CTC posterior heat map with the best path, decoder beam candidates, and the
    %     reference/hypothesis edit alignment (Evaluation.EditAlignment on the
    %     vocabulary unit: characters for a char model, words otherwise) when a reference
    %     transcript is known. The step list is the SpeechProcessAdapter reader; "Open in
    %     Process Explorer" opens the generic OceanViz.App.NLP.ProcessExplorer on it.
    %   FSDD source: the manifest defaults to SpeechManifest.fromFSDD().canonicalFSDDSplit()
    %   (built on first use); audio is loaded with Data.AudioDataset.waveform (same
    %   loading/resampling path as training). The reference is the manifest transcript
    %   normalised with the model vocabulary's normaliser.
    %   Construct with 'Visible', false for headless use (tests).

    properties (SetAccess = private)
        Figure
        Model = []
        CheckpointFile (1,1) string = ""
        Manifest = []
        Waveform = []
        UtteranceId (1,1) string = ""
        Reference (1,1) string = ""
        Transcript = []
        Reader = []
        Alignment = []
        PlotData struct = struct()
    end

    properties (Access = private)
        Viz
        CheckpointField
        SourceDropDown
        SplitDropDown
        UtteranceDropDown
        WavField
        DecoderDropDown
        BeamSpinner
        StepTable
        InfoArea
        StatusLabel
        Tabs
        TabGrids struct = struct()
        Explorer = []
    end

    properties (Constant, Access = private)
        Views = ["Waveform", "Spectrogram", "MelBank", "Features", "Pitch", "Posteriors", "Beam", "Alignment"]
        ViewTitles = ["Waveform", "Spectrogram", "Mel bank", "Features", "Pitch", "CTC posteriors", ...
            "Beam candidates", "Error alignment"]
        ViewStrategies = ["SpeechWaveform", "Spectrogram", "MelFilterBank", "SpeechFeatures", ...
            "PitchContour", "CTCPosterior", "BeamCandidates", "ErrorAlignment"]
    end

    methods
        function obj = SpeechExplorer(opts)
            arguments
                opts.Checkpoint (1,1) string = ""
                opts.Manifest = []
                opts.Visible (1,1) logical = true
                opts.Theme (1,1) string = "Dark"
                opts.Decoder (1,1) string {mustBeMember(opts.Decoder, ["greedy", "beam"])} = "beam"
                opts.BeamWidth (1,1) double {mustBePositive, mustBeInteger} = 8
            end
            obj.Viz = OceanViz.Core('Theme', opts.Theme);
            obj.buildUi(opts);
            if ~isempty(opts.Manifest)
                obj.useManifest(opts.Manifest);
            end
            if opts.Checkpoint ~= ""
                obj.loadCheckpoint(opts.Checkpoint);
            end
        end

        function loadCheckpoint(obj, file)
            % LOADCHECKPOINT  Rebuild the recogniser (config + weights) from a SpeechCheckpoint file.
            arguments
                obj (1,1) OceanViz.App.NLP.SpeechExplorer
                file (1,1) string
            end
            ck = OceanNLP.Speech.Training.SpeechCheckpoint.load(file);
            obj.Model = ck.Model;
            obj.CheckpointFile = file;
            obj.CheckpointField.Value = char(file);
            obj.setStatus(sprintf("Loaded %s (%s, %d classes)", ck.ModelConfig.Type, ...
                obj.Model.ModelId, obj.Model.Vocabulary.numClasses()));
        end

        function useManifest(obj, manifest)
            % USEMANIFEST  Corpus to pick utterances from (rows need a Split).
            arguments
                obj (1,1) OceanViz.App.NLP.SpeechExplorer
                manifest (1,1) OceanNLP.Speech.Data.SpeechManifest
            end
            obj.Manifest = manifest;
            splits = unique(manifest.Entries.Split, 'stable');
            splits = splits(splits ~= "");
            if isempty(splits)
                error('OceanViz:App:NLP:SpeechExplorer:NoSplit', 'Manifest rows have no Split assigned.');
            end
            obj.SplitDropDown.Items = cellstr(splits);
            if any(splits == "test")
                obj.SplitDropDown.Value = 'test';
            else
                obj.SplitDropDown.Value = char(splits(1));
            end
            obj.refreshUtterances();
        end

        function selectUtterance(obj, id)
            % SELECTUTTERANCE  Load manifest entry ID (its waveform and reference transcript).
            arguments
                obj (1,1) OceanViz.App.NLP.SpeechExplorer
                id (1,1) string
            end
            obj.requireModel();
            if isempty(obj.Manifest)
                obj.useManifest(OceanNLP.Speech.Data.SpeechManifest.fromFSDD().canonicalFSDDSplit());
            end
            i = find(obj.Manifest.Entries.Id == id, 1);
            if isempty(i)
                error('OceanViz:App:NLP:SpeechExplorer:UnknownUtterance', 'No manifest entry "%s".', id);
            end
            one = obj.Manifest.subset(i);
            ds = OceanNLP.Speech.Data.AudioDataset(one, obj.Model.Features, ...
                'SampleRate', obj.modelRate(), 'Cache', false);
            obj.Waveform = ds.waveform(1);
            obj.UtteranceId = id;
            obj.Reference = obj.normalize(one.Entries.Transcript(1));
            split = one.Entries.Split(1);
            if any(string(obj.SplitDropDown.Items) == split)
                obj.SplitDropDown.Value = char(split);
                obj.refreshUtterances();
            end
            obj.UtteranceDropDown.Value = char(id);
            obj.SourceDropDown.Value = 'Manifest';
            obj.setStatus(sprintf("Utterance %s (%s split, speaker %s): ""%s""", id, split, ...
                one.Entries.Speaker(1), obj.Reference));
        end

        function loadWav(obj, file, opts)
            % LOADWAV  Any audio file; resampled to the model's rate when it differs (recorded in History).
            arguments
                obj (1,1) OceanViz.App.NLP.SpeechExplorer
                file (1,1) string
                opts.Reference (1,1) string = ""
            end
            obj.requireModel();
            w = OceanNLP.Speech.Core.Waveform.read(file);
            fs = obj.modelRate();
            if w.SampleRate ~= fs
                y = [];
                for ch = 1:w.numChannels()
                    yc = OceanMath.Signal.ResampleSignal.resample(w.Samples(:, ch), w.SampleRate, fs);
                    y = [y, yc(:)]; %#ok<AGROW>
                end
                w = w.derive(y, fs, sprintf("resampled %g -> %g Hz (OceanMath.Signal.ResampleSignal)", w.SampleRate, fs));
            end
            obj.Waveform = w;
            obj.UtteranceId = file;
            obj.Reference = "";
            if opts.Reference ~= ""
                obj.Reference = obj.normalize(opts.Reference);
            end
            obj.WavField.Value = char(file);
            obj.SourceDropDown.Value = 'WAV file';
            obj.setStatus(sprintf("WAV %s (%.3f s)", file, w.duration()));
        end

        function analyze(obj)
            % ANALYZE  Run the recogniser through SpeechProcessAdapter and render every view.
            arguments
                obj (1,1) OceanViz.App.NLP.SpeechExplorer
            end
            obj.requireModel();
            if isempty(obj.Waveform)
                error('OceanViz:App:NLP:SpeechExplorer:NoAudio', 'Select an utterance or load a WAV file first.');
            end
            trace = OceanNLP.Speech.Inspect.SpeechTrace();
            [tr, reader] = OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(obj.Model, obj.Waveform, ...
                'Trace', trace, 'Decoder', string(obj.DecoderDropDown.Value), 'BeamWidth', obj.BeamSpinner.Value);
            obj.Transcript = tr;
            obj.Reader = reader;
            obj.PlotData = obj.collectPlotData(reader);
            for v = 1:numel(obj.Views)
                obj.render(obj.Views(v));
            end
            steps = reader.listSteps();
            obj.StepTable.Data = [cellstr([steps.Stage]'), cellstr([steps.Name]'), cellstr([steps.Status]')];
            lines = ["Utterance: " + obj.UtteranceId; "Hypothesis: """ + tr.Text + """"; ...
                "Decoder: " + tr.Decoder + sprintf(" (score %.3f nats)", tr.Score)];
            if obj.Reference ~= ""
                a = obj.Alignment;
                lines = [lines; "Reference: """ + obj.Reference + """"; ...
                    sprintf("Alignment: %d hits, %d S, %d D, %d I", a.Hits, a.Substitutions, a.Deletions, a.Insertions)];
            end
            obj.InfoArea.Value = lines;
            obj.setStatus("Analysed: """ + tr.Text + """");
        end

        function showStep(obj, index)
            % SHOWSTEP  Show step INDEX of the reader in the info area.
            arguments
                obj (1,1) OceanViz.App.NLP.SpeechExplorer
                index (1,1) double {mustBeInteger, mustBePositive}
            end
            if isempty(obj.Reader), return; end
            steps = obj.Reader.listSteps();
            if index > numel(steps), return; end
            s = steps(index);
            lines = [s.Name + " [" + s.Stage + "]"; s.Summary];
            for id = s.OutputIds
                art = obj.Reader.getArtifact(id);
                lines(end+1) = "Output " + id + ": " + art.Summary; %#ok<AGROW>
            end
            if ~isempty(s.InputIds)
                lines(end+1) = "Inputs: " + strjoin(s.InputIds, ", ");
            end
            obj.InfoArea.Value = lines;
        end

        function app = openProcessExplorer(obj)
            % OPENPROCESSEXPLORER  Generic step/artifact browser on the current reader.
            arguments
                obj (1,1) OceanViz.App.NLP.SpeechExplorer
            end
            if isempty(obj.Reader)
                error('OceanViz:App:NLP:SpeechExplorer:NotAnalysed', 'Run analyze() first.');
            end
            app = OceanViz.App.NLP.ProcessExplorer(obj.Reader, 'Visible', strcmp(obj.Figure.Visible, 'on'));
            obj.Explorer = app;
        end

        function ax = axesOf(obj, view)
            % AXESOF  The axes currently showing VIEW (for tests / export).
            arguments
                obj (1,1) OceanViz.App.NLP.SpeechExplorer
                view (1,1) string
            end
            ax = findall(obj.TabGrids.(view), 'Type', 'axes');
        end

        function close(obj)
            arguments
                obj (1,1) OceanViz.App.NLP.SpeechExplorer
            end
            if ~isempty(obj.Explorer), obj.Explorer.close(); end
            if isgraphics(obj.Figure), delete(obj.Figure); end
        end

        function delete(obj)
            if ~isempty(obj.Figure) && isgraphics(obj.Figure), delete(obj.Figure); end
        end
    end

    methods (Access = private)
        function buildUi(obj, opts)
            c = obj.Viz.Colors;
            obj.Figure = uifigure('Name', 'Speech Explorer', 'Position', [60 60 1480 880], ...
                'Color', c.Background, 'Visible', matlab.lang.OnOffSwitchState(opts.Visible));
            root = uigridlayout(obj.Figure, [2 2]);
            root.RowHeight = {44, '1x'};
            root.ColumnWidth = {340, '1x'};
            root.BackgroundColor = c.Background;
            t = uilabel(root, 'Text', 'SPEECH EXPLORER', 'FontSize', 20, 'FontWeight', 'bold', 'FontColor', c.Text);
            t.Layout.Row = 1; t.Layout.Column = 1;
            obj.StatusLabel = uilabel(root, 'Text', 'Load a checkpoint, choose an utterance, press Analyze.', ...
                'FontColor', c.Text);
            obj.StatusLabel.Layout.Row = 1; obj.StatusLabel.Layout.Column = 2;

            left = uigridlayout(root, [14 2]);
            left.Layout.Row = 2; left.Layout.Column = 1;
            left.BackgroundColor = c.Background;
            left.RowHeight = {22, 28, 28, 22, 28, 28, 28, 28, 28, 28, 28, 22, '1x', 130};
            left.ColumnWidth = {'1x', 90};
            obj.label(left, 'Checkpoint', 1);
            obj.CheckpointField = uieditfield(left, 'text', 'Value', char(opts.Checkpoint));
            obj.CheckpointField.Layout.Row = 2; obj.CheckpointField.Layout.Column = 1;
            b = uibutton(left, 'Text', 'Load', 'ButtonPushedFcn', @(~, ~) obj.onLoadCheckpoint());
            b.Layout.Row = 2; b.Layout.Column = 2;
            obj.label(left, 'Source', 4);
            obj.SourceDropDown = uidropdown(left, 'Items', {'Manifest', 'WAV file'}, 'Value', 'Manifest');
            obj.SourceDropDown.Layout.Row = 5; obj.SourceDropDown.Layout.Column = [1 2];
            obj.SplitDropDown = uidropdown(left, 'Items', {'(load manifest)'}, ...
                'ValueChangedFcn', @(~, ~) obj.refreshUtterances());
            obj.SplitDropDown.Layout.Row = 6; obj.SplitDropDown.Layout.Column = 1;
            b = uibutton(left, 'Text', 'FSDD', 'Tooltip', 'Build the canonical FSDD manifest', ...
                'ButtonPushedFcn', @(~, ~) obj.useManifest(OceanNLP.Speech.Data.SpeechManifest.fromFSDD().canonicalFSDDSplit()));
            b.Layout.Row = 6; b.Layout.Column = 2;
            obj.UtteranceDropDown = uidropdown(left, 'Items', {'(none)'}, ...
                'ValueChangedFcn', @(~, ~) obj.onUtterance());
            obj.UtteranceDropDown.Layout.Row = 7; obj.UtteranceDropDown.Layout.Column = [1 2];
            obj.WavField = uieditfield(left, 'text', 'Placeholder', 'WAV file path');
            obj.WavField.Layout.Row = 8; obj.WavField.Layout.Column = 1;
            b = uibutton(left, 'Text', 'Browse', 'ButtonPushedFcn', @(~, ~) obj.onBrowseWav());
            b.Layout.Row = 8; b.Layout.Column = 2;
            obj.DecoderDropDown = uidropdown(left, 'Items', {'greedy', 'beam'}, 'Value', char(opts.Decoder));
            obj.DecoderDropDown.Layout.Row = 9; obj.DecoderDropDown.Layout.Column = 1;
            obj.BeamSpinner = uispinner(left, 'Value', opts.BeamWidth, 'Limits', [1 64], 'Step', 1, ...
                'RoundFractionalValues', 'on', 'Tooltip', 'Beam width');
            obj.BeamSpinner.Layout.Row = 9; obj.BeamSpinner.Layout.Column = 2;
            b = uibutton(left, 'Text', 'Analyze', 'FontWeight', 'bold', 'ButtonPushedFcn', @(~, ~) obj.guarded(@() obj.analyze()));
            b.Layout.Row = 10; b.Layout.Column = [1 2];
            b = uibutton(left, 'Text', 'Open in Process Explorer', 'ButtonPushedFcn', @(~, ~) obj.guarded(@() obj.openProcessExplorer()));
            b.Layout.Row = 11; b.Layout.Column = [1 2];
            obj.label(left, 'Pipeline steps (SpeechProcessAdapter)', 12);
            obj.StepTable = uitable(left, 'ColumnName', {'Stage', 'Step', 'Status'}, ...
                'ColumnEditable', [false false false], 'CellSelectionCallback', @(~, e) obj.onStep(e));
            obj.StepTable.RowName = {};
            obj.StepTable.Layout.Row = 13; obj.StepTable.Layout.Column = [1 2];
            obj.InfoArea = uitextarea(left, 'Editable', 'off');
            obj.InfoArea.Layout.Row = 14; obj.InfoArea.Layout.Column = [1 2];

            obj.Tabs = uitabgroup(root);
            obj.Tabs.Layout.Row = 2; obj.Tabs.Layout.Column = 2;
            for v = 1:numel(obj.Views)
                tab = uitab(obj.Tabs, 'Title', obj.ViewTitles(v), 'BackgroundColor', c.Background);
                g = uigridlayout(tab, [1 1]);
                g.BackgroundColor = c.Background;
                obj.TabGrids.(obj.Views(v)) = g;
                ax = uiaxes(g);
                obj.Viz.styleAxes(ax);
            end
        end

        function label(obj, parent, text, row)
            l = uilabel(parent, 'Text', text, 'FontWeight', 'bold', 'FontColor', obj.Viz.Colors.Text);
            l.Layout.Row = row; l.Layout.Column = [1 2];
        end

        function d = collectPlotData(obj, reader)
            % Plot data from the reader's recorded artifacts (+ pitch and alignment, which
            % are not recogniser stages and are computed here by their library classes).
            d = struct();
            get = @(name) OceanViz.App.NLP.SpeechExplorer.artifactByName(reader, name);
            wave = get("Waveform");
            frames = get("Frames + window");
            if ~isempty(frames)
                wave.FrameStarts = frames.FrameStarts;
                wave.WindowLength = frames.WindowLength;
            end
            d.Waveform = wave;
            d.Spectrogram = get("Power spectrum");
            d.MelBank = get("Mel filter bank");
            d.Features = get("Model features");
            post = get("Posteriors");
            d.Posteriors = post;
            beam = get("Beam candidates");
            if ~isempty(beam) && obj.Reference ~= ""
                beam.Reference = obj.Reference;
            end
            d.Beam = beam;

            % pitch on the model's frame grid (same frame times as the features)
            pitchArgs = {};
            cfg = d.Features.Config;
            grid = [];
            if isfield(cfg, 'Grid'), grid = cfg.Grid;
            elseif isfield(cfg, 'LogMel'), grid = cfg.LogMel.Grid; end
            if ~isempty(grid)
                pitchArgs = {'WindowLength', grid.WindowLength, 'HopLength', grid.HopLength, 'NFFT', grid.NFFT, ...
                    'Window', grid.Window, 'Periodic', grid.Periodic, 'Center', grid.Center, 'PadMode', grid.PadMode};
            end
            pc = OceanNLP.Speech.Features.PitchContour(obj.Waveform.SampleRate, pitchArgs{:});
            d.Pitch = OceanViz.Strategies.Speech.PitchContourStrategy.fromFeatureSequence( ...
                pc.forward(obj.Waveform), 'VoicingThreshold', pc.VoicingThreshold);

            % reference vs hypothesis alignment on the vocabulary's unit
            obj.Alignment = [];
            d.Alignment = [];
            if obj.Reference ~= ""
                [refTok, hypTok, unit] = obj.tokens(obj.Reference, obj.Transcript.Text);
                obj.Alignment = OceanNLP.Speech.Evaluation.EditAlignment.align(refTok, hypTok);
                d.Alignment = struct('Alignment', obj.Alignment, 'Reference', refTok, ...
                    'Hypothesis', hypTok, 'Unit', unit);
            end
        end

        function render(obj, view)
            g = obj.TabGrids.(view);
            delete(g.Children);
            ax = uiaxes(g);
            data = obj.PlotData.(view);
            idx = find(obj.Views == view, 1);
            if isempty(data)
                obj.Viz.styleAxes(ax);
                title(ax, obj.ViewTitles(idx) + ": not available for this run", ...
                    'Color', obj.Viz.Colors.Text, 'Interpreter', 'none');
                return;
            end
            obj.Viz.plot(ax, obj.ViewStrategies(idx), data);
        end

        function [r, h, unit] = tokens(obj, ref, hyp)
            if obj.Model.Vocabulary.Unit == "char"
                unit = "character";
                r = string(num2cell(char(ref)));
                h = string(num2cell(char(hyp)));
            else
                unit = "word";
                r = split(ref, " ")'; h = split(hyp, " ")';
                r = r(r ~= ""); h = h(h ~= "");
            end
            r = reshape(r, 1, []); h = reshape(h, 1, []);
        end

        function s = normalize(obj, text)
            n = obj.Model.Vocabulary.Normalizer;
            if isempty(n)
                s = string(text);
            else
                s = n.forward(string(text));
            end
        end

        function fs = modelRate(obj)
            f = obj.Model.Features;
            if isprop(f, 'SampleRate')
                fs = f.SampleRate;
            elseif isa(f, 'OceanNLP.Speech.Features.MFCC')
                fs = f.LogMelTransform.SampleRate;
            elseif isa(f, 'OceanNLP.Speech.Features.FeaturePipeline') && isprop(f.Stages{1}, 'SampleRate')
                fs = f.Stages{1}.SampleRate;
            else
                error('OceanViz:App:NLP:SpeechExplorer:SampleRate', 'Cannot determine the model''s sample rate.');
            end
        end

        function requireModel(obj)
            if isempty(obj.Model)
                error('OceanViz:App:NLP:SpeechExplorer:NoModel', 'Load a checkpoint first.');
            end
        end

        function refreshUtterances(obj)
            if isempty(obj.Manifest), return; end
            split = string(obj.SplitDropDown.Value);
            ids = obj.Manifest.Entries.Id(obj.Manifest.Entries.Split == split);
            if isempty(ids), ids = "(none)"; end
            obj.UtteranceDropDown.Items = cellstr(ids);
        end

        function setStatus(obj, text)
            obj.StatusLabel.Text = char(text);
        end

        function guarded(obj, fn)
            try
                fn();
            catch ME
                obj.setStatus("Error: " + ME.message);
                if strcmp(obj.Figure.Visible, 'on')
                    uialert(obj.Figure, ME.message, 'Speech Explorer');
                end
            end
        end

        function onLoadCheckpoint(obj)
            obj.guarded(@() obj.loadCheckpoint(string(obj.CheckpointField.Value)));
        end

        function onUtterance(obj)
            id = string(obj.UtteranceDropDown.Value);
            if id == "(none)", return; end
            obj.guarded(@() obj.selectUtterance(id));
        end

        function onBrowseWav(obj)
            [f, p] = uigetfile({'*.wav;*.flac;*.ogg', 'Audio files'}, 'Choose an audio file');
            if isequal(f, 0), return; end
            obj.guarded(@() obj.loadWav(string(fullfile(p, f))));
        end

        function onStep(obj, event)
            if isempty(event.Indices), return; end
            obj.showStep(event.Indices(1, 1));
        end
    end

    methods (Static, Access = private)
        function v = artifactByName(reader, name)
            v = [];
            steps = reader.listSteps();
            k = find([steps.Name] == name, 1);
            if isempty(k), return; end
            art = reader.getArtifact(steps(k).OutputIds(1));
            if art.Status == "recorded"
                v = art.Value;
            end
        end
    end
end
