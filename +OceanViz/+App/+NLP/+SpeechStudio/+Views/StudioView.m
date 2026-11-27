classdef StudioView < handle
    % STUDIOVIEW  The Speech Pipeline Studio window: toolbar, settings sidebar, stage stepper,
    %   stage content (nine IStage plug-ins), nav row (warnings chip, segment, Back / Next)
    %   and status bar -- all uigridlayout; the chrome never moves. Workflow mode swaps only
    %   the content area for the Workflow tabs.
    %
    %   v = OceanViz.App.NLP.SpeechStudio.Views.StudioView(viewModel, 'Visible', true, 'Theme', "Dark")
    %   v.run()            Run (with uiprogressdlg when visible)
    %   v.stage(k)         the k-th stage object;  v.close()

    properties (SetAccess = private)
        Figure
        ViewModel
        Viz
        Sidebar
        Stepper
        Workflow
        Stages = {}
        RootGrid
        MainGrid
        ContentPanel
        RunButton
        PlayButton
        ModeButtons struct = struct()
        SidebarToggle
        WarningsButton
        SegmentDropDown
        StaleLabel
        BackButton
        NextButton
        StatusLabel
        ProgressLog (1,:) string = strings(1, 0)
        LastError (1,1) string = ""
        Player = []
    end

    properties (Access = private)
        Listeners = event.listener.empty
        RenderedKey = {}
        LevelTimer = []
    end

    methods
        function obj = StudioView(viewModel, opts)
            arguments
                viewModel (1,1) OceanViz.App.NLP.SpeechStudio.ViewModels.StudioViewModel
                opts.Visible (1,1) logical = true
                opts.Theme (1,1) string = "Dark"
                opts.Stages = []
            end
            obj.ViewModel = viewModel;
            obj.Viz = OceanViz.Core('Theme', opts.Theme);
            obj.Stages = opts.Stages;
            if isempty(obj.Stages), obj.Stages = OceanViz.App.NLP.SpeechStudio.Stages.StageRegistry.create(); end
            obj.RenderedKey = cell(1, numel(obj.Stages));
            obj.buildUi(opts);
            obj.bind();
            obj.syncAll();
        end

        function delete(obj)
            delete(obj.Listeners);
            obj.stopLevelTimer();
            if ~isempty(obj.Player), try, stop(obj.Player); catch, end, end
            if ~isempty(obj.Sidebar) && isvalid(obj.Sidebar), delete(obj.Sidebar); end
            if ~isempty(obj.Stepper) && isvalid(obj.Stepper), delete(obj.Stepper); end
            if ~isempty(obj.Figure) && isvalid(obj.Figure), delete(obj.Figure); end
        end

        function close(obj)
            % CLOSE  Delete the window, the view's listeners / timers and the ViewModel.
            vm = obj.ViewModel;
            sess = [];
            if ~isempty(vm) && isvalid(vm), sess = vm.Session; end
            delete(obj);
            if ~isempty(vm) && isvalid(vm), delete(vm); end
            if ~isempty(sess) && isvalid(sess), sess.releaseRuns(); end   % drop cached traces
        end

        function ok = run(obj)
            % RUN  Recompute the whole pipeline once (progress dialog when visible).
            obj.ProgressLog = strings(1, 0);
            dlg = [];
            if strcmp(obj.Figure.Visible, 'on')
                dlg = uiprogressdlg(obj.Figure, 'Title', 'Running the speech pipeline', 'Message', 'Starting...');
            end
            fcn = @(frac, msg) obj.onProgress(dlg, frac, msg);
            ok = obj.ViewModel.run(fcn);
            if ~isempty(dlg) && isvalid(dlg), close(dlg); end
            if ~ok, obj.LastError = obj.ViewModel.StatusText; end
        end

        function s = stage(obj, k)
            s = obj.Stages{k};
        end

        function ok = play(obj)
            % PLAY  Play the shown piece of conditioned audio; the playhead drives the cursor.
            ok = false;
            ctx = obj.ViewModel.Context;
            if isempty(ctx), return; end
            w = ctx.segmentWave();
            x = w.Samples(:, 1);
            x = x / max(1, max(abs(x)));
            try
                if ~isempty(obj.Player), stop(obj.Player); end
                p = audioplayer(x, w.SampleRate);
                p.TimerPeriod = 0.04;
                p.TimerFcn = @(src, ~) obj.ViewModel.setCursor(src.CurrentSample / src.SampleRate);
                p.StopFcn = @(~, ~) obj.ViewModel.setCursor(NaN);
                obj.Player = p;
                play(p);
                ok = true;
            catch err
                obj.reportError(err);
            end
        end

        function hover(obj)
            % HOVER  Pointer moved: linked cursor at the time under the pointer.
            if obj.ViewModel.Mode ~= "Pipeline" || ~obj.ViewModel.HasResult, return; end
            t = obj.Stages{obj.ViewModel.CurrentStage}.pointerTime(obj.Figure);
            if ~isnan(t), obj.ViewModel.setCursor(t); end
        end

        function click(obj)
            if obj.ViewModel.Mode ~= "Pipeline" || ~obj.ViewModel.HasResult, return; end
            s = obj.Stages{obj.ViewModel.CurrentStage};
            t = s.pointerTime(obj.Figure);
            if ~isnan(t), s.onClick(t); end
        end
    end

    methods (Access = private)
        function buildUi(obj, opts)
            obj.Figure = uifigure('Name', 'Speech Pipeline Studio', 'Position', [40 40 1560 920], ...
                'Visible', 'off', 'CloseRequestFcn', @(~, ~) obj.close(), ...
                'WindowButtonMotionFcn', @(~, ~) obj.hover(), 'WindowButtonDownFcn', @(~, ~) obj.click());
            try
                obj.Figure.Theme = lower(char(opts.Theme));    % built-in uifigure theme matching OceanViz
            catch
            end
            root = uigridlayout(obj.Figure, [3 2], 'RowHeight', {44, '1x', 24}, 'ColumnWidth', {280, '1x'}, ...
                'Padding', [0 0 0 0], 'RowSpacing', 0, 'ColumnSpacing', 0);
            obj.RootGrid = root;

            % toolbar
            tb = uigridlayout(root, [1 9], 'ColumnWidth', {32, 'fit', 90, 90, '1x', 90, 90, 32, 10}, ...
                'Padding', [6 6 6 6], 'ColumnSpacing', 6);
            tb.Layout.Row = 1; tb.Layout.Column = [1 2];
            obj.SidebarToggle = uibutton(tb, 'Text', char("☰"), 'Tooltip', 'Show / hide the settings sidebar', ...
                'ButtonPushedFcn', @(~, ~) obj.toggleSidebar());
            uilabel(tb, 'Text', 'Speech Pipeline Studio', 'FontSize', 18, 'FontWeight', 'bold');
            for m = ["Pipeline", "Workflow"]
                obj.ModeButtons.(m) = uibutton(tb, 'state', 'Text', char(m), ...
                    'ValueChangedFcn', @(~, ~) obj.ViewModel.setMode(m));
            end
            sp = uilabel(tb, 'Text', ''); sp.Layout.Column = 5;
            obj.RunButton = uibutton(tb, 'Text', char("▶ Run"), 'FontWeight', 'bold', ...
                'ButtonPushedFcn', @(~, ~) obj.run());
            obj.RunButton.Layout.Column = 6;
            obj.PlayButton = uibutton(tb, 'Text', char("♪ Play"), 'Tooltip', ...
                'Play the conditioned audio of the shown segment; the playhead moves the time cursor', ...
                'ButtonPushedFcn', @(~, ~) obj.play());
            obj.PlayButton.Layout.Column = 7;
            h = uibutton(tb, 'Text', char("ⓘ"), 'Tooltip', 'How to use the studio', ...
                'ButtonPushedFcn', @(~, ~) obj.showHelp());
            h.Layout.Column = 8;

            % sidebar
            sbHost = uigridlayout(root, [1 1], 'Padding', [4 4 0 4]);
            sbHost.Layout.Row = 2; sbHost.Layout.Column = 1;
            obj.Sidebar = OceanViz.App.NLP.SpeechStudio.Views.SidebarView(sbHost, obj.ViewModel, @(e) obj.reportError(e));

            % main: pipeline | workflow (row collapse)
            obj.MainGrid = uigridlayout(root, [2 1], 'RowHeight', {'1x', 0}, 'Padding', [0 0 0 0], 'RowSpacing', 0);
            obj.MainGrid.Layout.Row = 2; obj.MainGrid.Layout.Column = 2;
            pg = uigridlayout(obj.MainGrid, [3 1], 'RowHeight', {64, '1x', 48}, 'Padding', [0 0 0 0], 'RowSpacing', 0);
            pg.Layout.Row = 1;
            stp = uipanel(pg, 'BorderType', 'none');
            obj.Stepper = OceanViz.App.NLP.SpeechStudio.Views.StepperView(stp, obj.ViewModel);
            obj.ContentPanel = uipanel(pg, 'BorderType', 'none');
            for k = 1:numel(obj.Stages)
                obj.Stages{k}.build(obj.ContentPanel);
                obj.Stages{k}.Root.Visible = 'off';
            end
            nav = uigridlayout(pg, [1 6], 'ColumnWidth', {'fit', 'fit', 'fit', '1x', 100, 100}, 'Padding', [8 8 8 8]);
            obj.WarningsButton = uibutton(nav, 'Text', 'No warnings', 'ButtonPushedFcn', @(~, ~) obj.showWarnings());
            obj.SegmentDropDown = uidropdown(nav, 'Items', {'Segment 1 of 1'}, 'ItemsData', 1, ...
                'Tooltip', 'Audio piece whose stages are shown (long input is segmented at pauses)', ...
                'ValueChangedFcn', @(d, ~) obj.ViewModel.selectSegment(d.Value));
            obj.StaleLabel = uilabel(nav, 'Text', '', 'FontColor', [0.9 0.6 0.1], 'FontWeight', 'bold');
            obj.BackButton = uibutton(nav, 'Text', char("◀ Back"), 'ButtonPushedFcn', @(~, ~) obj.ViewModel.back());
            obj.BackButton.Layout.Column = 5;
            obj.NextButton = uibutton(nav, 'Text', char("Next ▶"), 'ButtonPushedFcn', @(~, ~) obj.ViewModel.next());
            obj.NextButton.Layout.Column = 6;
            wp = uipanel(obj.MainGrid, 'BorderType', 'none');
            wp.Layout.Row = 2;
            wpg = uigridlayout(wp, [1 1], 'Padding', [4 4 4 4]);
            obj.Workflow = OceanViz.App.NLP.SpeechStudio.Views.WorkflowView(wpg, obj.ViewModel.Session, obj.Viz, ...
                @(e) obj.reportError(e));

            % status bar
            obj.StatusLabel = uilabel(root, 'Text', '', 'FontSize', 11);
            obj.StatusLabel.Layout.Row = 3; obj.StatusLabel.Layout.Column = [1 2];
            obj.Figure.Visible = matlab.lang.OnOffSwitchState(opts.Visible);
        end

        function bind(obj)
            vm = obj.ViewModel;
            on = @(p, f) addlistener(vm, p, 'PostSet', @(~, ~) f());
            obj.Listeners = [ ...
                on('Mode', @() obj.syncMode()), ...
                on('CurrentStage', @() obj.syncStage()), ...
                on('Context', @() obj.syncStage()), ...
                on('SegmentIndex', @() obj.syncStage()), ...
                on('IsStale', @() obj.syncNav()), on('HasResult', @() obj.syncNav()), ...
                on('StageEnabled', @() obj.syncNav()), on('NumSegments', @() obj.syncNav()), ...
                on('Warnings', @() obj.syncNav()), on('StatusText', @() obj.syncStatus()), ...
                on('RunSummary', @() obj.syncStatus()), on('IsBusy', @() obj.syncStatus()), ...
                on('InputReady', @() obj.syncStatus()), on('NoSpeech', @() obj.syncNav()), ...
                on('CursorTime', @() obj.syncCursor()), on('IsRecording', @() obj.syncRecording())];
        end

        function syncAll(obj)
            obj.syncMode(); obj.syncNav(); obj.syncStatus(); obj.syncStage();
        end

        function syncMode(obj)
            m = obj.ViewModel.Mode;
            obj.ModeButtons.Pipeline.Value = m == "Pipeline";
            obj.ModeButtons.Workflow.Value = m == "Workflow";
            if m == "Pipeline"
                obj.MainGrid.RowHeight = {'1x', 0};
            else
                obj.MainGrid.RowHeight = {0, '1x'};
                obj.Workflow.ensureSelected();
            end
        end

        function syncStage(obj)
            vm = obj.ViewModel;
            k = vm.CurrentStage;
            for i = 1:numel(obj.Stages)
                obj.Stages{i}.Root.Visible = matlab.lang.OnOffSwitchState(i == k);
            end
            ctx = vm.Context;
            if isempty(ctx) || ~obj.Stages{k}.supports(ctx.Capabilities), obj.syncNav(); return; end
            ctx.Viz = obj.Viz;
            key = obj.RenderedKey{k};
            if isempty(key) || key{1} ~= ctx || key{2} ~= ctx.SegmentIndex
                obj.Stages{k}.render(ctx);
                obj.RenderedKey{k} = {ctx, ctx.SegmentIndex};
            end
            obj.syncNav();
        end

        function syncNav(obj)
            vm = obj.ViewModel;
            obj.BackButton.Enable = vm.canGoBack();
            obj.NextButton.Enable = vm.canGoNext();
            n = numel(vm.Warnings);
            if n == 0
                obj.WarningsButton.Text = 'No warnings';
            else
                obj.WarningsButton.Text = char(sprintf("⚠ %d warning(s)", n));
            end
            obj.WarningsButton.Tooltip = char(strjoin(vm.Warnings, newline));
            ns = vm.NumSegments;
            obj.SegmentDropDown.Items = cellstr(compose("Segment %d of %d", (1:ns)', ns));
            obj.SegmentDropDown.ItemsData = 1:ns;
            obj.SegmentDropDown.Value = min(vm.SegmentIndex, ns);
            obj.SegmentDropDown.Enable = ns > 1 && vm.HasResult;
            if vm.IsStale
                obj.StaleLabel.Text = char("Stale: settings changed, press ▶ Run");
            elseif ~vm.HasResult
                obj.StaleLabel.Text = 'Press ▶ Run to compute the pipeline';
            else
                obj.StaleLabel.Text = '';
            end
            obj.PlayButton.Enable = vm.HasResult;
        end

        function syncStatus(obj)
            vm = obj.ViewModel;
            obj.StatusLabel.Text = char(vm.RunSummary + "  |  " + vm.StatusText);
            obj.RunButton.Enable = ~vm.IsBusy && vm.InputReady;
            if vm.InputReady
                obj.RunButton.Tooltip = 'Compute the whole pipeline once';
            else
                obj.RunButton.Tooltip = char(vm.InputSummary);
            end
        end

        function syncCursor(obj)
            k = obj.ViewModel.CurrentStage;
            obj.Stages{k}.setCursor(obj.ViewModel.CursorTime);
        end

        function syncRecording(obj)
            if obj.ViewModel.IsRecording
                obj.stopLevelTimer();
                obj.LevelTimer = timer('ExecutionMode', 'fixedSpacing', 'Period', 0.1, ...
                    'TimerFcn', @(~, ~) obj.pollLevel(), 'BusyMode', 'drop');
                start(obj.LevelTimer);
            else
                obj.stopLevelTimer();
            end
        end

        function pollLevel(obj)
            try
                obj.ViewModel.pollLevel();
            catch err
                obj.stopLevelTimer();
                obj.reportError(err);
            end
        end

        function stopLevelTimer(obj)
            if ~isempty(obj.LevelTimer) && isvalid(obj.LevelTimer)
                stop(obj.LevelTimer);
                delete(obj.LevelTimer);
            end
            obj.LevelTimer = [];
        end

        function toggleSidebar(obj)
            if isequal(obj.RootGrid.ColumnWidth{1}, 0)
                obj.RootGrid.ColumnWidth{1} = 280;
            else
                obj.RootGrid.ColumnWidth{1} = 0;
            end
        end

        function onProgress(obj, dlg, frac, msg)
            obj.ProgressLog(end+1) = string(msg);
            if ~isempty(dlg) && isvalid(dlg)
                dlg.Value = max(0, min(1, frac));
                dlg.Message = char(msg);
            end
            drawnow limitrate;
        end

        function reportError(obj, err)
            obj.LastError = string(err.message);
            obj.StatusLabel.Text = char("Error: " + err.message);
            if isvalid(obj.Figure) && strcmp(obj.Figure.Visible, 'on')
                uialert(obj.Figure, err.message, 'Speech Pipeline Studio');
            end
        end

        function showWarnings(obj)
            w = obj.ViewModel.Warnings;
            if isempty(w) || ~strcmp(obj.Figure.Visible, 'on'), return; end
            uialert(obj.Figure, char(strjoin(w, newline)), 'Warnings for this run', 'Icon', 'warning');
        end

        function showHelp(obj)
            msg = ["1. Choose an input (FSDD utterance, WAV file or microphone) and a checkpoint in the sidebar.", ...
                "2. Press Run: the whole pipeline is computed once by OceanNLP.Speech.Inference.RecognitionPipeline.", ...
                "3. Step through the stages with the stepper or Back / Next; hover a time axis to move the linked cursor, " + ...
                "click it to select a frame, Play to hear the conditioned audio with a moving playhead.", ...
                "4. Changing any setting marks the run stale; Run again to recompute.", ...
                "5. Workflow shows the corpus, training curves, model comparison and validation from saved data."];
            uialert(obj.Figure, char(strjoin(msg, newline)), 'Speech Pipeline Studio', 'Icon', 'info');
        end
    end
end
