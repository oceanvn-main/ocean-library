classdef InformationStudio < handle
    % INFORMATIONSTUDIO  Interactive simulation dashboard (pure OOP UIFigure).
    %
    % A self-contained MATLAB window that:
    %   - Renders the simulation via a ParticleVisualizer on a central canvas
    %   - Provides playback controls: Rewind, Step Back, Play/Pause, Step Fwd
    %   - Displays a live metrics sidebar (time step, current state, entropy rate)
    %
    % Built as plain MATLAB OOP code (no .mlapp binary) for full version control.
    %
    % Usage:
    %   P      = [0.7 0.3; 0.4 0.6];
    %   mc     = OceanMath.Information.Processes.MarkovChain(P);
    %   sim    = OceanMath.Simulation.MarkovSimulator(mc);
    %   hist   = OceanMath.Simulation.SimulationHistory(sim);
    %   mapper = OceanViz.Adapters.GridMapper2D(4, 4);
    %
    %   studio = OceanViz.App.InformationStudio(sim, hist, mapper);
    %   studio.show();
    %
    % Playback controls:
    %   ⏮  Rewind       — jump to the recorded start state (t=1)
    %   ⏪  Step Back    — scrub one step backward through SimulationHistory
    %   ⏯  Play/Pause   — auto-advance at 'PlaybackFPS' frames per second
    %   ⏩  Step Forward  — advance one step; fires sim.step() if at the end
    %
    % See also: OceanViz.Visualizers.ParticleVisualizer
    %           OceanMath.Simulation.SimulationHistory
    %           OceanMath.Simulation.ISimulator

    % ------------------------------------------------------------------ %
    properties (Access = private)
        % Injected dependencies (typed in constructor arguments block)
        Simulator_   % OceanMath.Simulation.ISimulator
        History_     % OceanMath.Simulation.SimulationHistory
        Mapper_      % OceanViz.Adapters.ISpatialMapper

        % Rendering
        Visualizer_  % OceanViz.Visualizers.ParticleVisualizer

        % Current playback step (1-indexed into SimulationHistory)
        CurrentStep_ (1,1) double = 1

        % MATLAB timer for Play mode
        PlayTimer_
        IsPlaying_   (1,1) logical = false

        % ---- UI Component handles ----
        Fig_
        CanvasAx_       % main simulation canvas (UIAxes)

        BtnRewind_
        BtnBack_
        BtnPlay_
        BtnStop_
        BtnReset_
        BtnRandPos_
        BtnForward_

        LblStep_        % "Step: 001"
        LblState_       % "State: 4"
        LblSpeed_       % playback speed label

        SliderSpeed_    % FPS slider

        PlotAx_         % convergence plot axes
        BarHandle_      % empirical frequency bars
        LineHandle_     % theoretical stationary mu line

        StatusBar_      % bottom status text
    end

    properties (Access = private, Constant)
        % ---- Design tokens ----
        COLOR_BG        = [0.13 0.14 0.17]   % dark gunmetal background
        COLOR_PANEL     = [0.17 0.19 0.23]   % sidebar panel
        COLOR_BTN       = [0.22 0.25 0.32]   % button face
        COLOR_BTN_PLAY  = [0.15 0.65 0.40]   % green when playing
        COLOR_ACCENT    = [0.30 0.65 0.95]   % ocean blue highlights
        COLOR_TEXT      = [0.93 0.93 0.93]   % near-white text
        COLOR_DIM       = [0.55 0.58 0.63]   % dim secondary text

        FIG_WIDTH       = 1350   % widened to fit convergence plot
        FIG_HEIGHT      = 620
        SIDEBAR_W       = 210
        TOOLBAR_H       = 64
        BTN_H           = 38
        BTN_ICON_SIZE   = 18
        DEFAULT_FPS     = 8      % default playback frames per second
    end

    % ================================================================== %
    methods
        function obj = InformationStudio(simulator, history, mapper, options)
            % INFORMATIONSTUDIO  Construct and wire the studio.
            %
            % INPUTS:
            %   simulator - ISimulator instance
            %   history   - SimulationHistory instance (must already be attached)
            %   mapper    - ISpatialMapper instance
            %
            % Name-Value:
            %   PlaybackFPS (double): auto-play speed. Default: 8.
            arguments
                simulator OceanMath.Simulation.ISimulator
                history   OceanMath.Simulation.SimulationHistory
                mapper    OceanViz.Adapters.ISpatialMapper
                options.PlaybackFPS (1,1) double {mustBePositive} = ...
                    OceanViz.App.InformationStudio.DEFAULT_FPS
            end

            obj.Simulator_ = simulator;
            obj.History_   = history;
            obj.Mapper_    = mapper;

            obj.buildUI_();
            obj.buildTimer_(options.PlaybackFPS);

            % Attach visualizer to canvas
            obj.Visualizer_ = OceanViz.Visualizers.ParticleVisualizer(mapper, history);
            obj.Visualizer_.attachToAxes(obj.CanvasAx_);
            % NOTE: we do NOT attach to simulator here — we drive playback manually
            % through CurrentStep_ so the studio controls scrubbing independently.

            obj.goToStep_(1);
        end

        function show(obj)
            % SHOW  Make the studio window visible.
            figure(obj.Fig_);
        end

        function delete(obj)
            % DELETE  Clean teardown — stop timer, close window.
            obj.stopPlayback_();
            if ~isempty(obj.PlayTimer_) && isvalid(obj.PlayTimer_)
                delete(obj.PlayTimer_);
            end
            if ~isempty(obj.Fig_) && isvalid(obj.Fig_)
                delete(obj.Fig_);
            end
        end
    end

    % ================================================================== %
    methods (Access = private)

        % ---- UI Construction ---- %

        function buildUI_(obj)
            % BUILDUI_  Construct the full window layout.

            scr = get(0, 'ScreenSize');
            x0  = (scr(3) - obj.FIG_WIDTH)  / 2;
            y0  = (scr(4) - obj.FIG_HEIGHT) / 2;

            obj.Fig_ = uifigure( ...
                'Name',            'Information Studio', ...
                'Position',        [x0, y0, obj.FIG_WIDTH, obj.FIG_HEIGHT], ...
                'Color',           obj.COLOR_BG, ...
                'Resize',          'on', ...
                'CloseRequestFcn', @(~,~) obj.onClose_());

            % --- Central canvas & Convergence Plot ---
            % We split the remaining width between canvas and plot
            remainingW = obj.FIG_WIDTH - obj.SIDEBAR_W;
            canvasW    = floor(remainingW * 0.55);
            plotW      = remainingW - canvasW;
            canvasH    = obj.FIG_HEIGHT - obj.TOOLBAR_H;

            obj.CanvasAx_ = uiaxes(obj.Fig_, ...
                'Position',  [0, obj.TOOLBAR_H, canvasW, canvasH], ...
                'Color',     [0.10 0.12 0.16], ...
                'XColor',    obj.COLOR_DIM, ...
                'YColor',    obj.COLOR_DIM, ...
                'Box',       'on');
            title(obj.CanvasAx_, 'Simulation Tracking', 'Color', obj.COLOR_TEXT);

            obj.PlotAx_ = uiaxes(obj.Fig_, ...
                'Position',  [canvasW, obj.TOOLBAR_H, plotW, canvasH], ...
                'Color',     [0.10 0.12 0.16], ...
                'XColor',    obj.COLOR_DIM, ...
                'YColor',    obj.COLOR_DIM, ...
                'Box',       'on');
            title(obj.PlotAx_, 'Live Convergence (\mu vs Empirical)', 'Color', obj.COLOR_TEXT);
            obj.setupPlot_();

            % --- Sidebar panel ---
            obj.buildSidebar_(remainingW);

            % --- Playback toolbar ---
            obj.buildToolbar_();

            % --- Status bar (sits inside the bottom toolbar band) ---
            obj.StatusBar_ = uilabel(obj.Fig_, ...
                'Position',           [8, 4, obj.FIG_WIDTH - obj.SIDEBAR_W - 16, 20], ...
                'Text',               'Ready.', ...
                'FontColor',          obj.COLOR_DIM, ...
                'FontSize',           11, ...
                'HorizontalAlignment','left');
        end

        function buildSidebar_(obj, x0)
            % BUILDSIDEBAR_  Metrics panel on the right side.
            sH = obj.FIG_HEIGHT - obj.TOOLBAR_H;
            yl = obj.TOOLBAR_H;

            % Background panel
            uipanel(obj.Fig_, ...
                'Position',         [x0, yl, obj.SIDEBAR_W, sH], ...
                'BackgroundColor',  obj.COLOR_PANEL, ...
                'BorderType',       'none');

            pad = 14;
            w   = obj.SIDEBAR_W - 2*pad;

            % Title
            uilabel(obj.Fig_, ...
                'Position',   [x0+pad, yl+sH-46, w, 28], ...
                'Text',       'Live Metrics', ...
                'FontSize',   13, ...
                'FontWeight', 'bold', ...
                'FontColor',  obj.COLOR_ACCENT);

            % Divider
            uilabel(obj.Fig_, ...
                'Position',   [x0+pad, yl+sH-54, w, 2], ...
                'BackgroundColor', obj.COLOR_ACCENT, ...
                'Text', '');

            % Step counter
            obj.addMetricBlock_(x0+pad, yl+sH-110, w, 'Time Step', '—');
            obj.LblStep_ = uilabel(obj.Fig_, ...
                'Position',   [x0+pad, yl+sH-110, w, 28], ...
                'Text',       '—', ...
                'FontSize',   22, ...
                'FontWeight', 'bold', ...
                'FontColor',  obj.COLOR_TEXT);

            % Current state
            obj.addMetricBlock_(x0+pad, yl+sH-190, w, 'Current State', '—');
            obj.LblState_ = uilabel(obj.Fig_, ...
                'Position',   [x0+pad, yl+sH-190, w, 28], ...
                'Text',       '—', ...
                'FontSize',   22, ...
                'FontWeight', 'bold', ...
                'FontColor',  [0.99 0.40 0.10]);   % particle orange

            % Speed label
            uilabel(obj.Fig_, ...
                'Position',   [x0+pad, yl+30, w, 18], ...
                'Text',       'Playback Speed (FPS)', ...
                'FontSize',   10, ...
                'FontColor',  obj.COLOR_DIM);

            obj.LblSpeed_ = uilabel(obj.Fig_, ...
                'Position',   [x0+pad+w-30, yl+30, 30, 18], ...
                'Text',       sprintf('%d', obj.DEFAULT_FPS), ...
                'FontSize',   10, ...
                'FontColor',  obj.COLOR_TEXT, ...
                'HorizontalAlignment', 'right');

            obj.SliderSpeed_ = uislider(obj.Fig_, ...
                'Position',   [x0+pad, yl+10, w, 3], ...
                'Limits',     [1, 30], ...
                'Value',      obj.DEFAULT_FPS, ...
                'MajorTicks', [], ...
                'MinorTicks', [], ...
                'ValueChangedFcn', @(s,~) obj.onSpeedChanged_(s.Value));
        end

        function addMetricBlock_(obj, x, y, w, label, ~)
            % ADDMETRICBLOCK_  Helper: render a small label above a value.
            uilabel(obj.Fig_, ...
                'Position',   [x, y+30, w, 16], ...
                'Text',       label, ...
                'FontSize',   9, ...
                'FontColor',  obj.COLOR_DIM);
        end

        function buildToolbar_(obj)
            % BUILDTOOLBAR_  Playback controls at the bottom of the canvas.
            labels = {'Rewind', 'Step Back', 'Play', 'Stop', 'Reset', 'Rand. Pos', 'Step Fwd'};
            cbs    = { ...
                @(~,~) obj.onRewind_(), ...
                @(~,~) obj.onStepBack_(), ...
                @(~,~) obj.onPlayPause_(), ...
                @(~,~) obj.onStop_(), ...
                @(~,~) obj.onReset_(), ...
                @(~,~) obj.onRandomPosition_(), ...
                @(~,~) obj.onStepForward_() };

            toolW  = obj.FIG_WIDTH - obj.SIDEBAR_W;
            btnW   = 80;
            gap    = 6;
            nBtns  = numel(labels);
            totalW = nBtns*btnW + (nBtns-1)*gap;
            x0     = (toolW - totalW) / 2;
            yc     = (obj.TOOLBAR_H - obj.BTN_H) / 2;

            % Background
            uipanel(obj.Fig_, ...
                'Position',        [0, 0, toolW, obj.TOOLBAR_H], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType',      'none');

            handles = cell(1, nBtns);
            for i = 1:nBtns
                xi = x0 + (i-1)*(btnW+gap);
                handles{i} = uibutton(obj.Fig_, ...
                    'Position',          [xi, yc, btnW, obj.BTN_H], ...
                    'Text',              labels{i}, ...
                    'FontSize',          11, ...
                    'BackgroundColor',   obj.COLOR_BTN, ...
                    'FontColor',         obj.COLOR_TEXT, ...
                    'ButtonPushedFcn',   cbs{i});
            end

            obj.BtnRewind_  = handles{1};
            obj.BtnBack_    = handles{2};
            obj.BtnPlay_    = handles{3};
            obj.BtnStop_    = handles{4};
            obj.BtnReset_   = handles{5};
            obj.BtnRandPos_ = handles{6};
            obj.BtnForward_ = handles{7};
        end

        % ---- Timer ---- %

        function buildTimer_(obj, fps)
            % BUILDTIMER_  Create the MATLAB timer that drives auto-play.
            % Period is rounded to the nearest millisecond to avoid warnings.
            period = round(1000 / fps) / 1000;
            obj.PlayTimer_ = timer( ...
                'ExecutionMode', 'fixedRate', ...
                'Period',        period, ...
                'TimerFcn',      @(~,~) obj.timerTick_());
        end

        function timerTick_(obj)
            % TIMERTICK_  Called on every timer fire during Play.
            obj.onStepForward_();
        end

        % ---- Playback Actions ---- %

        function goToStep_(obj, t)
            % GOTOTEP_  Navigate to a specific 1-indexed history step.
            t              = max(1, min(t, obj.History_.getLength()));
            obj.CurrentStep_ = t;
            obj.Visualizer_.updateToStep(t);
            obj.updateMetrics_();
        end

        function onRewind_(obj)
            obj.stopPlayback_();
            obj.goToStep_(1);
            obj.setStatus_('Rewound to start.');
        end

        function onStepBack_(obj)
            obj.stopPlayback_();
            obj.goToStep_(obj.CurrentStep_ - 1);
            obj.setStatus_(sprintf('Step %d', obj.CurrentStep_));
        end

        function onPlayPause_(obj)
            if obj.IsPlaying_
                obj.stopPlayback_();
                obj.setStatus_('Paused.');
            else
                obj.startPlayback_();
                obj.setStatus_('Playing...');
            end
        end

        function onStop_(obj)
            % ONSTOP_  Stop playback and freeze at current step.
            obj.stopPlayback_();
            obj.setStatus_('Paused (stopped).');
        end

        function onReset_(obj)
            % ONRESET_  Stop playback, rewind, and zero the convergence chart.
            obj.stopPlayback_();
            obj.goToStep_(1);
            K = obj.Simulator_.Process_.getNumStates();
            obj.BarHandle_.YData = zeros(K,1);
            obj.setStatus_('Simulation reset. Convergence chart cleared.');
        end

        function onRandomPosition_(obj)
            % ONRANDOMPOSITION_  Restart the simulation from a random start state.
            %
            % Stops playback, picks a uniformly random state, resets the
            % simulator to t=0 at that state, clears the history tape, and
            % re-records the new initial step — giving a fresh start.
            obj.stopPlayback_();
            K = obj.Simulator_.Process_.getNumStates();
            s = randi(K);

            % 1. Reset simulator to t=0 at the random position
            obj.Simulator_.reset(s);

            % 2. Wipe the history tape
            obj.History_.clearHistory();

            % 3. Record initial state as step 1 (teleport fires StateChanged)
            obj.Simulator_.teleport(s);

            % 4. Clear convergence chart
            obj.BarHandle_.YData = zeros(K, 1);

            % 5. Navigate view to step 1
            obj.goToStep_(1);
            obj.setStatus_(sprintf('New start: state %d (uniform random). Press Play to begin.', s));
        end

        function onStepForward_(obj)
            % If we're at the end of recorded history, fire a new sim step.
            if obj.CurrentStep_ >= obj.History_.getLength()
                obj.Simulator_.step();
            end
            obj.goToStep_(obj.CurrentStep_ + 1);
        end

        function startPlayback_(obj)
            obj.IsPlaying_ = true;
            obj.BtnPlay_.BackgroundColor = obj.COLOR_BTN_PLAY;
            start(obj.PlayTimer_);
        end

        function stopPlayback_(obj)
            if obj.IsPlaying_
                stop(obj.PlayTimer_);
            end
            obj.IsPlaying_               = false;
            obj.BtnPlay_.BackgroundColor = obj.COLOR_BTN;
        end

        % ---- Metric Updates ---- %

        function setupPlot_(obj)
            % SETUPPLOT_ Initialize the bars and theoretical line
            K = obj.Simulator_.Process_.getNumStates();
            if obj.Simulator_.Process_.isErgodic()
                mu  = obj.Simulator_.Process_.getStationaryDistribution();
                muV = mu.evaluate((1:K)');
            else
                muV = nan(K, 1);
            end

            hold(obj.PlotAx_, 'on');
            obj.BarHandle_ = bar(obj.PlotAx_, 1:K, zeros(K,1), ...
                'FaceColor', obj.COLOR_ACCENT, 'EdgeColor', 'none', 'FaceAlpha', 0.85);

            obj.LineHandle_ = plot(obj.PlotAx_, 1:K, muV, 'r--o', ...
                'LineWidth', 1.8, 'MarkerFaceColor', 'r', 'MarkerSize', 5);

            xlim(obj.PlotAx_, [0.4, K+0.6]);
            topY = max([0.25, max(muV)*1.5]);  % generous top margin to see peaks
            ylim(obj.PlotAx_, [0, topY]);
            xticks(obj.PlotAx_, 1:K);
            xticklabels(obj.PlotAx_, arrayfun(@(s) sprintf('s%d', s), 1:K, 'UniformOutput', false));
            grid(obj.PlotAx_, 'on');
            legend(obj.PlotAx_, {'Empirical', 'Theoretical \mu'}, 'TextColor', obj.COLOR_TEXT, 'Color', obj.COLOR_PANEL);
        end

        function updateMetrics_(obj)
            t = obj.CurrentStep_;
            s = obj.History_.getStateAtStep(t);
            obj.LblStep_.Text  = sprintf('%d', t-1);
            obj.LblState_.Text = sprintf('%d', s);

            % Update convergence plot
            seq = double(obj.History_.getStateSequence());
            seq = seq(1:t);  % only count up to current playback step
            K   = obj.Simulator_.Process_.getNumStates();
            empirical = histcounts(seq, 1:K+1)' / numel(seq);
            obj.BarHandle_.YData = empirical;
        end

        function setStatus_(obj, msg)
            if isvalid(obj.StatusBar_)
                obj.StatusBar_.Text = msg;
            end
        end

        % ---- Event Callbacks ---- %

        function onSpeedChanged_(obj, fps)
            fps = round(fps);
            obj.LblSpeed_.Text = sprintf('%d', fps);
            wasPlaying         = obj.IsPlaying_;
            obj.stopPlayback_();
            obj.PlayTimer_.Period = round(1000 / fps) / 1000;
            if wasPlaying
                obj.startPlayback_();
            end
        end

        function onClose_(obj)
            obj.stopPlayback_();
            delete(obj);
        end
    end
end
