classdef InformationSandbox < handle
    % INFORMATIONSANDBOX Main GUI application for the Information Theory Sandbox.
    %
    % This unified application integrates the mathematical simulation backends
    % for Kelly Criterion Gambling and Source Coding Data Compression with premium
    % Dark Mode interactive visual rendering.
    %
    % Usage:
    %   app = OceanViz.App.InformationSandbox();
    %   app.show();
    %
    % See also: OceanMath.Information.Processes.RaceSimulator,
    %           OceanMath.Information.Processes.KellyOptimizer,
    %           OceanMath.Information.Processes.GamblingSimulator,
    %           OceanViz.Strategies.RaceTrackRenderer,
    %           OceanViz.Strategies.CodeTreeRenderer

    properties (Access = private)
        % Main window handles
        Fig_
        TabGroup_
        TabGambling_
        TabCompression_

        % ---- Tab 1: Gambling & Kelly Properties ----
        % Layout containers
        GambControlPanel_
        GambCenterPanel_
        GambStatsPanel_
        
        % Left Sidebar controls
        SpinNumHorses_
        ChkAutoOdds_
        DropdownStrategy_
        StartingWealthField_
        BtnPlayPause_
        BtnStep_
        BtnReset_
        SliderFPS_
        LabelFPSVal_

        % Dynamic Control Slots (Horses 1 to 8)
        HorseSlots_ % struct array with: panel, probLabel, probSlider, oddsField
        BetSlots_   % struct array with: panel, betLabel, betSlider, betValText

        % Central Views
        RaceAx_     % UIAxes for horse race animation
        WealthAx_   % UIAxes for log-wealth vs time

        % Right Sidebar Metrics
        LblGambRaceCount_
        LblGambCurrentWealth_
        LblGambExpectedGrowth_
        LblGambEmpiricalGrowth_
        LblGambEntropy_
        LblGambDivergence_
        LblGambMarketFairness_   % sum(1/o_i) color-coded fairness indicator
        LblGambCashReserve_      % 1 - sum(b_i) unbet fraction
        LblGambHedgeBadge_       % Synthesizing-cash confirmation badge

        % Left Sidebar — Fair Odds Button
        BtnFairOdds_             % Resets odds to 1/p_i when AutoOdds is OFF

        % Tab 1 State variables
        K_Horses_ = 4
        HorseProbs_ = [0.4, 0.3, 0.2, 0.1]
        HorseOdds_ = [2.5, 3.33, 5.0, 10.0]
        BetFractions_ = [0.4, 0.3, 0.2, 0.1] % Default Kelly Proportional
        StartingWealth_ = 100.0
        CurrentWealth_ = 100.0
        WealthHistory_ = [100.0]
        RaceWinners_ = []
        RaceActive_ = false
        UpdatingSliders_ = false
        
        % Playback Timer State
        RaceTimer_
        TimerFPS_ = 5
        HorsePositions_ = []
        TargetDistances_ = []
        WinnerIdx_ = 1

        % Visualizers & Backends
        TrackRenderer_
        RaceSimulator_

        % ---- Tab 2: Data Compression Properties ----
        % Layout containers
        CompControlPanel_
        CompCenterPanel_
        CompStatsPanel_

        % Left Sidebar controls
        SpinAlphabetSize_
        DropdownCoder_
        DropdownExplore_

        % Dynamic Control Slots (Symbols 1 to 8)
        SymbolSlots_ % struct array with: panel, label, slider, valText

        % Central Views
        TreeAx_     % UIAxes for prefix tree
        TableStats_ % UITable for symbol codewords and statistics

        % Right Sidebar Metrics
        LblCompEntropy_
        LblCompLavg_
        LblCompEfficiency_
        LblCompRedundancy_
        LblCompKraftSum_
        LblCompStatus_

        % Tab 2 State variables
        K_Symbols_ = 4
        SymbolProbs_ = [0.4, 0.3, 0.2, 0.1]
        CoderType_ = 'Huffman' % 'Huffman' or 'Shannon-Fano'

        % Visualizers
        TreeRenderer_
    end

    properties (Access = private, Constant)
        % Premium design aesthetics (Dark Theme)
        COLOR_BG        = [0.13, 0.14, 0.17]  % Dark gunmetal background
        COLOR_PANEL     = [0.17, 0.19, 0.23]  % Sidebar panel background
        COLOR_BTN       = [0.22, 0.25, 0.32]  % Button background
        COLOR_BTN_PLAY  = [0.15, 0.65, 0.40]  % Green when playing/active
        COLOR_ACCENT    = [0.30, 0.65, 0.95]  % Ocean blue highlights
        COLOR_TEXT      = [0.93, 0.93, 0.93]  % Near-white text
        COLOR_DIM       = [0.55, 0.58, 0.63]  % Dim secondary text
        COLOR_GREEN     = [0.15, 0.65, 0.40]  % Positive metrics
        COLOR_RED       = [0.85, 0.30, 0.30]  % Loss/divergence metrics

        FIG_WIDTH       = 1200
        FIG_HEIGHT      = 750
        SIDEBAR_W       = 320
        STATS_W         = 240
    end

    methods
        function obj = InformationSandbox()
            % INFORMATIONSANDBOX Construct the main application and build layouts.
            obj.buildUI_();
            
            % Setup backends and renderers
            obj.TrackRenderer_ = OceanViz.Strategies.RaceTrackRenderer(obj.RaceAx_, obj.K_Horses_);
            obj.TreeRenderer_ = OceanViz.Strategies.CodeTreeRenderer(obj.TreeAx_);
            
            % Build high-precision playback timer
            obj.buildTimer_();

            % Initialize components state
            obj.resetGamblingSimulation_();
            obj.updateCompressionCoding_();
        end

        function show(obj)
            % SHOW Make the Sandbox application window visible.
            if ~isempty(obj.Fig_) && isvalid(obj.Fig_)
                figure(obj.Fig_);
            end
        end

        function delete(obj)
            % DELETE Robust teardown of timers and window handles.
            obj.stopTimer_();
            if ~isempty(obj.RaceTimer_) && isvalid(obj.RaceTimer_)
                delete(obj.RaceTimer_);
            end
            if ~isempty(obj.Fig_) && isvalid(obj.Fig_)
                delete(obj.Fig_);
            end
        end
    end

    % ================================================================== %
    % UI Construction Methods
    % ================================================================== %
    methods (Access = private)
        function buildUI_(obj)
            scr = get(0, 'ScreenSize');
            x0  = (scr(3) - obj.FIG_WIDTH)  / 2;
            y0  = (scr(4) - obj.FIG_HEIGHT) / 2;

            obj.Fig_ = uifigure( ...
                'Name',            'Information Theory Sandbox — Ocean Library', ...
                'Position',        [x0, y0, obj.FIG_WIDTH, obj.FIG_HEIGHT], ...
                'Color',           obj.COLOR_BG, ...
                'Resize',          'off', ...
                'CloseRequestFcn', @(~,~) obj.onClose_());

            % Main Tab Group
            obj.TabGroup_ = uitabgroup(obj.Fig_, ...
                'Position', [0, 0, obj.FIG_WIDTH, obj.FIG_HEIGHT]);

            obj.TabGambling_ = uitab(obj.TabGroup_, ...
                'Title', 'Information & Gambling (Kelly Criterion)', ...
                'BackgroundColor', obj.COLOR_BG);

            obj.TabCompression_ = uitab(obj.TabGroup_, ...
                'Title', 'Data Compression (Source Coding)', ...
                'BackgroundColor', obj.COLOR_BG);

            % Build individual tab layouts
            obj.buildTabGambling_();
            obj.buildTabCompression_();
        end

        % ---- Tab 1: Gambling Layout Builder ----
        function buildTabGambling_(obj)
            % Left control sidebar panel
            obj.GambControlPanel_ = uipanel(obj.TabGambling_, ...
                'Position', [10, 10, obj.SIDEBAR_W, 690], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none', ...
                'Scrollable', 'on');

            % Center visual view container
            obj.GambCenterPanel_ = uipanel(obj.TabGambling_, ...
                'Position', [obj.SIDEBAR_W + 20, 10, obj.FIG_WIDTH - obj.SIDEBAR_W - obj.STATS_W - 40, 690], ...
                'BackgroundColor', obj.COLOR_BG, ...
                'BorderType', 'none');

            % Right stats sidebar panel
            obj.GambStatsPanel_ = uipanel(obj.TabGambling_, ...
                'Position', [obj.FIG_WIDTH - obj.STATS_W - 10, 10, obj.STATS_W, 690], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none');

            % Left Controls Layout inside Control Panel (Absolute coords for simplicity & speed)
            yPos = 650;
            
            % Title
            uilabel(obj.GambControlPanel_, ...
                'Position', [15, yPos, 280, 25], ...
                'Text', 'Simulation Parameters', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 15, ...
                'FontWeight', 'bold');
            
            % Divider
            yPos = yPos - 10;
            uilabel(obj.GambControlPanel_, 'Position', [15, yPos, 280, 1], 'BackgroundColor', obj.COLOR_DIM .* 0.5, 'Text', '');

            % Spinner for horse count
            yPos = yPos - 35;
            uilabel(obj.GambControlPanel_, ...
                'Position', [15, yPos, 140, 25], ...
                'Text', 'Number of Horses:', ...
                'FontColor', obj.COLOR_TEXT, ...
                'FontSize', 11);
            obj.SpinNumHorses_ = uispinner(obj.GambControlPanel_, ...
                'Position', [160, yPos, 100, 25], ...
                'Limits', [2, 8], ...
                'Value', obj.K_Horses_, ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(s, e) obj.onNumHorsesChanged_(s.Value));

            % Auto fair-odds checkbox
            yPos = yPos - 30;
            obj.ChkAutoOdds_ = uicheckbox(obj.GambControlPanel_, ...
                'Position', [15, yPos, 250, 25], ...
                'Text', 'Auto-calculate Fair Odds (o_i = 1 / p_i)', ...
                'Value', true, ...
                'FontColor', obj.COLOR_TEXT, ...
                'FontSize', 11, ...
                'ValueChangedFcn', @(s, e) obj.onAutoOddsChanged_(s.Value));

            % Fair Odds reset button — only active when AutoOdds is OFF
            yPos = yPos - 28;
            obj.BtnFairOdds_ = uibutton(obj.GambControlPanel_, ...
                'Position', [15, yPos, 160, 22], ...
                'Text', 'Reset to Fair Odds', ...
                'FontSize', 10, ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_DIM, ...
                'Enable', 'off', ...
                'ButtonPushedFcn', @(~,~) obj.onFairOddsPressed_());

            % Dynamic probability control slots
            yPos = yPos - 20;
            obj.HorseSlots_ = [];
            for i = 1:8
                slot = struct();
                yPos = yPos - 63;   % Phase C: 2-row slot (58px) + 5px gap

                % ---- Two-row slot panel (Phase C) ----
                slot.panel = uipanel(obj.GambControlPanel_, ...
                    'Position', [10, yPos, 285, 58], ...
                    'BackgroundColor', obj.COLOR_PANEL, ...
                    'BorderType', 'none', ...
                    'Visible', 'off');

                % ROW 1 (top): Probability row — label | prob slider | r_i readout
                slot.probLabel = uilabel(slot.panel, ...
                    'Position', [5, 38, 76, 15], ...
                    'Text', sprintf('H%d  p=0.250', i), ...
                    'FontColor', obj.COLOR_TEXT, ...
                    'FontSize', 10, ...
                    'FontWeight', 'bold');

                slot.probSlider = uislider(slot.panel, ...
                    'Position', [84, 47, 120, 3], ...
                    'Limits', [0, 1], ...
                    'Value', 0.25, ...
                    'MajorTicks', [], ...
                    'MinorTicks', [], ...
                    'ValueChangedFcn', @(s, e) obj.onHorseProbChanged_(i, s.Value));

                % Implied probability readout: r_i = 1/o_i
                % Direct visual comparison between p_i (what you believe)
                % and r_i (what the bookie implies). Equal => fair market.
                slot.impliedProbLabel = uilabel(slot.panel, ...
                    'Position', [210, 38, 72, 15], ...
                    'Text', 'r=0.250', ...
                    'FontColor', obj.COLOR_GREEN, ...
                    'FontSize', 10, ...
                    'FontWeight', 'bold', ...
                    'HorizontalAlignment', 'right');

                % ROW 2 (bottom): Odds row — label | odds slider | odds field
                uilabel(slot.panel, ...
                    'Position', [5, 7, 32, 14], ...
                    'Text', 'Odds:', ...
                    'FontColor', obj.COLOR_DIM, ...
                    'FontSize', 9);

                % Phase C1: Odds slider — convenience control for manual tuning
                % Range 1.01–20 covers most practical cases; use edit field for extremes
                slot.oddsSlider = uislider(slot.panel, ...
                    'Position', [40, 14, 145, 3], ...
                    'Limits', [1.01, 20.0], ...
                    'Value', 4.0, ...
                    'MajorTicks', [], ...
                    'MinorTicks', [], ...
                    'Enable', 'off', ...
                    'ValueChangedFcn', @(s, e) obj.onHorseOddsChanged_(i, s.Value));

                slot.oddsField = uieditfield(slot.panel, 'numeric', ...
                    'Position', [193, 4, 52, 20], ...
                    'Value', 4.0, ...
                    'Limits', [1.01, 1000], ...
                    'BackgroundColor', obj.COLOR_BTN, ...
                    'FontColor', obj.COLOR_TEXT, ...
                    'FontSize', 9, ...
                    'Enable', 'off', ...
                    'ValueChangedFcn', @(s, e) obj.onHorseOddsChanged_(i, s.Value));

                obj.HorseSlots_ = [obj.HorseSlots_, slot];
            end

            % Strategy header
            yPos = yPos - 15;
            uilabel(obj.GambControlPanel_, 'Position', [15, yPos, 280, 1], 'BackgroundColor', obj.COLOR_DIM .* 0.5, 'Text', '');
            yPos = yPos - 25;
            uilabel(obj.GambControlPanel_, ...
                'Position', [15, yPos, 280, 25], ...
                'Text', 'Betting Strategy', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 13, ...
                'FontWeight', 'bold');

            % Strategy Dropdown
            yPos = yPos - 30;
            obj.DropdownStrategy_ = uidropdown(obj.GambControlPanel_, ...
                'Position', [15, yPos, 250, 25], ...
                'Items', {'Kelly Proportional (b_i = p_i)', 'Uniform (b_i = 1 / K)', 'All-In Favorite', 'Custom Betting Fractions'}, ...
                'ItemsData', {'Kelly', 'Uniform', 'Favorite', 'Custom'}, ...
                'Value', 'Kelly', ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(s, e) obj.onStrategyChanged_(s.Value));

            % Dynamic custom betting fractions slots
            yPos = yPos - 15;
            obj.BetSlots_ = [];
            for i = 1:8
                slot = struct();
                yPos = yPos - 30;
                slot.panel = uipanel(obj.GambControlPanel_, ...
                    'Position', [10, yPos, 285, 28], ...
                    'BackgroundColor', obj.COLOR_PANEL, ...
                    'BorderType', 'none', ...
                    'Visible', 'off');

                % Bet label
                slot.betLabel = uilabel(slot.panel, ...
                    'Position', [5, 4, 70, 20], ...
                    'Text', sprintf('Bet H%d:', i), ...
                    'FontColor', obj.COLOR_DIM, ...
                    'FontSize', 10);

                % Bet slider
                slot.betSlider = uislider(slot.panel, ...
                    'Position', [80, 12, 130, 3], ...
                    'Limits', [0, 1], ...
                    'Value', 0.25, ...
                    'MajorTicks', [], ...
                    'MinorTicks', [], ...
                    'Enable', 'off', ...
                    'ValueChangedFcn', @(s, e) obj.onBetFractionChanged_(i, s.Value));

                % Value display text
                slot.betValText = uilabel(slot.panel, ...
                    'Position', [220, 4, 50, 20], ...
                    'Text', '0.25', ...
                    'FontColor', obj.COLOR_TEXT, ...
                    'FontSize', 10, ...
                    'HorizontalAlignment', 'right');

                obj.BetSlots_ = [obj.BetSlots_, slot];
            end

            % Settings section
            yPos = yPos - 15;
            uilabel(obj.GambControlPanel_, 'Position', [15, yPos, 280, 1], 'BackgroundColor', obj.COLOR_DIM .* 0.5, 'Text', '');
            yPos = yPos - 25;
            uilabel(obj.GambControlPanel_, ...
                'Position', [15, yPos, 280, 25], ...
                'Text', 'Simulation Settings', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 13, ...
                'FontWeight', 'bold');

            % Starting wealth field
            yPos = yPos - 30;
            uilabel(obj.GambControlPanel_, ...
                'Position', [15, yPos, 140, 25], ...
                'Text', 'Starting Wealth ($):', ...
                'FontColor', obj.COLOR_TEXT, ...
                'FontSize', 11);
            obj.StartingWealthField_ = uieditfield(obj.GambControlPanel_, 'numeric', ...
                'Position', [160, yPos, 100, 25], ...
                'Value', 100.0, ...
                'Limits', [1, 1e6], ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(s, e) obj.onStartingWealthChanged_(s.Value));

            % Playback Buttons Panel
            yPos = yPos - 45;
            obj.BtnPlayPause_ = uibutton(obj.GambControlPanel_, ...
                'Position', [15, yPos, 80, 32], ...
                'Text', 'Play', ...
                'FontWeight', 'bold', ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~,~) obj.onPlayPausePressed_());

            obj.BtnStep_ = uibutton(obj.GambControlPanel_, ...
                'Position', [105, yPos, 80, 32], ...
                'Text', 'Step', ...
                'FontWeight', 'bold', ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~,~) obj.onStepPressed_());

            obj.BtnReset_ = uibutton(obj.GambControlPanel_, ...
                'Position', [195, yPos, 80, 32], ...
                'Text', 'Reset', ...
                'FontWeight', 'bold', ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~,~) obj.onResetPressed_());

            % FPS Slider
            yPos = yPos - 35;
            uilabel(obj.GambControlPanel_, ...
                'Position', [15, yPos, 120, 20], ...
                'Text', 'Simulation FPS Speed:', ...
                'FontColor', obj.COLOR_DIM, ...
                'FontSize', 10);
            obj.LabelFPSVal_ = uilabel(obj.GambControlPanel_, ...
                'Position', [195, yPos, 80, 20], ...
                'Text', '5 FPS', ...
                'FontColor', obj.COLOR_TEXT, ...
                'FontSize', 10, ...
                'HorizontalAlignment', 'right');

            yPos = yPos - 20;
            obj.SliderFPS_ = uislider(obj.GambControlPanel_, ...
                'Position', [25, yPos, 230, 3], ...
                'Limits', [1, 20], ...
                'Value', obj.TimerFPS_, ...
                'MajorTicks', [], ...
                'MinorTicks', [], ...
                'ValueChangedFcn', @(s, e) obj.onFPSChanged_(s.Value));

            % --- Center Plots (GambCenterPanel_) ---
            gW = obj.FIG_WIDTH - obj.SIDEBAR_W - obj.STATS_W - 40;
            % Race axes takes top 320 px
            obj.RaceAx_ = uiaxes(obj.GambCenterPanel_, ...
                'Position', [0, 360, gW, 320], ...
                'Color', [0.10, 0.12, 0.16]);
            title(obj.RaceAx_, 'Stochastic Horse Race Simulator', 'Color', obj.COLOR_TEXT);
            
            % Log-wealth axes takes bottom 330 px
            obj.WealthAx_ = uiaxes(obj.GambCenterPanel_, ...
                'Position', [0, 10, gW, 330], ...
                'Color', [0.10, 0.12, 0.16]);
            title(obj.WealthAx_, 'Log-Wealth Trajectory vs. Theoretical Kelly Slope', 'Color', obj.COLOR_TEXT);

            % --- Right Stats Panel (GambStatsPanel_) ---
            sy = 650;
            % Title
            uilabel(obj.GambStatsPanel_, ...
                'Position', [15, sy, 210, 25], ...
                'Text', 'Race Analysis', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 15, ...
                'FontWeight', 'bold');
            
            sy = sy - 10;
            uilabel(obj.GambStatsPanel_, 'Position', [15, sy, 210, 1], 'BackgroundColor', obj.COLOR_DIM .* 0.5, 'Text', '');

            % Race Counter
            sy = sy - 50;
            uilabel(obj.GambStatsPanel_, 'Position', [15, sy+30, 210, 16], 'Text', 'Races Completed', 'FontColor', obj.COLOR_DIM, 'FontSize', 10);
            obj.LblGambRaceCount_ = uilabel(obj.GambStatsPanel_, ...
                'Position', [15, sy, 210, 28], 'Text', '0', 'FontSize', 22, 'FontWeight', 'bold', 'FontColor', obj.COLOR_TEXT);

            % Current Wealth
            sy = sy - 75;
            uilabel(obj.GambStatsPanel_, 'Position', [15, sy+30, 210, 16], 'Text', 'Current Wealth', 'FontColor', obj.COLOR_DIM, 'FontSize', 10);
            obj.LblGambCurrentWealth_ = uilabel(obj.GambStatsPanel_, ...
                'Position', [15, sy, 210, 28], 'Text', '$ 100.00', 'FontSize', 22, 'FontWeight', 'bold', 'FontColor', obj.COLOR_GREEN);

            % Expected Growth (W)
            sy = sy - 70;
            uilabel(obj.GambStatsPanel_, 'Position', [15, sy+30, 210, 16], 'Text', 'Expected Growth (W)', 'FontColor', obj.COLOR_DIM, 'FontSize', 9);
            obj.LblGambExpectedGrowth_ = uilabel(obj.GambStatsPanel_, ...
                'Position', [15, sy, 210, 28], 'Text', '0.000 bits', 'FontSize', 18, 'FontWeight', 'bold', 'FontColor', obj.COLOR_ACCENT);

            % Empirical Growth (G)
            sy = sy - 70;
            uilabel(obj.GambStatsPanel_, 'Position', [15, sy+30, 210, 16], 'Text', 'Empirical Avg Growth (G)', 'FontColor', obj.COLOR_DIM, 'FontSize', 9);
            obj.LblGambEmpiricalGrowth_ = uilabel(obj.GambStatsPanel_, ...
                'Position', [15, sy, 210, 28], 'Text', '0.000 bits', 'FontSize', 18, 'FontWeight', 'bold', 'FontColor', obj.COLOR_ACCENT);

            % Source Entropy H(P)
            sy = sy - 70;
            uilabel(obj.GambStatsPanel_, 'Position', [15, sy+30, 210, 16], 'Text', 'True Entropy H(P)', 'FontColor', obj.COLOR_DIM, 'FontSize', 9);
            obj.LblGambEntropy_ = uilabel(obj.GambStatsPanel_, ...
                'Position', [15, sy, 210, 28], 'Text', '0.000 bits', 'FontSize', 18, 'FontWeight', 'bold', 'FontColor', obj.COLOR_TEXT);

            % Divergence D(P || B)
            sy = sy - 70;
            uilabel(obj.GambStatsPanel_, 'Position', [15, sy+30, 210, 16], 'Text', 'Betting Loss D(P || B)', 'FontColor', obj.COLOR_DIM, 'FontSize', 9);
            obj.LblGambDivergence_ = uilabel(obj.GambStatsPanel_, ...
                'Position', [15, sy, 210, 28], 'Text', '0.000 bits', 'FontSize', 18, 'FontWeight', 'bold', 'FontColor', obj.COLOR_TEXT);

            % ---- Phase B: Market Information Section ----

            % Market Fairness: sum(1/o_i) — 1.0 = fair, <1 = favourable, >1 = vig
            sy = sy - 65;
            uilabel(obj.GambStatsPanel_, 'Position', [15, sy+30, 210, 16], 'Text', 'Market Fairness  sum(1/o_i)', 'FontColor', obj.COLOR_DIM, 'FontSize', 9);
            obj.LblGambMarketFairness_ = uilabel(obj.GambStatsPanel_, ...
                'Position', [15, sy, 210, 28], 'Text', '1.000 (Fair)', 'FontSize', 16, 'FontWeight', 'bold', 'FontColor', obj.COLOR_GREEN);

            % Cash Reserve: 1 - sum(b_i)
            sy = sy - 60;
            uilabel(obj.GambStatsPanel_, 'Position', [15, sy+25, 210, 16], 'Text', 'Cash Reserve  (1 - sum(b))', 'FontColor', obj.COLOR_DIM, 'FontSize', 9);
            obj.LblGambCashReserve_ = uilabel(obj.GambStatsPanel_, ...
                'Position', [15, sy, 210, 24], 'Text', '0.000  (0.0%)', 'FontSize', 14, 'FontWeight', 'bold', 'FontColor', obj.COLOR_TEXT);

            % Synthesizing-Cash hedge badge
            sy = sy - 52;
            uilabel(obj.GambStatsPanel_, 'Position', [15, sy+24, 210, 16], 'Text', 'Synthesizing Cash?', 'FontColor', obj.COLOR_DIM, 'FontSize', 9);
            obj.LblGambHedgeBadge_ = uilabel(obj.GambStatsPanel_, ...
                'Position', [15, sy, 210, 22], 'Text', char(8212), 'FontSize', 12, 'FontWeight', 'bold', 'FontColor', obj.COLOR_DIM);
        end

        % ---- Tab 2: Compression Layout Builder ----
        function buildTabCompression_(obj)
            % Left control sidebar panel
            obj.CompControlPanel_ = uipanel(obj.TabCompression_, ...
                'Position', [10, 10, obj.SIDEBAR_W, 690], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none', ...
                'Scrollable', 'on');

            % Center visual view container
            obj.CompCenterPanel_ = uipanel(obj.TabCompression_, ...
                'Position', [obj.SIDEBAR_W + 20, 10, obj.FIG_WIDTH - obj.SIDEBAR_W - obj.STATS_W - 40, 690], ...
                'BackgroundColor', obj.COLOR_BG, ...
                'BorderType', 'none');

            % Right stats sidebar panel
            obj.CompStatsPanel_ = uipanel(obj.TabCompression_, ...
                'Position', [obj.FIG_WIDTH - obj.STATS_W - 10, 10, obj.STATS_W, 690], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none');

            % Left Controls Layout inside Control Panel
            yPos = 650;
            
            % Title
            uilabel(obj.CompControlPanel_, ...
                'Position', [15, yPos, 280, 25], ...
                'Text', 'Source Parameters', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 15, ...
                'FontWeight', 'bold');
            
            % Divider
            yPos = yPos - 10;
            uilabel(obj.CompControlPanel_, 'Position', [15, yPos, 280, 1], 'BackgroundColor', obj.COLOR_DIM .* 0.5, 'Text', '');

            % Spinner for alphabet size
            yPos = yPos - 35;
            uilabel(obj.CompControlPanel_, ...
                'Position', [15, yPos, 140, 25], ...
                'Text', 'Alphabet Size (K):', ...
                'FontColor', obj.COLOR_TEXT, ...
                'FontSize', 11);
            obj.SpinAlphabetSize_ = uispinner(obj.CompControlPanel_, ...
                'Position', [160, yPos, 100, 25], ...
                'Limits', [2, 8], ...
                'Value', obj.K_Symbols_, ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(s, e) obj.onAlphabetSizeChanged_(s.Value));

            % Dynamic symbol probability sliders
            yPos = yPos - 20;
            obj.SymbolSlots_ = [];
            for i = 1:8
                slot = struct();
                yPos = yPos - 35;
                slot.panel = uipanel(obj.CompControlPanel_, ...
                    'Position', [10, yPos, 285, 30], ...
                    'BackgroundColor', obj.COLOR_PANEL, ...
                    'BorderType', 'none', ...
                    'Visible', 'off');

                % Symbol label
                slot.label = uilabel(slot.panel, ...
                    'Position', [5, 4, 80, 20], ...
                    'Text', sprintf('S%d (p=%.2f):', i, 0.25), ...
                    'FontColor', obj.COLOR_TEXT, ...
                    'FontSize', 10);

                % Probability slider
                slot.slider = uislider(slot.panel, ...
                    'Position', [95, 13, 130, 3], ...
                    'Limits', [0, 1], ...
                    'Value', 0.25, ...
                    'MajorTicks', [], ...
                    'MinorTicks', [], ...
                    'ValueChangedFcn', @(s, e) obj.onSymbolProbChanged_(i, s.Value));

                % Value display text
                slot.valText = uilabel(slot.panel, ...
                    'Position', [230, 4, 45, 20], ...
                    'Text', '0.250', ...
                    'FontColor', obj.COLOR_DIM, ...
                    'FontSize', 10, ...
                    'HorizontalAlignment', 'right');

                obj.SymbolSlots_ = [obj.SymbolSlots_, slot];
            end

            % Coding Strategy header
            yPos = yPos - 15;
            uilabel(obj.CompControlPanel_, 'Position', [15, yPos, 280, 1], 'BackgroundColor', obj.COLOR_DIM .* 0.5, 'Text', '');
            yPos = yPos - 25;
            uilabel(obj.CompControlPanel_, ...
                'Position', [15, yPos, 280, 25], ...
                'Text', 'Compression Coder', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 13, ...
                'FontWeight', 'bold');

            % Coder Selection Dropdown
            yPos = yPos - 30;
            obj.DropdownCoder_ = uidropdown(obj.CompControlPanel_, ...
                'Position', [15, yPos, 250, 25], ...
                'Items', {'Huffman Coder (Optimal)', 'Shannon-Fano Coder'}, ...
                'ItemsData', {'Huffman', 'Shannon-Fano'}, ...
                'Value', 'Huffman', ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(s, e) obj.onCoderTypeChanged_(s.Value));

            % Path Exploration Section
            yPos = yPos - 15;
            uilabel(obj.CompControlPanel_, 'Position', [15, yPos, 280, 1], 'BackgroundColor', obj.COLOR_DIM .* 0.5, 'Text', '');
            yPos = yPos - 25;
            uilabel(obj.CompControlPanel_, ...
                'Position', [15, yPos, 280, 25], ...
                'Text', 'Path Exploration', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 13, ...
                'FontWeight', 'bold');

            % Explore Symbol Dropdown
            yPos = yPos - 30;
            obj.DropdownExplore_ = uidropdown(obj.CompControlPanel_, ...
                'Position', [15, yPos, 250, 25], ...
                'Items', {'None'}, ...
                'Value', 'None', ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(s, e) obj.onExploreSymbolChanged_(s.Value));

            % --- Center Plots (CompCenterPanel_) ---
            gW = obj.FIG_WIDTH - obj.SIDEBAR_W - obj.STATS_W - 40;
            % Tree axes takes top 410 px
            obj.TreeAx_ = uiaxes(obj.CompCenterPanel_, ...
                'Position', [0, 260, gW, 420], ...
                'Color', [0.10, 0.12, 0.16]);
            title(obj.TreeAx_, 'Source Code Prefix Tree Representation', 'Color', obj.COLOR_TEXT);
            
            % Codeword stats table takes bottom 230 px
            obj.TableStats_ = uitable(obj.CompCenterPanel_, ...
                'Position', [0, 10, gW, 230], ...
                'BackgroundColor', [0.15, 0.17, 0.22], ...
                'ForegroundColor', obj.COLOR_TEXT, ...
                'ColumnName', {'Symbol', 'Probability', 'Codeword', 'Length l_k', 'Ideal -log2(p_k)', 'Expected p_k * l_k'}, ...
                'ColumnWidth', {80, 90, 100, 90, 120, 130});

            % --- Right Stats Panel (CompStatsPanel_) ---
            sy = 650;
            % Title
            uilabel(obj.CompStatsPanel_, ...
                'Position', [15, sy, 210, 25], ...
                'Text', 'Coding Metrics', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 15, ...
                'FontWeight', 'bold');
            
            sy = sy - 10;
            uilabel(obj.CompStatsPanel_, 'Position', [15, sy, 210, 1], 'BackgroundColor', obj.COLOR_DIM .* 0.5, 'Text', '');

            % Source Entropy
            sy = sy - 50;
            uilabel(obj.CompStatsPanel_, 'Position', [15, sy+30, 210, 16], 'Text', 'Source Entropy H(X)', 'FontColor', obj.COLOR_DIM, 'FontSize', 10);
            obj.LblCompEntropy_ = uilabel(obj.CompStatsPanel_, ...
                'Position', [15, sy, 210, 28], 'Text', '0.000 bits', 'FontSize', 22, 'FontWeight', 'bold', 'FontColor', obj.COLOR_TEXT);

            % Expected Length
            sy = sy - 75;
            uilabel(obj.CompStatsPanel_, 'Position', [15, sy+30, 210, 16], 'Text', 'Expected Length L_avg', 'FontColor', obj.COLOR_DIM, 'FontSize', 10);
            obj.LblCompLavg_ = uilabel(obj.CompStatsPanel_, ...
                'Position', [15, sy, 210, 28], 'Text', '0.000 bits', 'FontSize', 22, 'FontWeight', 'bold', 'FontColor', obj.COLOR_GREEN);

            % Coding Efficiency (eta)
            sy = sy - 70;
            uilabel(obj.CompStatsPanel_, 'Position', [15, sy+30, 210, 16], 'Text', 'Coding Efficiency (eta)', 'FontColor', obj.COLOR_DIM, 'FontSize', 9);
            obj.LblCompEfficiency_ = uilabel(obj.CompStatsPanel_, ...
                'Position', [15, sy, 210, 28], 'Text', '0.0%', 'FontSize', 18, 'FontWeight', 'bold', 'FontColor', obj.COLOR_ACCENT);

            % Redundancy (R)
            sy = sy - 70;
            uilabel(obj.CompStatsPanel_, 'Position', [15, sy+30, 210, 16], 'Text', 'Redundancy R', 'FontColor', obj.COLOR_DIM, 'FontSize', 9);
            obj.LblCompRedundancy_ = uilabel(obj.CompStatsPanel_, ...
                'Position', [15, sy, 210, 28], 'Text', '0.000 bits', 'FontSize', 18, 'FontWeight', 'bold', 'FontColor', obj.COLOR_RED);

            % Kraft Inequality Sum
            sy = sy - 70;
            uilabel(obj.CompStatsPanel_, 'Position', [15, sy+30, 210, 16], 'Text', 'Kraft Inequality Sum', 'FontColor', obj.COLOR_DIM, 'FontSize', 9);
            obj.LblCompKraftSum_ = uilabel(obj.CompStatsPanel_, ...
                'Position', [15, sy, 210, 28], 'Text', '0.000', 'FontSize', 18, 'FontWeight', 'bold', 'FontColor', obj.COLOR_TEXT);

            % Status Bound
            sy = sy - 75;
            uilabel(obj.CompStatsPanel_, 'Position', [15, sy+30, 210, 16], 'Text', 'Shannon Limits Bounds', 'FontColor', obj.COLOR_DIM, 'FontSize', 9);
            obj.LblCompStatus_ = uilabel(obj.CompStatsPanel_, ...
                'Position', [15, sy, 210, 32], ...
                'Text', 'H(X) <= L_avg < H(X) + 1', ...
                'FontSize', 11, ...
                'FontWeight', 'bold', ...
                'FontColor', obj.COLOR_TEXT, ...
                'WordWrap', 'on');
        end
    end

    % ================================================================== %
    % Normalized Slider Callback Engine & Core GUI Logic
    % ================================================================== %
    methods (Access = private)
        % ---- Dynamic Sliders Re-allocator & Multiplicative Scaler ----
        
        function updateHorseSlidersUI_(obj)
            obj.UpdatingSliders_ = true;
            K = obj.K_Horses_;
            for i = 1:8
                slot = obj.HorseSlots_(i);
                if i <= K
                    slot.panel.Visible = 'on';
                    p_i = obj.HorseProbs_(i);
                    o_i = obj.HorseOdds_(i);
                    r_i = 1.0 / o_i;     % implied probability for horse i

                    % Row 1: probability
                    slot.probLabel.Text  = sprintf('H%d  p=%.3f', i, p_i);
                    slot.probSlider.Value = p_i;

                    % Phase C2: implied prob readout — green when r_i ≈ p_i
                    slot.impliedProbLabel.Text = sprintf('r=%.3f', r_i);
                    if abs(r_i - p_i) < 0.005
                        slot.impliedProbLabel.FontColor = obj.COLOR_GREEN;
                    else
                        slot.impliedProbLabel.FontColor = obj.COLOR_RED;
                    end

                    % Row 2: odds — sync both slider (clamped) and field
                    slot.oddsField.Value  = o_i;
                    slot.oddsSlider.Value = min(max(o_i, 1.01), 20.0);
                else
                    slot.panel.Visible = 'off';
                end
            end
            obj.UpdatingSliders_ = false;
        end

        function onHorseProbChanged_(obj, idx, newValue)
            if obj.UpdatingSliders_, return; end
            K = obj.K_Horses_;
            
            % Multiplicative Re-normalization Callback
            p_old = obj.HorseProbs_;
            delta = newValue - p_old(idx);
            
            p_new = p_old;
            p_new(idx) = newValue;
            
            otherIdx = setdiff(1:K, idx);
            S_other = sum(p_old(otherIdx));
            
            if S_other > 1e-6
                % Scale the others dynamically keeping ratios intact
                scaleFactor = (1 - newValue) / S_other;
                p_new(otherIdx) = p_old(otherIdx) * scaleFactor;
            else
                % If other horses had 0 prob, distribute inverse delta equally
                p_new(otherIdx) = (1 - newValue) / (K - 1);
            end
            
            % Safe bounds clip & absolute normalization
            p_new = max(0, min(1, p_new));
            p_new = p_new / sum(p_new);
            
            obj.HorseProbs_ = p_new;

            % Re-calculate odds if auto odds are enabled
            if obj.ChkAutoOdds_.Value
                obj.HorseOdds_ = 1.0 ./ obj.HorseProbs_;
            end

            obj.updateHorseSlidersUI_();
            obj.updateStrategyBettingFractions_();
            obj.updateKellyTheoreticalGrowth_();
        end

        function updateSymbolSlidersUI_(obj)
            obj.UpdatingSliders_ = true;
            K = obj.K_Symbols_;
            for i = 1:8
                slot = obj.SymbolSlots_(i);
                if i <= K
                    slot.panel.Visible = 'on';
                    slot.label.Text = sprintf('S%d (p=%.2f):', i, obj.SymbolProbs_(i));
                    slot.slider.Value = obj.SymbolProbs_(i);
                    slot.valText.Text = sprintf('%.3f', obj.SymbolProbs_(i));
                else
                    slot.panel.Visible = 'off';
                end
            end
            obj.UpdatingSliders_ = false;
        end

        function onSymbolProbChanged_(obj, idx, newValue)
            if obj.UpdatingSliders_, return; end
            K = obj.K_Symbols_;
            
            p_old = obj.SymbolProbs_;
            p_new = p_old;
            p_new(idx) = newValue;
            
            otherIdx = setdiff(1:K, idx);
            S_other = sum(p_old(otherIdx));
            
            if S_other > 1e-6
                scaleFactor = (1 - newValue) / S_other;
                p_new(otherIdx) = p_old(otherIdx) * scaleFactor;
            else
                p_new(otherIdx) = (1 - newValue) / (K - 1);
            end
            
            p_new = max(0, min(1, p_new));
            p_new = p_new / sum(p_new);
            
            obj.SymbolProbs_ = p_new;
            
            obj.updateSymbolSlidersUI_();
            obj.updateCompressionCoding_();
        end

        % ---- Preset Strategy Binder (Tab 1) ----
        
        function onStrategyChanged_(obj, strategy)
            K = obj.K_Horses_;
            switch strategy
                case 'Kelly'
                    % Proportional betting
                    obj.BetFractions_ = obj.HorseProbs_;
                    obj.disableCustomBetSliders_(true);
                case 'Uniform'
                    obj.BetFractions_ = ones(1, K) / K;
                    obj.disableCustomBetSliders_(true);
                case 'Favorite'
                    [~, favIdx] = max(obj.HorseProbs_);
                    obj.BetFractions_ = zeros(1, K);
                    obj.BetFractions_(favIdx) = 1.0;
                    obj.disableCustomBetSliders_(true);
                case 'Custom'
                    % Keep existing custom bet fractions
                    obj.disableCustomBetSliders_(false);
            end
            
            obj.updateBetSlidersUI_();
            obj.updateKellyTheoreticalGrowth_();
        end

        function updateStrategyBettingFractions_(obj)
            % Re-calculates bets when probabilities change if strategy isn't custom
            strat = obj.DropdownStrategy_.Value;
            if ~strcmp(strat, 'Custom')
                obj.onStrategyChanged_(strat);
            end
        end

        function updateBetSlidersUI_(obj)
            obj.UpdatingSliders_ = true;
            K = obj.K_Horses_;
            for i = 1:8
                slot = obj.BetSlots_(i);
                if i <= K
                    slot.panel.Visible = 'on';
                    slot.betSlider.Value = obj.BetFractions_(i);
                    slot.betValText.Text = sprintf('%.3f', obj.BetFractions_(i));
                else
                    slot.panel.Visible = 'off';
                end
            end
            obj.UpdatingSliders_ = false;
        end

        function disableCustomBetSliders_(obj, tf)
            K = obj.K_Horses_;
            for i = 1:8
                if tf
                    obj.BetSlots_(i).betSlider.Enable = 'off';
                    obj.BetSlots_(i).betLabel.FontColor = obj.COLOR_DIM;
                else
                    if i <= K
                        obj.BetSlots_(i).betSlider.Enable = 'on';
                        obj.BetSlots_(i).betLabel.FontColor = obj.COLOR_TEXT;
                    else
                        obj.BetSlots_(i).betSlider.Enable = 'off';
                    end
                end
            end
        end

        function onBetFractionChanged_(obj, idx, newValue)
            if obj.UpdatingSliders_, return; end
            K = obj.K_Horses_;
            
            b_new = obj.BetFractions_;
            b_new(idx) = newValue;
            
            % Kelly constraint: sum(betting_fractions) <= 1.0
            sumB = sum(b_new);
            if sumB > 1.0
                % Adjust the others down proportionally to fit inside the wealth bounds
                remainder = 1.0 - newValue;
                otherIdx = setdiff(1:K, idx);
                sumOther = sum(obj.BetFractions_(otherIdx));
                if sumOther > 1e-6
                    b_new(otherIdx) = obj.BetFractions_(otherIdx) * (remainder / sumOther);
                else
                    b_new(otherIdx) = remainder / (K - 1);
                end
            end
            
            b_new = max(0, min(1, b_new));
            obj.BetFractions_ = b_new;
            
            obj.updateBetSlidersUI_();
            obj.updateKellyTheoreticalGrowth_();
        end

        % ---- Tab 1 Controls callbacks ----

        function onNumHorsesChanged_(obj, newK)
            obj.K_Horses_ = newK;
            
            % Re-initialize probabilities uniformly
            obj.HorseProbs_ = ones(1, newK) / newK;
            obj.HorseOdds_ = 1.0 ./ obj.HorseProbs_;

            % BUG-5 FIX: Always pre-seed BetFractions_ to the new K before
            % calling updateStrategyBettingFractions_. The 'Custom' strategy
            % path does NOT resize the array, so without this a K=4->K=2
            % transition leaves a stale 4-element array that causes indexing
            % errors on the next race.
            obj.BetFractions_ = obj.HorseProbs_;

            % Rebuild visual tracks
            obj.TrackRenderer_ = OceanViz.Strategies.RaceTrackRenderer(obj.RaceAx_, newK);
            
            obj.updateHorseSlidersUI_();
            obj.updateStrategyBettingFractions_();
            obj.resetGamblingSimulation_();
        end

        function onAutoOddsChanged_(obj, value)
            for i = 1:8
                if value
                    % AutoOdds ON: lock both edit field and slider
                    obj.HorseSlots_(i).oddsField.Enable  = 'off';
                    obj.HorseSlots_(i).oddsSlider.Enable = 'off';  % Phase C1
                else
                    if i <= obj.K_Horses_
                        % AutoOdds OFF: enable both for manual tuning
                        obj.HorseSlots_(i).oddsField.Enable  = 'on';
                        obj.HorseSlots_(i).oddsSlider.Enable = 'on';  % Phase C1
                    end
                end
            end
            % Phase B4b: Toggle Fair Odds button availability
            if value
                obj.BtnFairOdds_.Enable    = 'off';
                obj.BtnFairOdds_.FontColor = obj.COLOR_DIM;
            else
                obj.BtnFairOdds_.Enable    = 'on';
                obj.BtnFairOdds_.FontColor = obj.COLOR_TEXT;
            end
            if value
                obj.HorseOdds_ = 1.0 ./ obj.HorseProbs_;
                obj.updateHorseSlidersUI_();
                obj.updateKellyTheoreticalGrowth_();
            end
        end

        function onHorseOddsChanged_(obj, idx, value)
            % Phase C: guard against re-entrant calls when syncing slider <-> field
            if obj.UpdatingSliders_, return; end
            obj.UpdatingSliders_ = true;

            obj.HorseOdds_(idx) = value;

            % Sync both controls and the implied-probability readout
            slot = obj.HorseSlots_(idx);
            slot.oddsField.Value  = value;
            slot.oddsSlider.Value = min(max(value, 1.01), 20.0);

            % Phase C2: update r_i = 1/o_i readout with mismatch colour
            r_i = 1.0 / value;
            slot.impliedProbLabel.Text = sprintf('r=%.3f', r_i);
            p_i = obj.HorseProbs_(idx);
            if abs(r_i - p_i) < 0.005
                slot.impliedProbLabel.FontColor = obj.COLOR_GREEN;
            else
                slot.impliedProbLabel.FontColor = obj.COLOR_RED;
            end

            obj.UpdatingSliders_ = false;

            obj.updateStrategyBettingFractions_();   % BUG-4 FIX kept
            obj.updateKellyTheoreticalGrowth_();
        end

        function onStartingWealthChanged_(obj, value)
            obj.StartingWealth_ = value;
            obj.resetGamblingSimulation_();
        end

        function onFPSChanged_(obj, value)
            obj.TimerFPS_ = round(value);
            obj.LabelFPSVal_.Text = sprintf('%d FPS', obj.TimerFPS_);
            
            wasPlaying = strcmp(obj.BtnPlayPause_.Text, 'Pause');
            obj.stopTimer_();
            obj.buildTimer_();
            if wasPlaying
                obj.startTimer_();
            end
        end

        function onFairOddsPressed_(obj)
            % Phase B4 callback: snap all odds back to fair values (o_i = 1/p_i)
            obj.HorseOdds_ = 1.0 ./ obj.HorseProbs_;
            obj.updateHorseSlidersUI_();
            obj.updateStrategyBettingFractions_();
            obj.updateKellyTheoreticalGrowth_();
        end
    end

    % ================================================================== %
    % Gambling Core Backends & Timer Methods
    % ================================================================== %
    methods (Access = private)
        function buildTimer_(obj)
            period = round(1000 / obj.TimerFPS_) / 1000;
            obj.RaceTimer_ = timer( ...
                'ExecutionMode', 'fixedRate', ...
                'Period',        period, ...
                'TimerFcn',      @(~, ~) obj.timerTick_());
        end

        function startTimer_(obj)
            start(obj.RaceTimer_);
            obj.BtnPlayPause_.Text = 'Pause';
            obj.BtnPlayPause_.BackgroundColor = obj.COLOR_RED;
        end

        function stopTimer_(obj)
            if ~isempty(obj.RaceTimer_) && isvalid(obj.RaceTimer_)
                stop(obj.RaceTimer_);
            end
            obj.BtnPlayPause_.Text = 'Play';
            obj.BtnPlayPause_.BackgroundColor = obj.COLOR_BTN;
        end

        function resetGamblingSimulation_(obj)
            obj.stopTimer_();
            obj.CurrentWealth_ = obj.StartingWealth_;
            obj.WealthHistory_ = [obj.StartingWealth_];
            obj.RaceWinners_ = [];
            obj.RaceActive_ = false;
            
            % Reset track
            obj.HorsePositions_ = zeros(1, obj.K_Horses_);
            obj.TrackRenderer_.initializeTrack();
            obj.TrackRenderer_.updatePositions(obj.HorsePositions_);
            obj.TrackRenderer_.clearWinnerFlash();
            
            % Reset dual axes plots
            obj.updateLogWealthPlot_();
            
            % Update stats metrics UI
            obj.updateKellyTheoreticalGrowth_();
            obj.updateKellyStatsUI_();
        end

        function startNewRace_(obj)
            % BUG-2 FIX: Force-renormalize HorseProbs_ to guard against floating-point
            % drift accumulated across repeated multiplicative slider renormalization.
            % RaceSimulator rejects vectors whose sum deviates from 1.0 by > 1e-9.
            p = max(0, obj.HorseProbs_);
            obj.HorseProbs_ = p / sum(p);

            % 1. Create Race Simulator with current probabilities
            obj.RaceSimulator_ = OceanMath.Information.Processes.RaceSimulator(obj.HorseProbs_);
            
            % 2. Draw winner index beforehand to maintain mathematical guarantees
            obj.WinnerIdx_ = obj.RaceSimulator_.drawWinner();
            
            % 3. Set positions and distance targets
            K = obj.K_Horses_;
            obj.HorsePositions_ = zeros(1, K);
            obj.TargetDistances_ = 0.55 + 0.38 * rand(1, K);
            obj.TargetDistances_(obj.WinnerIdx_) = 1.00; % Predefined winner gets to exactly 1.00
            
            obj.TrackRenderer_.clearWinnerFlash();
            obj.RaceActive_ = true;
        end

        function timerTick_(obj)
            if ~obj.RaceActive_
                obj.startNewRace_();
            end

            % Advance positions slightly with standard random steps
            K = obj.K_Horses_;
            dx = 0.01 + 0.05 * rand(1, K);
            obj.HorsePositions_ = min(obj.TargetDistances_, obj.HorsePositions_ + dx);
            
            % Render positions
            obj.TrackRenderer_.updatePositions(obj.HorsePositions_);
            
            % Check if winner crossed finish line
            if obj.HorsePositions_(obj.WinnerIdx_) >= 1.00
                obj.raceFinished_();
            end
        end

        function raceFinished_(obj)
            obj.stopTimer_();
            obj.RaceActive_ = false;
            
            % Flash winning horse lane
            obj.TrackRenderer_.triggerWinnerFlash(obj.WinnerIdx_);
            
            % Calculate portfolio wealth update (unbet wealth is kept)
            b = obj.BetFractions_;
            o = obj.HorseOdds_;
            cash = 1.0 - sum(b);
            
            payoffFactor = b(obj.WinnerIdx_) * o(obj.WinnerIdx_) + cash;
            obj.CurrentWealth_ = obj.CurrentWealth_ * payoffFactor;
            
            % Handle early bankruptcy bounds
            if obj.CurrentWealth_ <= 1e-4
                obj.CurrentWealth_ = 0.0;
            end
            
            obj.WealthHistory_ = [obj.WealthHistory_, obj.CurrentWealth_];
            obj.RaceWinners_ = [obj.RaceWinners_, obj.WinnerIdx_];

            % Re-plot history
            obj.updateLogWealthPlot_();
            obj.updateKellyStatsUI_();
            
            if obj.CurrentWealth_ == 0
                uialert(obj.Fig_, 'Your portfolio has gone completely bankrupt! Reset to try again.', 'Bankruptcy Notification', 'Icon', 'warning');
            end
        end

        % ---- Playback buttons actions ----

        function onPlayPausePressed_(obj)
            if obj.CurrentWealth_ == 0
                uialert(obj.Fig_, 'Cannot simulate. You are bankrupt! Click Reset first.', 'Bankruptcy Warning');
                return;
            end
            
            isTimerActive = strcmp(obj.BtnPlayPause_.Text, 'Pause');
            if isTimerActive
                obj.stopTimer_();
            else
                obj.startTimer_();
            end
        end

        function onStepPressed_(obj)
            if obj.CurrentWealth_ == 0
                uialert(obj.Fig_, 'Cannot simulate. You are bankrupt! Click Reset first.', 'Bankruptcy Warning');
                return;
            end
            
            obj.stopTimer_();
            if ~obj.RaceActive_
                obj.startNewRace_();
            end
            
            % Fast-forward to the end of the race
            obj.HorsePositions_ = obj.TargetDistances_;
            obj.TrackRenderer_.updatePositions(obj.HorsePositions_);
            obj.raceFinished_();
        end

        function onResetPressed_(obj)
            obj.resetGamblingSimulation_();
        end

        % ---- Core Calculations Updates (Tab 1) ----

        function updateKellyTheoreticalGrowth_(obj)
            % Slice all state vectors to exactly K elements for safety
            K = obj.K_Horses_;
            p = obj.HorseProbs_(1:K);
            b = obj.BetFractions_(1:K);
            o = obj.HorseOdds_(1:K);

            % Expected Growth W
            W = OceanMath.Information.Processes.KellyOptimizer.computeExpectedGrowth(p, b, o);
            obj.LblGambExpectedGrowth_.Text = sprintf('%.3f bits', W);
            
            % Shannon Entropy H(P)
            H = OceanMath.Information.Processes.KellyOptimizer.entropy(p);
            obj.LblGambEntropy_.Text = sprintf('%.3f bits', H);
            
            % Betting KL Divergence Loss D(P || B)
            % Normalize subjective bets B so it maps correctly to probability distributions for KL
            sumB = sum(b);
            if sumB > 0
                b_dist = b / sumB;
                D = OceanMath.Information.Processes.KellyOptimizer.divergence(p, b_dist);
            else
                D = Inf;
            end
            
            if isinf(D)
                obj.LblGambDivergence_.Text = 'Inf (Zero Bet)';
            else
                obj.LblGambDivergence_.Text = sprintf('%.3f bits', D);
            end

            % ================================================================
            % Phase B: Market Information Metrics
            % ================================================================

            % B1 — Market Fairness: sum(1/o_i)
            %   = 1.0  → fair market  (synthesizing cash yields EXACTLY W=0)
            %   < 1.0  → favourable  (track take < 0, rare in practice)
            %   > 1.0  → vig/unfair  (house edge, W is reduced)
            r = 1.0 ./ o;
            marketFairness = sum(r);
            if abs(marketFairness - 1.0) < 0.005
                fairnessStr = sprintf('%.4f  (Fair)', marketFairness);
                obj.LblGambMarketFairness_.FontColor = obj.COLOR_GREEN;
            elseif marketFairness < 1.0
                fairnessStr = sprintf('%.4f  (Favourable)', marketFairness);
                obj.LblGambMarketFairness_.FontColor = obj.COLOR_ACCENT;
            else
                fairnessStr = sprintf('%.4f  (Vig / Unfair)', marketFairness);
                obj.LblGambMarketFairness_.FontColor = obj.COLOR_RED;
            end
            obj.LblGambMarketFairness_.Text = fairnessStr;

            % B2 — Cash Reserve: fraction of wealth held outside bets
            cashReserve = max(0, 1.0 - sumB);
            obj.LblGambCashReserve_.Text = sprintf('%.4f  (%.1f%%)', cashReserve, cashReserve * 100);
            if cashReserve < 1e-3
                obj.LblGambCashReserve_.FontColor = obj.COLOR_GREEN;   % fully invested
            else
                obj.LblGambCashReserve_.FontColor = obj.COLOR_TEXT;
            end

            % B3 — Synthesizing Cash badge
            %   Perfect hedge condition: b_i = 1/o_i for ALL i  (inverse-odds portfolio)
            %   When the market is fair, this yields W = 0 exactly — cash is synthesized.
            diffVec = abs(b - r);
            isFullyInvested = sumB > 0.99;
            isHedged = isFullyInvested && all(diffVec < 0.015);
            if isHedged
                obj.LblGambHedgeBadge_.Text       = 'YES  — Perfect Hedge!';
                obj.LblGambHedgeBadge_.FontColor  = obj.COLOR_GREEN;
            elseif isFullyInvested
                obj.LblGambHedgeBadge_.Text       = sprintf('No  (max |b-1/o| = %.3f)', max(diffVec));
                obj.LblGambHedgeBadge_.FontColor  = obj.COLOR_DIM;
            else
                obj.LblGambHedgeBadge_.Text       = sprintf('No  (%.1f%% held as cash)', cashReserve * 100);
                obj.LblGambHedgeBadge_.FontColor  = obj.COLOR_DIM;
            end
        end

        function updateKellyStatsUI_(obj)
            numRaces = length(obj.RaceWinners_);
            obj.LblGambRaceCount_.Text = sprintf('%d', numRaces);
            obj.LblGambCurrentWealth_.Text = sprintf('$ %.2f', obj.CurrentWealth_);
            
            % Empirical growth G
            if numRaces > 0 && obj.CurrentWealth_ > 0
                G = (log2(obj.CurrentWealth_) - log2(obj.StartingWealth_)) / numRaces;
                obj.LblGambEmpiricalGrowth_.Text = sprintf('%.3f bits', G);
            else
                obj.LblGambEmpiricalGrowth_.Text = '0.000 bits';
            end
        end

        function updateLogWealthPlot_(obj)
            ax = obj.WealthAx_;
            cla(ax);
            hold(ax, 'on');
            
            t = 0:(length(obj.WealthHistory_)-1);
            empiricalLogS = log2(obj.WealthHistory_);
            
            % Plot empirical trajectory
            plot(ax, t, empiricalLogS, 'LineWidth', 1.8, 'Color', obj.COLOR_ACCENT, ...
                'DisplayName', 'Empirical Log-Wealth log2(S_t)');
            
            % Plot theoretical Kelly growth slope
            W = OceanMath.Information.Processes.KellyOptimizer.computeExpectedGrowth( ...
                obj.HorseProbs_, obj.BetFractions_, obj.HorseOdds_);
            
            if ~isinf(W) && ~isnan(W)
                theoLogS = log2(obj.StartingWealth_) + W * t;
                plot(ax, t, theoLogS, 'r--', 'LineWidth', 1.5, ...
                    'DisplayName', 'Expected Growth slope W * t');
            end
            
            % Grid & aesthetics
            grid(ax, 'on');
            ax.XColor = obj.COLOR_DIM;
            ax.YColor = obj.COLOR_DIM;
            ax.Box = 'on';
            xlabel(ax, 'Races Run (Time Step)', 'Color', obj.COLOR_TEXT);
            ylabel(ax, 'Log2 Wealth (bits)', 'Color', obj.COLOR_TEXT);
            legend(ax, 'TextColor', obj.COLOR_TEXT, 'Color', obj.COLOR_PANEL, 'EdgeColor', 'none', 'Location', 'northwest');
            
            hold(ax, 'off');
        end
    end

    % ================================================================== %
    % Data Compression Core Backends Methods (Tab 2)
    % ================================================================== %
    methods (Access = private)
        % ---- Tab 2 Controls callbacks ----

        function onAlphabetSizeChanged_(obj, newK)
            obj.K_Symbols_ = newK;
            
            % Re-initialize probabilities uniformly
            obj.SymbolProbs_ = ones(1, newK) / newK;
            
            obj.updateSymbolSlidersUI_();
            obj.updateCompressionCoding_();
        end

        function onCoderTypeChanged_(obj, coder)
            obj.CoderType_ = coder;
            obj.updateCompressionCoding_();
        end

        function onExploreSymbolChanged_(obj, symbol)
            if strcmp(symbol, 'None')
                obj.TreeRenderer_.resetHighlights_();
            else
                obj.TreeRenderer_.highlightPathTo(symbol);
            end
        end

        % ---- Core Calculations Updates (Tab 2) ----

        function updateCompressionCoding_(obj)
            K = obj.K_Symbols_;
            symbols = arrayfun(@(i) sprintf('S%d', i), 1:K, 'UniformOutput', false);
            probs = obj.SymbolProbs_;

            % 1. Run source coding backend algorithm
            if strcmp(obj.CoderType_, 'Huffman')
                coder = OceanMath.Information.Coding.HuffmanCoder();
            else
                coder = OceanMath.Information.Coding.ShannonFanoCoder();
            end
            
            try
                % Encode symbol alphabet
                table = coder.encode(symbols(:), probs);
            catch ME
                % Fail-safe backup if some inputs are edge cases
                uialert(obj.Fig_, sprintf('Source coder error: %s', ME.message), 'Coding Error');
                return;
            end

            % 2. Render prefix-tree
            obj.TreeRenderer_.renderTree(table);
            
            % 3. Update interactive explore symbols list
            obj.DropdownExplore_.Items = ['None', symbols];
            obj.DropdownExplore_.Value = 'None';

            % 4. Render statistics UITable
            codewords = table.getCodewords();
            lengths = table.getLengths();
            optLengths = table.getOptimalLengths();
            
            tableData = cell(K, 6);
            for k = 1:K
                tableData{k, 1} = symbols{k};
                tableData{k, 2} = sprintf('%.3f', probs(k));
                tableData{k, 3} = codewords{k};
                tableData{k, 4} = lengths(k);
                tableData{k, 5} = sprintf('%.3f', optLengths(k));
                tableData{k, 6} = sprintf('%.3f', probs(k) * lengths(k));
            end
            obj.TableStats_.Data = tableData;

            % 5. Update metrics indicators (Right Sidebar)
            H = table.entropyBits();          % Shannon Entropy H(X)
            L = table.expectedLength();       % Expected codeword length L_avg
            eta = table.efficiency();         % Coding Efficiency
            R = table.redundancy();           % Coding Redundancy
            Ks = table.kraftSum();            % Kraft inequality sum

            obj.LblCompEntropy_.Text = sprintf('%.3f bits', H);
            obj.LblCompLavg_.Text = sprintf('%.3f bits', L);
            obj.LblCompEfficiency_.Text = sprintf('%.1f %%', eta * 100);
            obj.LblCompRedundancy_.Text = sprintf('%.3f bits', R);
            obj.LblCompKraftSum_.Text = sprintf('%.4f', Ks);

            % Update status alert text (Shannon Bounds display)
            if abs(Ks - 1.0) < 1e-9
                kraftStatus = 'Complete Prefix Code';
            elseif Ks < 1.0
                kraftStatus = 'Incomplete Prefix Code';
            else
                kraftStatus = 'Violates Kraft Inequality!';
            end
            
            obj.LblCompStatus_.Text = sprintf('%s\nLimits: %.3f <= %.3f < %.3f', ...
                kraftStatus, H, L, H + 1.0);
        end
    end

    % ================================================================== %
    % Application Closing & Helper Methods
    % ================================================================== %
    methods (Access = private)
        function onClose_(obj)
            obj.stopTimer_();
            delete(obj);
        end
    end
end
