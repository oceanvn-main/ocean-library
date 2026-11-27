classdef CardDeckSandbox < handle
    % CARDDECKSANDBOX Programmatic, responsive dark-mode sequential card deck gambling dashboard.
    
    properties
        FigureHandle
        
        % Layout Containers
        MainGridLayout
        LeftPanel
        CenterPanel
        RightPanel
        
        % Controls
        NumRedRSpinner
        NumBlackBSpinner
        OddsRedField
        OddsBlackField
        StartingWealthField
        StrategyDropdown
        BeliefDeviationSlider
        BeliefDeviationLabel
        BtnPlayPause
        BtnStep
        BtnReset
        BtnSimulate100
        
        % Central Canvas
        CardGridAx
        WealthAx
        
        % Metrics Panel Labels
        LblCardsDrawn
        LblRemaining
        LblCurrentPRed
        LblCurrentPBlack
        LblCurrentWealth
        LblTheoreticalFinal
        BadgeEquivalence
        
        % Simulator and Renderer
        Simulator
        Renderer
        
        % App State
        CurrentWealth
        WealthHistory % 1x(N+1) double
        ActiveBeta % [b_Red, b_Black]
        TimerPlayback
        IsPlaying = false
        
        % Graphic handles for trajectories
        EmpiricalPathHandle
        TheoreticalPathHandle
        PathsLinesHandles = []
        HoverTextHandle
        
        % Fast Sim 100 bounds for limits update
        Sim100MinY = []
        Sim100MaxY = []
        
        % Color theme tokens
        COLOR_BG = [0.08, 0.09, 0.12]
        COLOR_PANEL_BG = [0.12, 0.14, 0.18]
        COLOR_TEXT = [0.90, 0.90, 0.90]
        COLOR_SUBTEXT = [0.60, 0.65, 0.70]
        COLOR_CYAN = [0.00, 0.80, 0.85]
        COLOR_RED = [0.85, 0.30, 0.30]
        COLOR_GREEN = [0.15, 0.65, 0.40]
        COLOR_MUTED_GREEN = [0.08, 0.25, 0.18]
        COLOR_GRAY = [0.35, 0.35, 0.35]
        COLOR_ORANGE = [0.85, 0.50, 0.15]
    end
    
    methods
        function obj = CardDeckSandbox()
            % Construct the app programmatically in invisible state
            obj.FigureHandle = uifigure('Name', 'Card Deck Gambling & Shuffle Sandbox', ...
                'Color', obj.COLOR_BG, ...
                'Position', [100, 100, 1150, 750], ...
                'Visible', 'off');
            
            % Close request cleanup
            obj.FigureHandle.CloseRequestFcn = @(src, event) obj.closeApp();
            
            % Setup Grid Layout
            obj.MainGridLayout = uigridlayout(obj.FigureHandle, [2, 3], ...
                'RowHeight', {40, '1x'}, ...
                'ColumnWidth', {280, '1x', 280}, ...
                'BackgroundColor', obj.COLOR_BG);
            
            % Header Title
            titleLabel = uilabel(obj.MainGridLayout, ...
                'Text', 'CARD DECK GAMBLING & SHUFFLE SANDBOX', ...
                'FontName', 'Segoe UI', ...
                'FontSize', 18, ...
                'FontWeight', 'bold', ...
                'FontColor', obj.COLOR_CYAN, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'center');
            titleLabel.Layout.Row = 1;
            titleLabel.Layout.Column = [1, 3];
            
            % 1. Left Controls Panel
            obj.LeftPanel = uipanel(obj.MainGridLayout, ...
                'Title', 'CONTROLS', ...
                'BackgroundColor', obj.COLOR_PANEL_BG, ...
                'ForegroundColor', obj.COLOR_TEXT, ...
                'FontName', 'Segoe UI', ...
                'FontWeight', 'bold', ...
                'FontSize', 12);
            obj.LeftPanel.Layout.Row = 2;
            obj.LeftPanel.Layout.Column = 1;
            
            ctrlGrid = uigridlayout(obj.LeftPanel, [11, 2], ...
                'RowHeight', {30, 30, 30, 30, 30, 40, 35, 25, 45, 45, '1x'}, ...
                'ColumnWidth', {'1.1x', '1x'}, ...
                'BackgroundColor', obj.COLOR_PANEL_BG);
            
            % Inputs
            uilabel(ctrlGrid, 'Text', 'Red Cards (R):', 'FontColor', obj.COLOR_TEXT);
            obj.NumRedRSpinner = uispinner(ctrlGrid, 'Limits', [0, 50], 'Value', 26, ...
                'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT);
            
            uilabel(ctrlGrid, 'Text', 'Black Cards (B):', 'FontColor', obj.COLOR_TEXT);
            obj.NumBlackBSpinner = uispinner(ctrlGrid, 'Limits', [0, 50], 'Value', 26, ...
                'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT);
            
            uilabel(ctrlGrid, 'Text', 'Red Odds (o_Red):', 'FontColor', obj.COLOR_TEXT);
            obj.OddsRedField = uieditfield(ctrlGrid, 'numeric', 'Value', 2.0, ...
                'Limits', [1.01, 10.0], 'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT);
            
            uilabel(ctrlGrid, 'Text', 'Black Odds (o_Black):', 'FontColor', obj.COLOR_TEXT);
            obj.OddsBlackField = uieditfield(ctrlGrid, 'numeric', 'Value', 2.0, ...
                'Limits', [1.01, 10.0], 'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT);
            
            uilabel(ctrlGrid, 'Text', 'Start Wealth (S_0):', 'FontColor', obj.COLOR_TEXT);
            obj.StartingWealthField = uieditfield(ctrlGrid, 'numeric', 'Value', 100.0, ...
                'Limits', [1.0, 10000.0], 'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT);
            
            uilabel(ctrlGrid, 'Text', 'Betting Strategy:', 'FontColor', obj.COLOR_TEXT);
            obj.StrategyDropdown = uidropdown(ctrlGrid, ...
                'Items', {'Kelly Proportional (b = p)', 'Constant Fraction (b = alpha)', 'Subjective Belief Mismatch'}, ...
                'Value', 'Kelly Proportional (b = p)', ...
                'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT);
            
            uilabel(ctrlGrid, 'Text', 'Slider Config:', 'FontColor', obj.COLOR_TEXT);
            obj.BeliefDeviationSlider = uislider(ctrlGrid, 'Limits', [0, 1], 'Value', 0, ...
                'Enable', 'off');
            
            obj.BeliefDeviationLabel = uilabel(ctrlGrid, 'Text', '0% Deviation', ...
                'FontColor', obj.COLOR_SUBTEXT, 'HorizontalAlignment', 'center', ...
                'FontWeight', 'bold');
            obj.BeliefDeviationLabel.Layout.Column = [1, 2];
            
            % Playback Buttons
            obj.BtnPlayPause = uibutton(ctrlGrid, 'Text', 'Play', ...
                'ButtonPushedFcn', @(btn, event) obj.togglePlay(), ...
                'BackgroundColor', [0.15, 0.50, 0.80], 'FontColor', [1,1,1], 'FontWeight', 'bold');
            
            obj.BtnStep = uibutton(ctrlGrid, 'Text', 'Step', ...
                'ButtonPushedFcn', @(btn, event) obj.stepPressed(), ...
                'BackgroundColor', [0.25, 0.27, 0.32], 'FontColor', [1,1,1], 'FontWeight', 'bold');
            
            obj.BtnReset = uibutton(ctrlGrid, 'Text', 'Reset', ...
                'ButtonPushedFcn', @(btn, event) obj.resetSimulation(), ...
                'BackgroundColor', [0.60, 0.25, 0.25], 'FontColor', [1,1,1], 'FontWeight', 'bold');
            
            obj.BtnSimulate100 = uibutton(ctrlGrid, 'Text', 'Simulate 100', ...
                'ButtonPushedFcn', @(btn, event) obj.runSimulate100(), ...
                'BackgroundColor', obj.COLOR_GREEN, 'FontColor', [1,1,1], 'FontWeight', 'bold');
            
            % 2. Center Panel
            obj.CenterPanel = uipanel(obj.MainGridLayout, ...
                'Title', 'ANIMATION & WEALTH TRAJECTORY CANVAS', ...
                'BackgroundColor', obj.COLOR_BG, ...
                'ForegroundColor', obj.COLOR_TEXT, ...
                'FontName', 'Segoe UI', ...
                'FontWeight', 'bold', ...
                'FontSize', 12);
            obj.CenterPanel.Layout.Row = 2;
            obj.CenterPanel.Layout.Column = 2;
            
            canvasGrid = uigridlayout(obj.CenterPanel, [2, 1], ...
                'RowHeight', {'1.1x', '1x'}, ...
                'BackgroundColor', obj.COLOR_BG);
            
            obj.CardGridAx = uiaxes(canvasGrid);
            obj.CardGridAx.Layout.Row = 1;
            obj.CardGridAx.Layout.Column = 1;
            
            obj.WealthAx = uiaxes(canvasGrid);
            obj.WealthAx.Layout.Row = 2;
            obj.WealthAx.Layout.Column = 1;
            
            % 3. Right Metrics Panel
            obj.RightPanel = uipanel(obj.MainGridLayout, ...
                'Title', 'METRICS & THEORETICAL DUALITY', ...
                'BackgroundColor', obj.COLOR_PANEL_BG, ...
                'ForegroundColor', obj.COLOR_TEXT, ...
                'FontName', 'Segoe UI', ...
                'FontWeight', 'bold', ...
                'FontSize', 12);
            obj.RightPanel.Layout.Row = 2;
            obj.RightPanel.Layout.Column = 3;
            
            metricsGrid = uigridlayout(obj.RightPanel, [8, 1], ...
                'RowHeight', {35, 35, 35, 35, 35, 35, 60, '1x'}, ...
                'BackgroundColor', obj.COLOR_PANEL_BG);
            
            obj.LblCardsDrawn = uilabel(metricsGrid, 'Text', 'Cards Drawn: 0 / 52', 'FontColor', obj.COLOR_TEXT, 'FontSize', 12);
            obj.LblRemaining = uilabel(metricsGrid, 'Text', 'Remaining Cards: 52', 'FontColor', obj.COLOR_TEXT, 'FontSize', 12);
            obj.LblCurrentPRed = uilabel(metricsGrid, 'Text', 'Current P(Red): 50.0%', 'FontColor', obj.COLOR_TEXT, 'FontSize', 12);
            obj.LblCurrentPBlack = uilabel(metricsGrid, 'Text', 'Current P(Black): 50.0%', 'FontColor', obj.COLOR_TEXT, 'FontSize', 12);
            obj.LblCurrentWealth = uilabel(metricsGrid, 'Text', 'Empirical Wealth S_t: $100.00', 'FontColor', obj.COLOR_CYAN, 'FontSize', 13, 'FontWeight', 'bold');
            obj.LblTheoreticalFinal = uilabel(metricsGrid, 'Text', 'Theoretical final S_N*: $9.08', 'FontColor', obj.COLOR_TEXT, 'FontSize', 12);
            
            % Badge Panel
            obj.BadgeEquivalence = uipanel(metricsGrid, ...
                'Title', '', ...
                'BackgroundColor', obj.COLOR_MUTED_GREEN, ...
                'BorderType', 'line', ...
                'HighlightColor', obj.COLOR_GREEN);
            obj.BadgeEquivalence.Layout.Row = 7;
            obj.BadgeEquivalence.Layout.Column = 1;
            
            badgeLabel = uilabel(obj.BadgeEquivalence, ...
                'Text', 'EQUIVALENCE VERIFIED', ...
                'FontName', 'Segoe UI', ...
                'FontSize', 12, ...
                'FontWeight', 'bold', ...
                'FontColor', obj.COLOR_GREEN, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'center');
            
            badgeGrid = uigridlayout(obj.BadgeEquivalence, [1, 1], ...
                'Padding', [0, 0, 0, 0]);
            badgeLabel.Parent = badgeGrid;
            
            % Configure mouse movement on the figure for card tooltips
            obj.FigureHandle.WindowButtonMotionFcn = @(src, event) obj.mouseMoved();
            
            % Bind all input callbacks
            obj.NumRedRSpinner.ValueChangedFcn = @(src, event) obj.resetSimulation();
            obj.NumBlackBSpinner.ValueChangedFcn = @(src, event) obj.resetSimulation();
            obj.OddsRedField.ValueChangedFcn = @(src, event) obj.resetSimulation();
            obj.OddsBlackField.ValueChangedFcn = @(src, event) obj.resetSimulation();
            obj.StartingWealthField.ValueChangedFcn = @(src, event) obj.resetSimulation();
            obj.StrategyDropdown.ValueChangedFcn = @(src, event) obj.strategyChanged();
            obj.BeliefDeviationSlider.ValueChangedFcn = @(src, event) obj.sliderChanged();
            
            % Initialize simulation parameters
            obj.resetSimulation();
        end
        
        function show(obj)
            obj.FigureHandle.Visible = 'on';
            drawnow;
            obj.resetSimulation();
        end
        
        function resetSimulation(obj)
            obj.pauseSimulation();
            
            R = obj.NumRedRSpinner.Value;
            B = obj.NumBlackBSpinner.Value;
            S_0 = obj.StartingWealthField.Value;
            
            % Import components
            import OceanMath.Information.Processes.CardDeckSimulator
            import OceanMath.Information.Processes.CardDeckBettingOptimizer
            import OceanViz.Strategies.CardDeckRenderer
            
            obj.Simulator = CardDeckSimulator(R, B);
            
            if isempty(obj.Renderer) || ~isvalid(obj.Renderer)
                obj.Renderer = CardDeckRenderer(obj.CardGridAx);
            end
            obj.Renderer.initializeLayout(R, B);
            
            obj.CurrentWealth = S_0;
            obj.WealthHistory = S_0;
            
            % Clear empirical plot path
            if ~isempty(obj.EmpiricalPathHandle) && isvalid(obj.EmpiricalPathHandle)
                delete(obj.EmpiricalPathHandle);
                obj.EmpiricalPathHandle = [];
            end
            
            % Clear 100-shuffle lines
            if ~isempty(obj.PathsLinesHandles)
                delete(obj.PathsLinesHandles);
                obj.PathsLinesHandles = [];
            end
            
            obj.Sim100MinY = [];
            obj.Sim100MaxY = [];
            
            % Redraw expectation
            if ~isempty(obj.TheoreticalPathHandle) && isvalid(obj.TheoreticalPathHandle)
                delete(obj.TheoreticalPathHandle);
                obj.TheoreticalPathHandle = [];
            end
            
            hold(obj.WealthAx, 'on');
            
            tTheoretical = [0, R + B];
            S_N_star = CardDeckBettingOptimizer.computeTheoreticalFinalWealth(R, B, obj.OddsRedField.Value, obj.OddsBlackField.Value, S_0);
            yTheoretical = [log2(S_0), log2(S_N_star)];
            
            obj.TheoreticalPathHandle = plot(obj.WealthAx, tTheoretical, yTheoretical, ...
                '--', 'Color', obj.COLOR_RED, 'LineWidth', 2, 'DisplayName', 'Theoretical Final');
            
            % Configure axis labels
            obj.WealthAx.Title.String = 'Log-Wealth Trajectory (log2(S_t) vs t)';
            obj.WealthAx.Title.Color = obj.COLOR_TEXT;
            obj.WealthAx.Title.FontName = 'Segoe UI';
            obj.WealthAx.Title.FontSize = 11;
            
            obj.WealthAx.XLabel.String = 'Card Draws (t)';
            obj.WealthAx.XLabel.Color = obj.COLOR_SUBTEXT;
            obj.WealthAx.YLabel.String = 'Log2 Wealth (log2(S_t))';
            obj.WealthAx.YLabel.Color = obj.COLOR_SUBTEXT;
            
            obj.WealthAx.XLim = [0, max(1, R + B)];
            obj.WealthAx.YLim = [log2(S_0) - 2, max(log2(S_0), log2(S_N_star)) + 2];
            obj.WealthAx.Color = obj.COLOR_BG;
            obj.WealthAx.GridColor = obj.COLOR_SUBTEXT;
            obj.WealthAx.GridAlpha = 0.2;
            grid(obj.WealthAx, 'on');
            
            obj.updatePlots();
            obj.updateMetrics();
            obj.updateEquivalenceBadge();
            drawnow;
        end
        
        function runSequentialStep(obj)
            N = obj.Simulator.NumRedR + obj.Simulator.NumBlackB;
            if obj.Simulator.CurrentIndex >= N
                obj.pauseSimulation();
                return;
            end
            
            % Bankruptcy check
            if obj.CurrentWealth <= 1e-4
                obj.pauseSimulation();
                uialert(obj.FigureHandle, 'Bankruptcy declared! S_t <= 1e-4. Please reset the simulation.', 'Bankruptcy Warning');
                return;
            end
            
            % Current stats
            r = obj.Simulator.RemainingRed;
            b = obj.Simulator.RemainingBlack;
            
            import OceanMath.Information.Processes.CardDeckBettingOptimizer
            [pRed, pBlack] = CardDeckBettingOptimizer.computeConditionalProbs(r, b);
            
            % Betting fractions based on active strategy
            strategy = obj.StrategyDropdown.Value;
            dev = obj.BeliefDeviationSlider.Value;
            
            if strcmp(strategy, 'Kelly Proportional (b = p)')
                b_Red = pRed;
                b_Black = pBlack;
            elseif strcmp(strategy, 'Constant Fraction (b = alpha)')
                b_Red = dev;
                b_Black = 1 - dev;
            else % Subjective Belief Mismatch
                pRed_subj = (1 - dev) * pRed + dev * 0.5;
                pBlack_subj = 1 - pRed_subj;
                b_Red = pRed_subj;
                b_Black = pBlack_subj;
            end
            
            obj.ActiveBeta = [b_Red, b_Black];
            
            % Draw
            [card, ~, ~] = obj.Simulator.drawNext();
            
            % Payoff
            oRed = obj.OddsRedField.Value;
            oBlack = obj.OddsBlackField.Value;
            payoff = CardDeckBettingOptimizer.getStepPayoff(card, r, b, oRed, oBlack, obj.ActiveBeta);
            
            obj.CurrentWealth = obj.CurrentWealth * payoff;
            obj.WealthHistory(end+1) = obj.CurrentWealth;
            
            % Update renderer
            obj.Renderer.addCard(card, obj.Simulator.CurrentIndex, pRed);
            obj.Renderer.updateUndrawnCount(N - obj.Simulator.CurrentIndex);
            
            % Replot
            obj.updatePlots();
            obj.updateMetrics();
            obj.updateEquivalenceBadge();
            
            % New bankruptcy check
            if obj.CurrentWealth <= 1e-4
                obj.pauseSimulation();
                uialert(obj.FigureHandle, 'Bankruptcy declared! S_t <= 1e-4. Please reset the simulation.', 'Bankruptcy Warning');
            end
            
            % Force UI update for smooth real-time animation
            drawnow;
        end
        
        function runSimulate100(obj)
            % Run 100 random shuffles and overlay
            obj.pauseSimulation();
            
            R = obj.NumRedRSpinner.Value;
            B = obj.NumBlackBSpinner.Value;
            N = R + B;
            S_0 = obj.StartingWealthField.Value;
            
            if N == 0, return; end
            
            numPaths = 100;
            permutations = zeros(numPaths, N);
            for p = 1:numPaths
                seq = [ones(1, R), 2 * ones(1, B)];
                permutations(p, :) = seq(randperm(N));
            end
            
            % Pre-allocated vectors for fast math
            r = R * ones(numPaths, 1);
            b = B * ones(numPaths, 1);
            S = S_0 * ones(numPaths, 1);
            
            wealthPaths = zeros(numPaths, N + 1);
            wealthPaths(:, 1) = S_0;
            
            strategy = obj.StrategyDropdown.Value;
            dev = obj.BeliefDeviationSlider.Value;
            oRed = obj.OddsRedField.Value;
            oBlack = obj.OddsBlackField.Value;
            
            import OceanMath.Information.Processes.CardDeckBettingOptimizer
            
            % Extremely fast vectorized loop over steps
            for t = 1:N
                total = r + b;
                
                % Compute probs
                pRed = r ./ total;
                pBlack = b ./ total;
                
                % Apply strategy
                if strcmp(strategy, 'Kelly Proportional (b = p)')
                    b_Red = pRed;
                    b_Black = pBlack;
                elseif strcmp(strategy, 'Constant Fraction (b = alpha)')
                    b_Red = dev * ones(numPaths, 1);
                    b_Black = (1 - dev) * ones(numPaths, 1);
                else
                    pRed_subj = (1 - dev) * pRed + dev * 0.5;
                    pBlack_subj = 1 - pRed_subj;
                    b_Red = pRed_subj;
                    b_Black = pBlack_subj;
                end
                
                cash = 1 - (b_Red + b_Black);
                drawn = permutations(:, t);
                
                payoff = zeros(numPaths, 1);
                red_idx = (drawn == 1);
                black_idx = (drawn == 2);
                
                payoff(red_idx) = oRed * b_Red(red_idx) + cash(red_idx);
                payoff(black_idx) = oBlack * b_Black(black_idx) + cash(black_idx);
                
                S = S .* payoff;
                wealthPaths(:, t + 1) = S;
                
                r(red_idx) = r(red_idx) - 1;
                b(black_idx) = b(black_idx) - 1;
            end
            
            % Delete previous paths
            if ~isempty(obj.PathsLinesHandles)
                delete(obj.PathsLinesHandles);
                obj.PathsLinesHandles = [];
            end
            
            % Prevent log2(0) = -Inf for bankrupt paths, flooring at 1e-4
            wealthPaths = max(1e-4, wealthPaths);
            
            % Render 100 lines at once in transparent grey-blue
            tData = 0:N;
            yData = log2(wealthPaths');
            
            hold(obj.WealthAx, 'on');
            obj.PathsLinesHandles = plot(obj.WealthAx, tData, yData, ...
                'Color', [0.35, 0.45, 0.55, 0.15], ...
                'LineWidth', 1.0, ...
                'HandleVisibility', 'off');
            
            % Adjust limits
            minY = min(yData(:));
            maxY = max(yData(:));
            
            % Robust visual clamping for bankrupt paths to prevent plot compression
            S_N_star = CardDeckBettingOptimizer.computeTheoreticalFinalWealth(R, B, oRed, oBlack, S_0);
            clampedMinY = min(log2(S_0), log2(S_N_star)) - 3;
            minY = max(clampedMinY, minY);
            
            obj.Sim100MinY = minY;
            obj.Sim100MaxY = maxY;
            
            if minY == maxY
                obj.WealthAx.YLim = [minY - 1, maxY + 1];
            else
                obj.WealthAx.YLim = [minY - 0.5, maxY + 0.5];
            end
            
            % Bring empirical path and expectation to front
            if ~isempty(obj.TheoreticalPathHandle) && isvalid(obj.TheoreticalPathHandle)
                uistack(obj.TheoreticalPathHandle, 'top');
            end
            if ~isempty(obj.EmpiricalPathHandle) && isvalid(obj.EmpiricalPathHandle)
                uistack(obj.EmpiricalPathHandle, 'top');
            end
            
            % Force UI refresh to show the overlay instantly
            drawnow;
        end
        
        function updatePlots(obj)
            t = 0:(length(obj.WealthHistory) - 1);
            yData = log2(max(1e-4, obj.WealthHistory));
            
            if isempty(obj.EmpiricalPathHandle) || ~isvalid(obj.EmpiricalPathHandle)
                obj.EmpiricalPathHandle = plot(obj.WealthAx, t, yData, ...
                    'Color', obj.COLOR_CYAN, 'LineWidth', 3, 'DisplayName', 'Empirical Run');
            else
                obj.EmpiricalPathHandle.XData = t;
                obj.EmpiricalPathHandle.YData = yData;
            end
            
            % Dynamically adjust Y limits efficiently
            minY = min(yData);
            maxY = max(yData);
            if ~isempty(obj.Sim100MinY)
                minY = min(minY, obj.Sim100MinY);
                maxY = max(maxY, obj.Sim100MaxY);
            end
            
            % Robust visual clamping for bankrupt paths to prevent plot compression
            R = obj.NumRedRSpinner.Value;
            B = obj.NumBlackBSpinner.Value;
            S_0 = obj.StartingWealthField.Value;
            oRed = obj.OddsRedField.Value;
            oBlack = obj.OddsBlackField.Value;
            import OceanMath.Information.Processes.CardDeckBettingOptimizer
            S_N_star = CardDeckBettingOptimizer.computeTheoreticalFinalWealth(R, B, oRed, oBlack, S_0);
            clampedMinY = min(log2(S_0), log2(S_N_star)) - 3;
            minY = max(clampedMinY, minY);
            
            if minY == maxY
                obj.WealthAx.YLim = [minY - 1, maxY + 1];
            else
                obj.WealthAx.YLim = [minY - 0.5, maxY + 0.5];
            end
            
            if ~isempty(obj.TheoreticalPathHandle) && isvalid(obj.TheoreticalPathHandle)
                uistack(obj.TheoreticalPathHandle, 'top');
            end
            if ~isempty(obj.EmpiricalPathHandle) && isvalid(obj.EmpiricalPathHandle)
                uistack(obj.EmpiricalPathHandle, 'top');
            end
        end
        
        function updateMetrics(obj)
            R = obj.NumRedRSpinner.Value;
            B = obj.NumBlackBSpinner.Value;
            N = R + B;
            
            drawnCount = obj.Simulator.CurrentIndex;
            remainingCount = N - drawnCount;
            
            obj.LblCardsDrawn.Text = sprintf('Cards Drawn: %d / %d', drawnCount, N);
            obj.LblRemaining.Text = sprintf('Remaining Cards: %d', remainingCount);
            
            r = obj.Simulator.RemainingRed;
            b = obj.Simulator.RemainingBlack;
            
            import OceanMath.Information.Processes.CardDeckBettingOptimizer
            [pRed, pBlack] = CardDeckBettingOptimizer.computeConditionalProbs(r, b);
            
            obj.LblCurrentPRed.Text = sprintf('Current P(Red): %.1f%%', pRed * 100);
            obj.LblCurrentPBlack.Text = sprintf('Current P(Black): %.1f%%', pBlack * 100);
            
            obj.LblCurrentWealth.Text = sprintf('Empirical Wealth S_t: $%.2f', obj.CurrentWealth);
            
            oRed = obj.OddsRedField.Value;
            oBlack = obj.OddsBlackField.Value;
            S_0 = obj.StartingWealthField.Value;
            
            S_N_star = CardDeckBettingOptimizer.computeTheoreticalFinalWealth(R, B, oRed, oBlack, S_0);
            obj.LblTheoreticalFinal.Text = sprintf('Theoretical final S_N*: $%.2f', S_N_star);
        end
        
        function updateEquivalenceBadge(obj)
            strategy = obj.StrategyDropdown.Value;
            dev = obj.BeliefDeviationSlider.Value;
            
            badgeLbl = findobj(obj.BadgeEquivalence, 'Type', 'uilabel');
            
            if strcmp(strategy, 'Kelly Proportional (b = p)') && dev == 0
                obj.BadgeEquivalence.BackgroundColor = obj.COLOR_MUTED_GREEN;
                obj.BadgeEquivalence.HighlightColor = obj.COLOR_GREEN;
                if ~isempty(badgeLbl)
                    badgeLbl.FontColor = obj.COLOR_GREEN;
                    badgeLbl.Text = 'EQUIVALENCE VERIFIED';
                end
            else
                obj.BadgeEquivalence.BackgroundColor = [0.25, 0.18, 0.12];
                obj.BadgeEquivalence.HighlightColor = obj.COLOR_ORANGE;
                if ~isempty(badgeLbl)
                    badgeLbl.FontColor = obj.COLOR_ORANGE;
                    badgeLbl.Text = 'MISMATCH DETECTED';
                end
            end
        end
        
        function strategyChanged(obj)
            strategy = obj.StrategyDropdown.Value;
            if strcmp(strategy, 'Kelly Proportional (b = p)')
                obj.BeliefDeviationSlider.Enable = 'off';
                obj.BeliefDeviationSlider.Value = 0;
                obj.BeliefDeviationLabel.Text = '0% Deviation';
            elseif strcmp(strategy, 'Constant Fraction (b = alpha)')
                obj.BeliefDeviationSlider.Enable = 'on';
                obj.BeliefDeviationSlider.Value = 0.5;
                obj.BeliefDeviationLabel.Text = 'alpha = 0.50';
            else % Subjective Belief Mismatch
                obj.BeliefDeviationSlider.Enable = 'on';
                obj.BeliefDeviationSlider.Value = 0.2;
                obj.BeliefDeviationLabel.Text = '20% Deviation';
            end
            
            obj.updateEquivalenceBadge();
            obj.resetSimulation();
        end
        
        function sliderChanged(obj)
            strategy = obj.StrategyDropdown.Value;
            val = obj.BeliefDeviationSlider.Value;
            if strcmp(strategy, 'Constant Fraction (b = alpha)')
                obj.BeliefDeviationLabel.Text = sprintf('alpha = %.2f', val);
            else
                obj.BeliefDeviationLabel.Text = sprintf('%.0f%% Deviation', val * 100);
            end
            
            obj.updateEquivalenceBadge();
            obj.resetSimulation();
        end
        
        function stepPressed(obj)
            obj.pauseSimulation();
            obj.runSequentialStep();
        end
        
        function togglePlay(obj)
            if obj.IsPlaying
                obj.pauseSimulation();
            else
                obj.startSimulation();
            end
        end
        
        function startSimulation(obj)
            N = obj.Simulator.NumRedR + obj.Simulator.NumBlackB;
            if obj.Simulator.CurrentIndex >= N
                return;
            end
            
            obj.IsPlaying = true;
            obj.BtnPlayPause.Text = 'Pause';
            obj.BtnPlayPause.BackgroundColor = [0.80, 0.30, 0.30];
            
            obj.cleanupTimer();
            
            obj.TimerPlayback = timer(...
                'ExecutionMode', 'fixedRate', ...
                'Period', 0.2, ...
                'TimerFcn', @(src, event) obj.safeTimerStep(), ...
                'Name', 'CardDeckPlaybackTimer');
            start(obj.TimerPlayback);
        end
        
        function pauseSimulation(obj)
            obj.IsPlaying = false;
            obj.BtnPlayPause.Text = 'Play';
            obj.BtnPlayPause.BackgroundColor = [0.15, 0.50, 0.80];
            obj.cleanupTimer();
        end
        
        function safeTimerStep(obj)
            if ~isvalid(obj) || ~isvalid(obj.FigureHandle)
                obj.cleanupTimer();
                return;
            end
            try
                obj.runSequentialStep();
            catch ME
                obj.pauseSimulation();
                uialert(obj.FigureHandle, ME.message, 'Simulation Error');
            end
        end
        
        function mouseMoved(obj)
            if isempty(obj.CardGridAx) || ~isvalid(obj.CardGridAx) || isempty(obj.Renderer) || ~isvalid(obj.Renderer)
                return;
            end
            
            % Query mouse location inside CardGridAx
            cp = obj.CardGridAx.CurrentPoint;
            xCursor = cp(1, 1);
            yCursor = cp(1, 2);
            
            % Sync current odds to renderer
            obj.Renderer.OddsRed = obj.OddsRedField.Value;
            obj.Renderer.OddsBlack = obj.OddsBlackField.Value;
            
            % Delegate hover check and tooltip rendering entirely to the renderer
            obj.Renderer.checkHover(xCursor, yCursor);
        end
        
        function cleanupTimer(obj)
            if ~isempty(obj.TimerPlayback) && isvalid(obj.TimerPlayback)
                if strcmp(obj.TimerPlayback.Running, 'on')
                    stop(obj.TimerPlayback);
                end
                delete(obj.TimerPlayback);
                obj.TimerPlayback = [];
            end
        end
        
        function closeApp(obj)
            obj.cleanupTimer();
            delete(obj.FigureHandle);
        end
    end
end
