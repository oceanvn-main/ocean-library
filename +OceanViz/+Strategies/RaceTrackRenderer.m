classdef RaceTrackRenderer < handle
    % RACETRACKRENDERER  Premium high-fidelity horse race visualizer.
    %
    % Renders horizontal tracks with glowing neon borders (cyberpunk dark mode),
    % stylized horse markers, leader spotlight, and trailing sparks.
    %
    % Design tokens are dynamically integrated with OceanViz.Core Themes.
    % Bypasses native LaTeX to ensure cross-platform compatibility.
    
    properties
        AxesHandle
        NumHorses
        CoreObj
        
        % Graphics handles for blitting / fast updates
        LaneBackgrounds   % Array of patch handles for lane backgrounds
        LaneDividers      % Array of line handles for division rails
        CenterLines       % Array of line handles for tracks
        StartGateLine     % Line handle for start gate
        FinishLineGrid    % Array of line/patch handles for finish line
        
        HorseMarkers      % Array of line/scatter handles for horse dots
        HorseTexts        % Array of text handles for horse labels
        
        LeaderHighlightRing % Circle handle for leader spotlight
        LeaderSparks        % Cell array of line handles for trailing sparks
        
        WinnerText        % Text handle for winning banner
    end
    
    methods
        function obj = RaceTrackRenderer(axesHandle, numHorses, coreObj)
            arguments
                axesHandle
                numHorses (1,1) double {mustBeInteger, mustBePositive}
                coreObj = []
            end
            
            obj.AxesHandle = axesHandle;
            obj.NumHorses = numHorses;
            
            if isempty(coreObj)
                try
                    obj.CoreObj = OceanViz.Core('Theme', 'Dark');
                catch
                    % Fallback to standard Neon/Dark theme if Core is unavailable
                    obj.CoreObj = struct();
                    obj.CoreObj.ThemeName = "Dark";
                    obj.CoreObj.Colors = struct(...
                        'Background', [0.08, 0.08, 0.10], ...
                        'Text', [0.95, 0.95, 0.95], ...
                        'Grid', [0.20, 0.22, 0.28], ...
                        'Primary', [0.0, 0.8, 1.0], ...
                        'Secondary', [1.0, 0.3, 0.5], ...
                        'Boundary', [1.0, 0.8, 0.2] ...
                    );
                    obj.CoreObj.getColorSequence = @(n) obj.fallbackColors(n);
                end
            else
                obj.CoreObj = coreObj;
            end
            
            % Setup basic axes properties
            obj.styleAxes();
        end
        
        function colors = fallbackColors(~, n)
            % A beautiful neon custom color map fallback
            baseColors = [
                0.0, 1.0, 1.0;   % Cyan
                1.0, 0.4, 0.7;   % Hot Pink
                0.0, 1.0, 0.5;   % Lime Green
                1.0, 0.8, 0.0;   % Yellow/Gold
                0.6, 0.4, 1.0;   % Purple
                1.0, 0.5, 0.0;   % Orange
                0.2, 0.6, 1.0;   % Ocean Blue
                0.9, 0.9, 0.9    % White/Gray
            ];
            if n <= size(baseColors, 1)
                colors = baseColors(1:n, :);
            else
                colors = jet(n);
            end
        end
        
        function styleAxes(obj)
            ax = obj.AxesHandle;
            bg = obj.CoreObj.Colors.Background;
            txt = obj.CoreObj.Colors.Text;
            gridCol = obj.CoreObj.Colors.Grid;
            
            set(ax, 'Color', bg, ...
                    'XColor', txt, ...
                    'YColor', txt, ...
                    'GridColor', gridCol, ...
                    'GridAlpha', 0.15, ...
                    'LineWidth', 1.2, ...
                    'FontName', 'Helvetica', ...
                    'TickLabelInterpreter', 'none', ...
                    'Box', 'on');
            grid(ax, 'off');
        end
        
        function initializeTrack(obj)
            % Draws the static track, lanes, start gate, and finish line.
            ax = obj.AxesHandle;
            hold(ax, 'on');
            
            % Clear any previous elements in this axes
            cla(ax);
            
            % Reset winner text handle
            obj.WinnerText = [];
            
            bg = obj.CoreObj.Colors.Background;
            gridCol = obj.CoreObj.Colors.Grid;
            horseColors = obj.CoreObj.getColorSequence(obj.NumHorses);
            
            K = obj.NumHorses;
            
            % 1. Draw lane backgrounds
            obj.LaneBackgrounds = gobjects(1, K);
            for k = 1:K
                y_bottom = k - 0.42;
                y_top = k + 0.42;
                
                % Subtle glow background for each lane
                obj.LaneBackgrounds(k) = patch(ax, [-0.08, 1.08, 1.08, -0.08], ...
                    [y_bottom, y_bottom, y_top, y_top], ...
                    bg * 1.3, 'EdgeColor', 'none', 'FaceAlpha', 0.25);
            end
            
            % 2. Draw lane dividers (rails)
            obj.LaneDividers = gobjects(1, K + 1);
            for k = 0:K
                y_rail = k + 0.5;
                obj.LaneDividers(k+1) = line(ax, [-0.08, 1.08], [y_rail, y_rail], ...
                    'Color', gridCol, 'LineWidth', 1.5, 'LineStyle', '-');
            end
            
            % 3. Draw center track guides
            obj.CenterLines = gobjects(1, K);
            for k = 1:K
                obj.CenterLines(k) = line(ax, [0, 1], [k, k], ...
                    'Color', horseColors(k, :), 'LineWidth', 0.8, 'LineStyle', ':');
            end
            
            % 4. Draw Start Gate
            obj.StartGateLine = line(ax, [0, 0], [0.5, K + 0.5], ...
                'Color', [1.0, 0.4, 0.4], 'LineWidth', 2.0, 'LineStyle', '--');
            text(ax, 0, K + 0.58, 'START', 'Color', [1.0, 0.4, 0.4], ...
                'FontSize', 9, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', 'Interpreter', 'none');
            
            % 5. Draw Finish Line
            obj.FinishLineGrid = line(ax, [1, 1], [0.5, K + 0.5], ...
                'Color', [0.0, 1.0, 0.5], 'LineWidth', 3.0, 'LineStyle', '-');
            text(ax, 1, K + 0.58, 'FINISH', 'Color', [0.0, 1.0, 0.5], ...
                'FontSize', 9, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', 'Interpreter', 'none');
            
            % 6. Draw leader spotlight ring (invisible initially)
            obj.LeaderHighlightRing = plot(ax, 0, 1, 'o', ...
                'MarkerSize', 24, 'MarkerFaceColor', 'none', ...
                'MarkerEdgeColor', [1.0, 0.85, 0.0], 'LineWidth', 2.0, ...
                'LineStyle', '--', 'Visible', 'off');
            
            % 7. Draw leader sparks (invisible initially)
            numSparks = 4;
            obj.LeaderSparks = cell(1, numSparks);
            sparkColors = [
                1.0, 0.85, 0.0;   % Bright gold
                1.0, 0.60, 0.0;   % Orange
                0.8, 0.40, 0.0;   % Dark orange
                0.5, 0.20, 0.0    % Faded red-brown
            ];
            for s = 1:numSparks
                obj.LeaderSparks{s} = plot(ax, 0, 1, 'o', ...
                    'MarkerSize', 11 - 2*s, 'MarkerFaceColor', sparkColors(s, :), ...
                    'MarkerEdgeColor', 'none', 'Visible', 'off');
            end
            
            % 8. Draw Horse Markers and Labels
            obj.HorseMarkers = gobjects(1, K);
            obj.HorseTexts = gobjects(1, K);
            for k = 1:K
                obj.HorseMarkers(k) = plot(ax, 0, k, 'o', ...
                    'MarkerSize', 16, 'MarkerFaceColor', horseColors(k, :), ...
                    'MarkerEdgeColor', [1.0, 1.0, 1.0], 'LineWidth', 1.8);
                obj.HorseTexts(k) = text(ax, 0, k, sprintf('H%d', k), ...
                    'Color', [1.0, 1.0, 1.0], 'FontSize', 8, 'FontWeight', 'bold', ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                    'Interpreter', 'none');
            end
            
            % Configure limits and ticks
            set(ax, 'XLim', [-0.08, 1.08], 'YLim', [0.3, K + 0.7]);
            set(ax, 'YTick', 1:K);
            
            % Convert numeric y-ticks to custom lane labels, avoiding raw LaTeX
            tickLabels = cell(1, K);
            for k = 1:K
                tickLabels{k} = sprintf('Lane %d', k);
            end
            set(ax, 'YTickLabel', tickLabels);
            
            hold(ax, 'off');
        end
        
        function updatePositions(obj, positions)
            % Accepts a 1xK vector of values between 0 (start) and 1 (finish).
            % Smoothly shifts the horse shapes to their new coordinates.
            arguments
                obj
                positions (1,:) double {mustBeNonnegative}
            end
            
            K = obj.NumHorses;
            if numel(positions) ~= K
                error('Positions must be 1x%d vector', K);
            end
            
            % Update horse marker positions
            for k = 1:K
                xVal = positions(k);
                set(obj.HorseMarkers(k), 'XData', xVal, 'YData', k);
                set(obj.HorseTexts(k), 'Position', [xVal, k, 0]);
            end
            
            % Spotlight leading horse
            [maxPos, leaderIdx] = max(positions);
            if maxPos > 0
                % Highlight ring
                set(obj.LeaderHighlightRing, 'XData', maxPos, 'YData', leaderIdx, 'Visible', 'on');
                
                % Sparks trailing behind
                numSparks = numel(obj.LeaderSparks);
                for s = 1:numSparks
                    sparkX = maxPos - s * 0.03;
                    if sparkX >= 0
                        set(obj.LeaderSparks{s}, 'XData', sparkX, 'YData', leaderIdx, 'Visible', 'on');
                    else
                        set(obj.LeaderSparks{s}, 'Visible', 'off');
                    end
                end
            else
                set(obj.LeaderHighlightRing, 'Visible', 'off');
                for s = 1:numel(obj.LeaderSparks)
                    set(obj.LeaderSparks{s}, 'Visible', 'off');
                end
            end
            
            % Hide winner banner if the race is just starting/resetting
            if maxPos == 0 && ~isempty(obj.WinnerText) && ishandle(obj.WinnerText)
                set(obj.WinnerText, 'Visible', 'off');
            end
            
            drawnow limitrate;
        end
        
        function triggerWinnerFlash(obj, winnerIndex)
            % Triggers a micro-animation flashing the winning horse's lane green
            % and rendering a "Horse X Wins!" text banner.
            arguments
                obj
                winnerIndex (1,1) double {mustBeInteger, mustBePositive}
            end
            
            ax = obj.AxesHandle;
            hold(ax, 'on');
            
            % 1. Render/Configure Winner Banner
            bgCol = obj.CoreObj.Colors.Background;
            if isempty(obj.WinnerText) || ~ishandle(obj.WinnerText)
                obj.WinnerText = text(ax, 0.5, obj.NumHorses / 2 + 0.5, ...
                    sprintf('Horse %d Wins!', winnerIndex), ...
                    'Color', [0.0, 1.0, 0.5], ... % Neon green
                    'FontSize', 22, ...
                    'FontWeight', 'bold', ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', ...
                    'BackgroundColor', [bgCol * 0.4, 0.85], ...
                    'EdgeColor', [0.0, 1.0, 0.5], ...
                    'LineWidth', 2.0, ...
                    'Margin', 12, ...
                    'Interpreter', 'none', ...
                    'Visible', 'on');
            else
                set(obj.WinnerText, 'String', sprintf('Horse %d Wins!', winnerIndex), ...
                    'Color', [0.0, 1.0, 0.5], ...
                    'EdgeColor', [0.0, 1.0, 0.5], ...
                    'Visible', 'on');
            end
            
            % 2. Flash winning lane background green
            laneBg = obj.LaneBackgrounds(winnerIndex);
            originalColor = get(laneBg, 'FaceColor');
            flashColor = [0.0, 0.45, 0.15]; % Glowing green
            
            for i = 1:6
                if mod(i, 2) == 1
                    set(laneBg, 'FaceColor', flashColor);
                    set(obj.WinnerText, 'Visible', 'on');
                else
                    set(laneBg, 'FaceColor', originalColor);
                    set(obj.WinnerText, 'Visible', 'off');
                end
                drawnow;
                pause(0.12);
            end
            
            % Set to final winning highlight state
            set(laneBg, 'FaceColor', flashColor * 0.6 + originalColor * 0.4);
            set(obj.WinnerText, 'Visible', 'on');
            drawnow;
            hold(ax, 'off');
        end
        
        function clearWinnerFlash(obj)
            % Hides the winner text banner and resets all lane backgrounds
            % to their default subtle theme color.
            if ~isempty(obj.WinnerText) && ishandle(obj.WinnerText)
                set(obj.WinnerText, 'Visible', 'off');
            end
            
            if ~isempty(obj.LaneBackgrounds)
                bg = obj.CoreObj.Colors.Background;
                for k = 1:obj.NumHorses
                    if ishandle(obj.LaneBackgrounds(k))
                        set(obj.LaneBackgrounds(k), 'FaceColor', bg * 1.3);
                    end
                end
            end
            drawnow;
        end
    end
end
