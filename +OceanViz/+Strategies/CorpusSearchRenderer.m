classdef CorpusSearchRenderer < handle
    % CORPUSSEARCHRENDERER Renders a simulated page of text and animates the Monte Carlo search cursor.
    %
    % This class manages the visual grid representing Shannon's physical
    % "book-flip" search. It slices local sub-windows of the corpus around
    % the search area, displays them in a clean grid, highlights the active
    % scanning path in glowing amber, and flashes the chosen next character
    % in neon cyan.
    %
    % See also: OceanMath.Information.Processes.MonteCarloSearchEngine

    properties
        AxesHandle (1, 1) matlab.ui.control.UIAxes
        TextHandles = gobjects(0)
        HighlightPatches
        Rows (1, 1) double = 0
        Cols (1, 1) double = 0
    end

    methods
        function obj = CorpusSearchRenderer(ax)
            % CORPUSSEARCHRENDERER Constructor.
            arguments
                ax matlab.ui.control.UIAxes
            end
            obj.AxesHandle = ax;

            % Configure dark mode axes
            obj.AxesHandle.BackgroundColor = [0.10, 0.12, 0.16];
            obj.AxesHandle.XColor = 'none';
            obj.AxesHandle.YColor = 'none';
            obj.AxesHandle.XTick = [];
            obj.AxesHandle.YTick = [];
            obj.AxesHandle.XGrid = 'off';
            obj.AxesHandle.YGrid = 'off';
            obj.AxesHandle.Box = 'on';
            obj.AxesHandle.Title.Interpreter = 'none';
            
            % Initialize default 10x25 grid
            obj.initializeGrid(10, 25);
        end

        function initializeGrid(obj, rows, cols)
            % INITIALIZEGRID Pre-allocates text objects and patches in the axes.
            arguments
                obj
                rows (1, 1) double {mustBeInteger, mustBePositive}
                cols (1, 1) double {mustBeInteger, mustBePositive}
            end

            obj.Rows = rows;
            obj.Cols = cols;

            % Clear any existing children in axes
            cla(obj.AxesHandle);

            % Set limits
            obj.AxesHandle.XLim = [0.5, cols + 0.5];
            obj.AxesHandle.YLim = [0.5, rows + 0.5];

            % Hold axes for layers
            hold(obj.AxesHandle, 'on');

            % Create struct for highlight patches
            obj.HighlightPatches = struct();

            % 1. Create Cursor (active search cursor)
            obj.HighlightPatches.Cursor = rectangle(obj.AxesHandle, ...
                'Position', [0, 0, 0, 0], ...
                'FaceColor', [0.99, 0.40, 0.10], ...
                'EdgeColor', 'none', ...
                'Curvature', 0.2, ...
                'Visible', 'off');

            % 2. Create Drawn (sampled character)
            obj.HighlightPatches.Drawn = rectangle(obj.AxesHandle, ...
                'Position', [0, 0, 0, 0], ...
                'FaceColor', [0.30, 0.65, 0.95], ...
                'EdgeColor', 'none', ...
                'Curvature', 0.2, ...
                'Visible', 'off');

            % 3. Create Match (highlight sequence)
            obj.HighlightPatches.Match = rectangle(obj.AxesHandle, ...
                'Position', [0, 0, 0, 0], ...
                'FaceColor', 'none', ...
                'EdgeColor', [0.99, 0.80, 0.20], ...
                'LineWidth', 1.5, ...
                'Curvature', 0.2, ...
                'Visible', 'off');

            % 4. Create text handles grid
            obj.TextHandles = gobjects(rows, cols);
            for r = 1:rows
                for c = 1:cols
                    y = rows - r + 1;
                    x = c;
                    obj.TextHandles(r, c) = text(obj.AxesHandle, x, y, ' ', ...
                        'Color', [0.45, 0.48, 0.53], ...
                        'FontName', 'Courier New', ...
                        'FontSize', 12, ...
                        'FontWeight', 'normal', ...
                        'HorizontalAlignment', 'center', ...
                        'VerticalAlignment', 'middle', ...
                        'Interpreter', 'none');
                end
            end

            hold(obj.AxesHandle, 'off');

            % Put text handles on top layer so they are drawn above patches
            uistack(obj.TextHandles(:), 'top');
        end

        function renderSearchStep(obj, displayText, scanPos, matchPos, contextLen, drawnCharPos)
            % RENDERSEARCHSTEP Renders a static snapshot of the search grid state.
            arguments
                obj
                displayText char
                scanPos
                matchPos
                contextLen
                drawnCharPos
            end

            [rows, cols] = size(obj.TextHandles);
            W = rows * cols;

            % Ensure length of displayText matches grid size
            if length(displayText) < W
                displayText = [displayText, repmat(' ', 1, W - length(displayText))];
            end

            % 1. Reset all text to default styling
            for r = 1:rows
                for c = 1:cols
                    p = (r - 1) * cols + c;
                    set(obj.TextHandles(r, c), ...
                        'String', displayText(p), ...
                        'Color', [0.45, 0.48, 0.53], ...
                        'FontWeight', 'normal');
                end
            end

            % 2. Render Active Scan Pointer
            if ~isempty(scanPos) && scanPos >= 1 && scanPos <= W
                r_scan = ceil(scanPos / cols);
                c_scan = scanPos - (r_scan - 1) * cols;
                y_scan = rows - r_scan + 1;
                set(obj.HighlightPatches.Cursor, ...
                    'Position', [c_scan - 0.45, y_scan - 0.45, 0.9, 0.9], ...
                    'Visible', 'on');
                set(obj.TextHandles(r_scan, c_scan), ...
                    'Color', [1.0, 1.0, 1.0], ...
                    'FontWeight', 'bold');
            else
                set(obj.HighlightPatches.Cursor, 'Visible', 'off');
            end

            % 3. Render Match Highlights
            if ~isempty(matchPos) && matchPos >= 1 && matchPos <= W && contextLen > 0
                r_m = ceil(matchPos / cols);
                c_m = matchPos - (r_m - 1) * cols;
                y_m = rows - r_m + 1;
                
                % Single rectangle covering the match sequence width
                widthVal = min(contextLen, cols - c_m + 1) - 0.04;
                if widthVal < 0.96
                    widthVal = 0.96;
                end
                
                set(obj.HighlightPatches.Match, ...
                    'Position', [c_m - 0.48, y_m - 0.48, widthVal, 0.96], ...
                    'Visible', 'on');
                
                % Also highlight each text character in the match sequence
                for i = 1:contextLen
                    p_m = matchPos + i - 1;
                    if p_m <= W
                        curr_r = ceil(p_m / cols);
                        curr_c = p_m - (curr_r - 1) * cols;
                        set(obj.TextHandles(curr_r, curr_c), ...
                            'Color', [0.99, 0.80, 0.20], ...
                            'FontWeight', 'bold');
                    end
                end
            else
                set(obj.HighlightPatches.Match, 'Visible', 'off');
            end

            % 4. Render Sampled Letter Draw
            if ~isempty(drawnCharPos) && drawnCharPos >= 1 && drawnCharPos <= W
                r_draw = ceil(drawnCharPos / cols);
                c_draw = drawnCharPos - (r_draw - 1) * cols;
                y_draw = rows - r_draw + 1;
                set(obj.HighlightPatches.Drawn, ...
                    'Position', [c_draw - 0.45, y_draw - 0.45, 0.9, 0.9], ...
                    'Visible', 'on');
                set(obj.TextHandles(r_draw, c_draw), ...
                    'Color', [1.0, 1.0, 1.0], ...
                    'FontWeight', 'bold');
            else
                set(obj.HighlightPatches.Drawn, 'Visible', 'off');
            end
        end

        function animateScan(obj, corpus, searchLog, speedMs, contextLen)
            % ANIMATESCAN Runs step-by-step linear scan search animations.
            arguments
                obj
                corpus char
                searchLog struct
                speedMs (1, 1) double
                contextLen (1, 1) double = 1
            end

            [rows, cols] = size(obj.TextHandles);
            W = rows * cols;
            N = length(corpus);

            if N == 0
                return;
            end

            % Start local window at searchLog.startIndex
            startIndex = searchLog.startIndex;
            displayIndices = mod((startIndex : startIndex + W - 1) - 1, N) + 1;
            displayText = corpus(displayIndices);

            % Update grid immediately with initial text segment
            for p = 1:W
                r = ceil(p / cols);
                c = p - (r - 1) * cols;
                set(obj.TextHandles(r, c), 'String', displayText(p));
            end

            % Calculate pause duration in seconds
            dt = speedMs / 1000.0;
            
            scannedCount = length(searchLog.scannedIndices);
            
            % Cap frame animations to preserve performance
            maxFrames = 30;
            stepSize = 1;
            if scannedCount > maxFrames
                stepSize = ceil(scannedCount / maxFrames);
            end

            % Animate scan progression
            for idx = 1:stepSize:scannedCount
                currCorpusIdx = searchLog.scannedIndices(idx);
                scanPos = find(displayIndices == currCorpusIdx, 1, 'first');
                if ~isempty(scanPos)
                    obj.renderSearchStep(displayText, scanPos, [], 0, []);
                    drawnow;
                    if dt > 0
                        pause(dt);
                    end
                end
            end

            % Highlight final match and flash sampled letter
            matchPos = find(displayIndices == searchLog.matchIndex, 1, 'first');
            if ~isempty(matchPos)
                drawnCharCorpusIdx = mod(searchLog.matchIndex + contextLen - 1, N) + 1;
                drawnCharPos = find(displayIndices == drawnCharCorpusIdx, 1, 'first');

                obj.renderSearchStep(displayText, [], matchPos, contextLen, drawnCharPos);
                drawnow;

                % Flashing loop to draw visual attention to chosen character
                if dt > 0
                    for flash = 1:3
                        set(obj.HighlightPatches.Drawn, 'Visible', 'off');
                        drawnow;
                        pause(0.08);
                        set(obj.HighlightPatches.Drawn, 'Visible', 'on');
                        drawnow;
                        pause(0.08);
                    end
                end
            end
        end
    end
end
