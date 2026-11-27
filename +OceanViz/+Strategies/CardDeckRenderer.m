classdef CardDeckRenderer < handle
    % CARDDECKRENDERER Renders the card drawing grid, drawn sequence, and remaining deck pile on a UIAxes.
    %
    % Compliant with plain Unicode/ASCII requirements (No LaTeX).
    %
    % See also: OceanMath.Information.Processes.CardDeckSimulator

    properties
        AxesHandle (1, 1) matlab.ui.control.UIAxes
        CardPatches % struct containing Handles (rectangles), Labels (cell array), Positions (Nx4), Types (double array)
        ProbValues (1, :) double = []
        DeckPilePatches = gobjects(0)
        DeckCountLabel = gobjects(0)
        TooltipRect = gobjects(0)
        TooltipText = gobjects(0)
        R (1, 1) double = 0
        B (1, 1) double = 0
    end

    methods
        function obj = CardDeckRenderer(ax)
            % CARDDECKRENDERER Constructor.
            arguments
                ax matlab.ui.control.UIAxes
            end
            obj.AxesHandle = ax;

            % Configure dark mode axes properties
            obj.AxesHandle.BackgroundColor = [0.10, 0.12, 0.16];
            obj.AxesHandle.XColor = 'none';
            obj.AxesHandle.YColor = 'none';
            obj.AxesHandle.XTick = [];
            obj.AxesHandle.YTick = [];
            obj.AxesHandle.XGrid = 'off';
            obj.AxesHandle.YGrid = 'off';
            obj.AxesHandle.Box = 'on';
            obj.AxesHandle.XLim = [0, 10];
            obj.AxesHandle.YLim = [0, 10];
            obj.AxesHandle.Title.Interpreter = 'none';
            obj.AxesHandle.TickLabelInterpreter = 'none';

            % Initialize properties
            obj.CardPatches = struct();
            obj.CardPatches.Handles = gobjects(0);
            obj.CardPatches.Labels = {};
            obj.CardPatches.Positions = [];
            obj.CardPatches.Types = [];
            obj.ProbValues = [];
        end

        function initializeLayout(obj, R, B)
            % INITIALIZELAYOUT Clears the axes, computes coordinates for the pile, and draws slot frameworks.
            arguments
                obj
                R (1, 1) double {mustBeNonnegative, mustBeInteger}
                B (1, 1) double {mustBeNonnegative, mustBeInteger}
            end

            obj.R = R;
            obj.B = B;

            % 1. Clear everything
            obj.clear();
            cla(obj.AxesHandle);

            % 2. Re-configure Axes limits & styling
            obj.AxesHandle.BackgroundColor = [0.10, 0.12, 0.16];
            obj.AxesHandle.XColor = 'none';
            obj.AxesHandle.YColor = 'none';
            obj.AxesHandle.XTick = [];
            obj.AxesHandle.YTick = [];
            obj.AxesHandle.XGrid = 'off';
            obj.AxesHandle.YGrid = 'off';
            obj.AxesHandle.Box = 'on';
            obj.AxesHandle.XLim = [0, 10];
            obj.AxesHandle.YLim = [0, 10];
            obj.AxesHandle.Title.Interpreter = 'none';
            obj.AxesHandle.TickLabelInterpreter = 'none';

            % 3. Pre-allocate graphics vectors
            N_total = R + B;
            obj.CardPatches = struct();
            obj.CardPatches.Handles = gobjects(1, N_total);
            obj.CardPatches.Labels = cell(1, N_total);
            obj.CardPatches.Positions = zeros(N_total, 4);
            obj.CardPatches.Types = zeros(1, N_total);
            obj.ProbValues = zeros(1, N_total);

            % Re-create the tooltip elements
            obj.TooltipRect = rectangle(obj.AxesHandle, ...
                'Position', [0, 0, 0, 0], ...
                'FaceColor', [0.17, 0.19, 0.23], ...
                'EdgeColor', [0.30, 0.65, 0.95], ...
                'LineWidth', 1.2, ...
                'Curvature', 0.1, ...
                'Visible', 'off');
            obj.TooltipText = text(obj.AxesHandle, 0, 0, '', ...
                'Color', [0.85, 0.88, 0.93], ...
                'FontName', 'Courier New', ...
                'FontSize', 9, ...
                'Interpreter', 'none', ...
                'Visible', 'off');

            % 4. Draw slot frameworks (empty card slots as faint dashed rounded rectangles)
            if N_total > 0
                cols = 10;
                rows = ceil(N_total / cols);

                X_start = 0.5;
                X_end = 7.5;
                Y_start = 1.0;
                Y_end = 9.0;

                cell_w = (X_end - X_start) / cols;
                cell_h = (Y_end - Y_start) / rows;

                hold(obj.AxesHandle, 'on');

                for idx = 1:N_total
                    c = mod(idx - 1, cols) + 1;
                    r = ceil(idx / cols);

                    x_center = X_start + (c - 0.5) * cell_w;
                    y_center = Y_end - (r - 0.5) * cell_h;

                    w = cell_w * 0.85;
                    h = cell_h * 0.85;

                    rectangle(obj.AxesHandle, ...
                        'Position', [x_center - w/2, y_center - h/2, w, h], ...
                        'FaceColor', 'none', ...
                        'EdgeColor', [0.18, 0.20, 0.25], ...
                        'LineStyle', '--', ...
                        'LineWidth', 1.0, ...
                        'Curvature', 0.15);
                end

                hold(obj.AxesHandle, 'off');
            end

            % 5. Draw undrawn deck pile initially
            obj.updateUndrawnCount(N_total);
        end

        function addCard(obj, cardType, index, pRed)
            % ADDCARD Adds a card patch dynamically at the designated grid position.
            arguments
                obj
                cardType
                index (1, 1) double {mustBePositive, mustBeInteger}
                pRed (1, 1) double
            end

            % Determine card color type: 1 for Red, 2 for Black
            if isnumeric(cardType)
                cardTypeVal = cardType; % 1 or 2
            else
                if strcmpi(cardType, 'Red') || strcmpi(cardType, 'R')
                    cardTypeVal = 1;
                else
                    cardTypeVal = 2;
                end
            end

            N_total = obj.R + obj.B;
            if index > N_total
                return;
            end

            % Set grid layout cols and rows
            cols = 10;
            rows = ceil(N_total / cols);

            % Grid margins
            X_start = 0.5;
            X_end = 7.5;
            Y_start = 1.0;
            Y_end = 9.0;

            cell_w = (X_end - X_start) / cols;
            cell_h = (Y_end - Y_start) / rows;

            c = mod(index - 1, cols) + 1;
            r = ceil(index / cols);

            x_center = X_start + (c - 0.5) * cell_w;
            y_center = Y_end - (r - 0.5) * cell_h;

            w = cell_w * 0.85;
            h = cell_h * 0.85;

            x_min = x_center - w/2;
            y_min = y_center - h/2;

            % Save to internal structures
            obj.ProbValues(index) = pRed;
            obj.CardPatches.Positions(index, :) = [x_min, y_min, w, h];
            obj.CardPatches.Types(index) = cardTypeVal;

            % Color tokens
            if cardTypeVal == 1
                % Red card (coral/neon red)
                faceCol = [0.22, 0.14, 0.16];
                edgeCol = [0.85, 0.30, 0.30];
                suitColor = [0.85, 0.30, 0.30];
                labelColor = [0.85, 0.30, 0.30];
                labelText = "RED";
                % Pick suit character (heart or diamond)
                if mod(index, 2) == 0
                    suitStr = char(9829); % Heart ♥
                else
                    suitStr = char(9830); % Diamond ♦
                end
            else
                % Black card (cool slate/deep gunmetal)
                faceCol = [0.15, 0.17, 0.22];
                edgeCol = [0.30, 0.35, 0.45];
                suitColor = [0.55, 0.58, 0.63];
                labelColor = [0.55, 0.58, 0.63];
                labelText = "BLACK";
                % Pick suit character (spade or club)
                if mod(index, 2) == 0
                    suitStr = char(9824); % Spade ♠
                else
                    suitStr = char(9827); % Club ♣
                end
            end

            hold(obj.AxesHandle, 'on');

            % Draw the actual card rounded rectangle
            obj.CardPatches.Handles(index) = rectangle(obj.AxesHandle, ...
                'Position', [x_min, y_min, w, h], ...
                'FaceColor', faceCol, ...
                'EdgeColor', edgeCol, ...
                'LineWidth', 1.5, ...
                'Curvature', 0.15);

            % Add labels (Suit & text)
            obj.CardPatches.Labels{index} = struct();

            % Center Suit Text
            obj.CardPatches.Labels{index}.Suit = text(obj.AxesHandle, x_center, y_center, suitStr, ...
                'Color', suitColor, ...
                'FontName', 'Courier New', ...
                'FontSize', 12, ...
                'FontWeight', 'bold', ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'Interpreter', 'none');

            % Text label (RED or BLACK) at the bottom edge
            obj.CardPatches.Labels{index}.Label = text(obj.AxesHandle, x_center, y_min + h*0.2, labelText, ...
                'Color', labelColor, ...
                'FontName', 'Helvetica', ...
                'FontSize', 7, ...
                'FontWeight', 'bold', ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'Interpreter', 'none');

            hold(obj.AxesHandle, 'off');

            % Make sure text and tooltips are on top
            uistack(obj.CardPatches.Labels{index}.Suit, 'top');
            uistack(obj.CardPatches.Labels{index}.Label, 'top');
            if isgraphics(obj.TooltipRect)
                uistack(obj.TooltipRect, 'top');
            end
            if isgraphics(obj.TooltipText)
                uistack(obj.TooltipText, 'top');
            end
        end

        function updateUndrawnCount(obj, count)
            % UPDATEUNDRAWNCOUNT Redraws the 3D-stacked card pile layout with a neon label.
            arguments
                obj
                count (1, 1) double {mustBeNonnegative, mustBeInteger}
            end

            % Delete existing stack
            if ~isempty(obj.DeckPilePatches)
                delete(obj.DeckPilePatches);
                obj.DeckPilePatches = gobjects(0);
            end
            if isgraphics(obj.DeckCountLabel)
                delete(obj.DeckCountLabel);
            end

            if count > 0
                X_center = 8.75;
                Y_center = 5.0;
                w_deck = 1.0;
                h_deck = 1.5;
                offset = 0.06;

                num_stacked = min(3, count);
                obj.DeckPilePatches = gobjects(1, num_stacked);

                hold(obj.AxesHandle, 'on');

                % Draw stacked 3D effect from back to front
                for s = num_stacked:-1:1
                    dx = (s - 1) * offset;
                    dy = (s - 1) * offset;
                    obj.DeckPilePatches(s) = rectangle(obj.AxesHandle, ...
                        'Position', [X_center - w_deck/2 + dx, Y_center - h_deck/2 + dy, w_deck, h_deck], ...
                        'FaceColor', [0.12, 0.18, 0.26], ...
                        'EdgeColor', [0.30, 0.65, 0.95], ...
                        'LineWidth', 1.5, ...
                        'Curvature', 0.15);
                end

                % Neon text label on the top card of the pile
                obj.DeckCountLabel = text(obj.AxesHandle, X_center + (num_stacked-1)*offset, Y_center + (num_stacked-1)*offset, ...
                    [string(count), "Cards"], ...
                    'Color', [0.30, 0.65, 0.95], ...
                    'FontName', 'Courier New', ...
                    'FontSize', 11, ...
                    'FontWeight', 'bold', ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', ...
                    'Interpreter', 'none');

                hold(obj.AxesHandle, 'off');
                uistack(obj.DeckCountLabel, 'top');
            end
        end

        function checkHover(obj, xCursor, yCursor, oRed, oBlack)
            % CHECKHOVER Checks if cursor is hovering over a card and displays customized in-axes tooltip.
            arguments
                obj
                xCursor (1, 1) double
                yCursor (1, 1) double
                oRed (1, 1) double = 2.0
                oBlack (1, 1) double = 2.0
            end

            hoverIndex = 0;
            if ~isempty(obj.CardPatches) && isfield(obj.CardPatches, 'Positions')
                N = size(obj.CardPatches.Positions, 1);
                for i = 1:N
                    if obj.CardPatches.Types(i) > 0
                        pos = obj.CardPatches.Positions(i, :);
                        if xCursor >= pos(1) && xCursor <= pos(1) + pos(3) && ...
                           yCursor >= pos(2) && yCursor <= pos(2) + pos(4)
                            hoverIndex = i;
                            break;
                        end
                    end
                end
            end

            if hoverIndex > 0
                t = hoverIndex;
                pRed = obj.ProbValues(t);
                cardType = obj.CardPatches.Types(t);
                if cardType == 1
                    typeName = "RED";
                    payoff = oRed * pRed;
                else
                    typeName = "BLACK";
                    payoff = oBlack * (1 - pRed);
                end

                % Formulate plain Unicode text (NO LaTeX)
                tooltipStr = sprintf(...
                    "Draw Index: %d (%s)\n" + ...
                    "P(Red)    : %.3f\n" + ...
                    "P(Black)  : %.3f\n" + ...
                    "Payoff    : %.3fx", ...
                    t, typeName, pRed, 1 - pRed, payoff);

                % Position the tooltip slightly offset from the cursor
                tw = 2.5;
                th = 1.3;
                tx = xCursor + 0.15;
                ty = yCursor + 0.15;

                % Boundary check so the tooltip doesn't get cut off
                if tx + tw > 10
                    tx = xCursor - tw - 0.15;
                end
                if ty + th > 10
                    ty = yCursor - th - 0.15;
                end

                set(obj.TooltipRect, ...
                    'Position', [tx, ty, tw, th], ...
                    'Visible', 'on');
                set(obj.TooltipText, ...
                    'Position', [tx + 0.1, ty + th/2], ...
                    'String', tooltipStr, ...
                    'Visible', 'on');

                % Put tooltip objects on top layer
                uistack(obj.TooltipRect, 'top');
                uistack(obj.TooltipText, 'top');
            else
                % Hide tooltip
                if isgraphics(obj.TooltipRect)
                    set(obj.TooltipRect, 'Visible', 'off');
                end
                if isgraphics(obj.TooltipText)
                    set(obj.TooltipText, 'Visible', 'off');
                end
            end
        end

        function clear(obj)
            % CLEAR Wipes all card patches and resets the probability history.
            if isstruct(obj.CardPatches)
                if isfield(obj.CardPatches, 'Handles')
                    for i = 1:length(obj.CardPatches.Handles)
                        if isgraphics(obj.CardPatches.Handles(i))
                            delete(obj.CardPatches.Handles(i));
                        end
                    end
                end
                if isfield(obj.CardPatches, 'Labels')
                    for i = 1:length(obj.CardPatches.Labels)
                        if isstruct(obj.CardPatches.Labels{i})
                            if isfield(obj.CardPatches.Labels{i}, 'Suit') && isgraphics(obj.CardPatches.Labels{i}.Suit)
                                delete(obj.CardPatches.Labels{i}.Suit);
                            end
                            if isfield(obj.CardPatches.Labels{i}, 'Label') && isgraphics(obj.CardPatches.Labels{i}.Label)
                                delete(obj.CardPatches.Labels{i}.Label);
                            end
                        end
                    end
                end
            end

            if ~isempty(obj.DeckPilePatches)
                delete(obj.DeckPilePatches);
                obj.DeckPilePatches = gobjects(0);
            end
            if isgraphics(obj.DeckCountLabel)
                delete(obj.DeckCountLabel);
            end

            if isgraphics(obj.TooltipRect)
                set(obj.TooltipRect, 'Visible', 'off');
            end
            if isgraphics(obj.TooltipText)
                set(obj.TooltipText, 'Visible', 'off');
            end

            % Re-initialize properties
            obj.CardPatches = struct();
            obj.CardPatches.Handles = gobjects(0);
            obj.CardPatches.Labels = {};
            obj.CardPatches.Positions = [];
            obj.CardPatches.Types = [];
            obj.ProbValues = [];
        end
    end
end
