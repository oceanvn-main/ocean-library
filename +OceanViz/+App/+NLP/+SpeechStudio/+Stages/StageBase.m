classdef (Abstract) StageBase < OceanViz.App.NLP.SpeechStudio.Stages.IStage
    % STAGEBASE  Shared layout of every stage: plots (left, 2/3) and a notes column
    %   (right, 1/3) with LaTeX notation + symbol legend (always visible), a shapes &
    %   parameters table, provenance (Ocean class + file) and an "i  How it works" drawer
    %   for the long explanation (hidden until asked). Also owns the linked time cursor.
    %
    %   Subclasses implement:
    %     buildPlots(obj, grid)            create plot slots/controls in GRID (once)
    %     [rows, classes] = draw(obj, ctx) draw a run; ROWS = n x 2 string (quantity, value)
    %                                      actually used; CLASSES = Ocean classes that computed it
    %     L = notation(obj)                LaTeX lines (formulas, then "symbol: meaning")
    %     s = description(obj)             long explanation (drawer)
    %   and call obj.slotAxes(name) to get a fresh uiaxes in a slot, obj.addTimeAxes(ax,
    %   offset) for every axis whose x is time in seconds.

    properties (SetAccess = protected)
        Root = []
        PlotGrid = []
        NotesGrid = []
        ShapesTable = []
        ProvenanceArea = []
        Drawer = []
        InfoButton = []
        NotationGrid = []
        ShownNotation (1,:) string = strings(1, 0)
        ErrorLabel = []
        Slots struct = struct()
        TimeAxes cell = {}
        TimeOffsets double = zeros(1, 0)
        Cursors cell = {}
        Context = []
        LastError (1,1) string = ""
        Rows string = strings(0, 2)
        Classes (1,:) string = strings(1, 0)
    end

    methods (Abstract)
        buildPlots(obj, grid)
        [rows, classes] = draw(obj, ctx)
        L = notation(obj)
        s = description(obj)
    end

    methods
        function build(obj, parent)
            obj.Root = uigridlayout(parent, [1 2], 'ColumnWidth', {'2x', '1x'}, 'Padding', [6 6 6 6], ...
                'ColumnSpacing', 8);
            left = uigridlayout(obj.Root, [2 1], 'RowHeight', {0, '1x'}, 'Padding', [0 0 0 0], 'RowSpacing', 2);
            left.Layout.Column = 1;
            obj.ErrorLabel = uilabel(left, 'Text', '', 'FontColor', [0.85 0.2 0.2], 'WordWrap', 'on');
            obj.ErrorLabel.Layout.Row = 1;
            obj.PlotGrid = uigridlayout(left, [1 1], 'Padding', [0 0 0 0]);
            obj.PlotGrid.Layout.Row = 2;
            obj.buildPlots(obj.PlotGrid);
            obj.buildNotes();
        end

        function render(obj, ctx)
            obj.Context = ctx;
            obj.TimeAxes = {};
            obj.TimeOffsets = zeros(1, 0);
            obj.Cursors = {};
            obj.LastError = "";
            obj.ErrorLabel.Parent.RowHeight{1} = 0;
            try
                [rows, classes] = obj.draw(ctx);
            catch err
                obj.LastError = string(err.message);
                obj.ErrorLabel.Text = "Could not draw this stage: " + err.message;
                obj.ErrorLabel.Parent.RowHeight{1} = 'fit';
                rows = strings(0, 2); classes = strings(1, 0);
            end
            obj.Rows = rows;
            obj.Classes = unique(classes, 'stable');
            obj.ShapesTable.Data = cellstr(rows);
            obj.ProvenanceArea.Value = cellstr(obj.provenance(obj.Classes));
            obj.showNotation(obj.notationFor(ctx));
            obj.addCursors();
        end

        function L = notationFor(obj, ~)
            % NOTATIONFOR  Notation for this run (default: the static notation()).
            L = obj.notation();
        end

        function L = shownNotation(obj)
            L = obj.ShownNotation;
        end

        function setCursor(obj, t)
            for i = 1:numel(obj.Cursors)
                c = obj.Cursors{i};
                if ~isvalid(c), continue; end
                if isnan(t)
                    c.Visible = 'off';
                else
                    c.Value = t + obj.TimeOffsets(i);
                    c.Visible = 'on';
                end
            end
        end

        function t = pointerTime(obj, fig)
            % Time (segment-local, s) under the pointer when it is over a time axis.
            t = NaN;
            p = fig.CurrentPoint;
            for i = 1:numel(obj.TimeAxes)
                ax = obj.TimeAxes{i};
                if ~isvalid(ax), continue; end
                pos = getpixelposition(ax, true);
                if p(1) >= pos(1) && p(1) <= pos(1) + pos(3) && p(2) >= pos(2) && p(2) <= pos(2) + pos(4)
                    cp = ax.CurrentPoint;
                    xl = ax.XLim;
                    if cp(1, 1) >= xl(1) && cp(1, 1) <= xl(2)
                        t = cp(1, 1) - obj.TimeOffsets(i);
                    end
                    return;
                end
            end
        end

        function toggleDrawer(obj)
            if isequal(obj.NotesGrid.RowHeight{end}, 0)
                obj.NotesGrid.RowHeight{end} = '1x';
                obj.InfoButton.Text = char("ⓘ  Hide explanation");
            else
                obj.NotesGrid.RowHeight{end} = 0;
                obj.InfoButton.Text = char("ⓘ  How it works");
            end
        end

        function tf = drawerOpen(obj)
            tf = ~isequal(obj.NotesGrid.RowHeight{end}, 0);
        end
    end

    methods (Access = protected)
        function g = addSlots(obj, grid, names, rows, cols)
            % ADDSLOTS  Named 1x1 plot slots laid out row-major in a rows x cols grid.
            g = uigridlayout(grid, [rows cols], 'Padding', [0 0 0 0], 'RowSpacing', 6, 'ColumnSpacing', 6);
            for k = 1:numel(names)
                s = uigridlayout(g, [1 1], 'Padding', [0 0 0 0]);
                s.Layout.Row = ceil(k / cols);
                s.Layout.Column = mod(k - 1, cols) + 1;
                obj.Slots.(names(k)) = s;
            end
        end

        function ax = slotAxes(obj, name)
            % SLOTAXES  A fresh uiaxes in slot NAME (old axes, legend, colour bar removed).
            s = obj.Slots.(name);
            delete(s.Children);
            ax = uiaxes(s);
            ax.Tag = char("Studio_" + name);
        end

        function emptySlot(obj, name, message)
            ax = obj.slotAxes(name);
            obj.Context.Viz.styleAxes(ax);
            title(ax, message, 'Interpreter', 'none', 'Color', obj.Context.Viz.Colors.Text, 'FontSize', 10);
        end

        function addTimeAxes(obj, ax, offset)
            if nargin < 3, offset = 0; end
            obj.TimeAxes{end+1} = ax;
            obj.TimeOffsets(end+1) = offset;
        end

        function plot(obj, ax, strategy, data)
            obj.Context.Viz.plot(ax, strategy, data);
        end
    end

    methods (Access = private)
        function buildNotes(obj)
            p = uipanel(obj.Root, 'Title', 'Notes');
            p.Layout.Column = 2;
            g = uigridlayout(p, [9 1], 'RowHeight', {'fit', 'fit', 'fit', 'fit', '1x', 'fit', 70, 'fit', 0}, ...
                'Padding', [6 6 6 6], 'RowSpacing', 4);
            obj.NotesGrid = g;
            uilabel(g, 'Text', char(obj.title()), 'FontWeight', 'bold', 'FontSize', 14);
            uilabel(g, 'Text', 'Notation', 'FontWeight', 'bold');
            obj.NotationGrid = uigridlayout(g, [1 1], 'Padding', [0 0 0 0], 'RowSpacing', 2);
            obj.showNotation(obj.notation());
            uilabel(g, 'Text', 'Shapes & parameters (this run)', 'FontWeight', 'bold');
            obj.ShapesTable = uitable(g, 'ColumnName', {'Quantity', 'Value'}, 'RowName', {}, ...
                'ColumnEditable', [false false], 'ColumnWidth', {'auto', 'auto'});
            uilabel(g, 'Text', 'Source (Ocean class · file)', 'FontWeight', 'bold');
            obj.ProvenanceArea = uitextarea(g, 'Editable', 'off', 'FontSize', 11);
            obj.InfoButton = uibutton(g, 'Text', char("ⓘ  How it works"), ...
                'ButtonPushedFcn', @(~, ~) obj.toggleDrawer());
            obj.Drawer = uitextarea(g, 'Editable', 'off', 'Value', cellstr(obj.description()), 'WordWrap', 'on');
        end

        function showNotation(obj, L)
            if isequal(L, obj.ShownNotation), return; end
            ng = obj.NotationGrid;
            delete(ng.Children);
            n = max(numel(L), 1);
            ng.RowHeight = repmat({'fit'}, 1, n);
            for i = 1:numel(L)
                lb = uilabel(ng, 'Text', char(L(i)), 'Interpreter', 'latex', 'WordWrap', 'on', 'FontSize', 13);
                lb.Layout.Row = i;
            end
            obj.ShownNotation = L;
        end

        function addCursors(obj)
            obj.Cursors = cell(1, numel(obj.TimeAxes));
            for i = 1:numel(obj.TimeAxes)
                ax = obj.TimeAxes{i};
                hold(ax, 'on');
                obj.Cursors{i} = xline(ax, ax.XLim(1), '-', 'Color', [0.95 0.35 0.35], 'LineWidth', 1.2, ...
                    'HandleVisibility', 'off', 'Visible', 'off', 'Tag', 'StudioCursor', 'HitTest', 'off');
                hold(ax, 'off');
            end
        end
    end

    methods (Static)
        function lines = provenance(classes)
            % PROVENANCE  "class  (repo-relative file)" for each Ocean class.
            lines = strings(0, 1);
            root = OceanViz.App.NLP.SpeechStudio.Models.StudioSession.repoRoot();
            for c = classes
                f = string(which(c));
                if f == "", lines(end+1, 1) = c; continue; end %#ok<AGROW>
                rel = f;
                if startsWith(f, root), rel = extractAfter(f, strlength(root) + 1); end
                lines(end+1, 1) = c + "  (" + replace(rel, "\", "/") + ")"; %#ok<AGROW>
            end
        end

        function s = shape(x)
            s = strjoin(string(size(x)), " x ");
        end
    end
end
