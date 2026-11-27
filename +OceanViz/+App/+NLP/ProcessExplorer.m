classdef ProcessExplorer < handle
    %PROCESSEXPLORER Text-first inspector for OceanNLP.Inspect readers.
    %   app = OceanViz.App.NLP.ProcessExplorer(reader) opens a separate app.
    %   The reader supplies listSteps(), getStep(id), and getArtifact(id).

    properties (SetAccess = private)
        Reader
        Figure
        SelectedId (1,1) string = ""
        Steps = struct([])
    end

    properties (Access = private)
        Viz
        SearchField
        MapTable
        Header
        Summary
        InputText
        DecisionText
        OutputText
        ContextText
        ResultTabs
        TextTab
        TableTab
        StructureTab
        TextArea
        DataTable
        GraphAxes
        RawArea
        PositionLabel
        VisibleSteps = struct([])
        PageOffset (1,1) double = 1
        HasMore (1,1) logical = false
    end

    properties (Constant, Access = private)
        PageSize = 200
    end

    methods
        function obj = ProcessExplorer(reader, opts)
            arguments
                reader
                opts.Visible (1,1) logical = true
                opts.Theme (1,1) string = "Dark"
            end
            obj.Reader = reader;
            obj.Viz = OceanViz.Core('Theme', opts.Theme);
            c = obj.Viz.Colors;
            obj.Figure = uifigure('Name', 'NLP Process Explorer', ...
                'Position', [80 80 1420 850], 'Color', c.Background, ...
                'Visible', matlab.lang.OnOffSwitchState(opts.Visible), ...
                'KeyPressFcn', @(~, e) obj.onKey(e));
            root = uigridlayout(obj.Figure, [3 3]);
            root.RowHeight = {60, '1x', 48};
            root.ColumnWidth = {280, '1x', 285};
            root.BackgroundColor = c.Background;
            titleLabel = uilabel(root, 'Text', 'PROCESS EXPLORER', 'FontSize', 20, ...
                'FontWeight', 'bold', 'FontColor', c.Text);
            titleLabel.Layout.Row = 1; titleLabel.Layout.Column = [1 3];

            left = uigridlayout(root, [2 1]);
            left.Layout.Row = 2; left.Layout.Column = 1;
            left.RowHeight = {35, '1x'}; left.BackgroundColor = c.Background;
            obj.SearchField = uieditfield(left, 'text', 'Placeholder', 'Search steps', ...
                'ValueChangedFcn', @(~, ~) obj.search(string(obj.SearchField.Value)));
            obj.MapTable = uitable(left, 'ColumnName', {'Stage', 'Step', 'Status'}, ...
                'ColumnEditable', [false false false], ...
                'CellSelectionCallback', @(~, e) obj.onMapSelection(e));
            obj.MapTable.RowName = {};
            obj.MapTable.ColumnWidth = {85, 'auto', 75};

            center = uigridlayout(root, [5 1]);
            center.Layout.Row = 2; center.Layout.Column = 2;
            center.RowHeight = {34, 55, 28, 102, '1x'};
            center.BackgroundColor = c.Background;
            obj.Header = uilabel(center, 'Text', 'Select a process step', ...
                'FontSize', 18, 'FontWeight', 'bold', 'FontColor', c.Text);
            obj.Summary = uitextarea(center, 'Editable', 'off');
            uilabel(center, 'Text', 'INPUT     →     DECISION     →     OUTPUT', ...
                'FontWeight', 'bold', 'FontColor', c.Text);
            account = uigridlayout(center, [1 3]);
            account.ColumnWidth = {'1x', '1x', '1x'};
            obj.InputText = uitextarea(account, 'Editable', 'off');
            obj.DecisionText = uitextarea(account, 'Editable', 'off');
            obj.OutputText = uitextarea(account, 'Editable', 'off');
            obj.ResultTabs = uitabgroup(center);
            obj.TextTab = uitab(obj.ResultTabs, 'Title', 'Text');
            obj.TextArea = uitextarea(uigridlayout(obj.TextTab, [1 1]), 'Editable', 'off');
            obj.TableTab = uitab(obj.ResultTabs, 'Title', 'Table');
            obj.DataTable = uitable(uigridlayout(obj.TableTab, [1 1]));
            obj.StructureTab = uitab(obj.ResultTabs, 'Title', 'Structure');
            obj.GraphAxes = uiaxes(uigridlayout(obj.StructureTab, [1 1]));
            obj.Viz.styleAxes(obj.GraphAxes);
            t = uitab(obj.ResultTabs, 'Title', 'Raw');
            obj.RawArea = uitextarea(uigridlayout(t, [1 1]), 'Editable', 'off');

            right = uigridlayout(root, [2 1]);
            right.Layout.Row = 2; right.Layout.Column = 3;
            right.RowHeight = {32, '1x'}; right.BackgroundColor = c.Background;
            uilabel(right, 'Text', 'CONTEXT', 'FontWeight', 'bold', 'FontColor', c.Text);
            obj.ContextText = uitextarea(right, 'Editable', 'off');

            foot = uigridlayout(root, [1 4]);
            foot.Layout.Row = 3; foot.Layout.Column = [1 3];
            foot.ColumnWidth = {100, 100, '1x', 100};
            uibutton(foot, 'Text', '← Previous', 'ButtonPushedFcn', @(~, ~) obj.move(-1));
            uibutton(foot, 'Text', 'Next →', 'ButtonPushedFcn', @(~, ~) obj.move(1));
            obj.PositionLabel = uilabel(foot, 'Text', '', 'FontColor', c.Text);
            uibutton(foot, 'Text', 'Refresh', 'ButtonPushedFcn', @(~, ~) obj.reload());
            obj.reload();
        end

        function reload(obj)
            arguments
                obj (1,1) OceanViz.App.NLP.ProcessExplorer
            end
            obj.PageOffset = 1;
            obj.loadPage();
        end

        function search(obj, query)
            arguments
                obj (1,1) OceanViz.App.NLP.ProcessExplorer
                query (1,1) string
            end
            obj.SearchField.Value = query;
            obj.PageOffset = 1;
            obj.loadPage();
        end

        function selectStep(obj, id)
            arguments
                obj (1,1) OceanViz.App.NLP.ProcessExplorer
                id (1,1) string
            end
            step = obj.Reader.getStep(id);
            obj.SelectedId = id;
            obj.Header.Text = string(step.Name);
            obj.Summary.Value = string(step.Summary);
            obj.InputText.Value = obj.describeIds(step.InputIds, 'No recorded input');
            obj.DecisionText.Value = string(step.Summary);
            obj.OutputText.Value = obj.describeIds(step.OutputIds, 'No recorded output');
            obj.ContextText.Value = ["Stage: " + string(step.Stage); ...
                "Status: " + string(step.Status); ...
                "Sequence: " + string(step.Sequence); ...
                "Step ID: " + id; ...
                "Input IDs: " + strjoin(string(step.InputIds), ', '); ...
                "Output IDs: " + strjoin(string(step.OutputIds), ', ')];
            obj.renderResult(step);
            idx = find(string({obj.VisibleSteps.Id}) == id, 1);
            if ~isempty(idx)
                obj.PositionLabel.Text = string(idx) + " / " + ...
                    string(numel(obj.VisibleSteps)) + " shown";
            end
        end

        function move(obj, delta)
            arguments
                obj (1,1) OceanViz.App.NLP.ProcessExplorer
                delta (1,1) double {mustBeInteger}
            end
            if isempty(obj.VisibleSteps), return; end
            ids = string({obj.VisibleSteps.Id});
            idx = find(ids == obj.SelectedId, 1);
            if isempty(idx)
                idx = 1;
            else
                if delta > 0 && idx == numel(ids) && obj.HasMore
                    obj.PageOffset = obj.PageOffset + obj.PageSize;
                    obj.loadPage();
                    return;
                elseif delta < 0 && idx == 1 && obj.PageOffset > 1
                    obj.PageOffset = max(1, obj.PageOffset - obj.PageSize);
                    obj.loadPage();
                    obj.selectStep(string(obj.VisibleSteps(end).Id));
                    return;
                end
                idx = min(max(idx + delta, 1), numel(ids));
            end
            obj.selectStep(ids(idx));
        end

        function close(obj)
            arguments
                obj (1,1) OceanViz.App.NLP.ProcessExplorer
            end
            if isgraphics(obj.Figure), delete(obj.Figure); end
        end
    end

    methods (Access = private)
        function clearSelection(obj, message)
            obj.SelectedId = "";
            obj.Header.Text = string(message);
            obj.Summary.Value = "";
            obj.InputText.Value = "";
            obj.DecisionText.Value = "";
            obj.OutputText.Value = "";
            obj.ContextText.Value = "";
            obj.TextArea.Value = "";
            obj.RawArea.Value = "";
            obj.DataTable.Data = cell(0, 2);
            cla(obj.GraphAxes);
            obj.PositionLabel.Text = "0 / 0 shown";
        end

        function loadPage(obj)
            query = strtrim(string(obj.SearchField.Value));
            try
                fetched = obj.Reader.listSteps('Query', query, ...
                    'Offset', obj.PageOffset, 'Limit', obj.PageSize + 1);
            catch originalError
                try
                    allSteps = obj.Reader.listSteps();
                catch
                    rethrow(originalError);
                end
                if strlength(query) > 0 && ~isempty(allSteps)
                    names = lower(string({allSteps.Name}));
                    stages = lower(string({allSteps.Stage}));
                    summaries = lower(string({allSteps.Summary}));
                    q = lower(query);
                    allSteps = allSteps(contains(names, q) | contains(stages, q) | contains(summaries, q));
                end
                first = min(obj.PageOffset, numel(allSteps) + 1);
                last = min(numel(allSteps), first + obj.PageSize);
                fetched = allSteps(first:last);
            end
            obj.HasMore = numel(fetched) > obj.PageSize;
            steps = fetched(1:min(numel(fetched), obj.PageSize));
            obj.Steps = steps;
            obj.VisibleSteps = steps;
            if isempty(steps)
                obj.MapTable.Data = cell(0, 3);
                obj.clearSelection("No steps match the search.");
                return;
            end
            obj.MapTable.Data = [cellstr(string({steps.Stage})'), ...
                cellstr(string({steps.Name})'), cellstr(string({steps.Status})')];
            if ~any(string({steps.Id}) == obj.SelectedId)
                obj.selectStep(string(steps(1).Id));
            end
        end

        function onMapSelection(obj, event)
            if isempty(event.Indices), return; end
            row = event.Indices(1, 1);
            if row <= numel(obj.VisibleSteps)
                obj.selectStep(string(obj.VisibleSteps(row).Id));
            end
        end

        function onKey(obj, event)
            switch string(event.Key)
                case {'downarrow', 'rightarrow'}
                    obj.move(1);
                case {'uparrow', 'leftarrow'}
                    obj.move(-1);
            end
        end

        function lines = describeIds(obj, ids, emptyText)
            if isempty(ids), lines = string(emptyText); return; end
            ids = string(ids);
            lines = strings(numel(ids), 1);
            for k = 1:numel(ids)
                try
                    a = obj.Reader.getArtifact(ids(k));
                    lines(k) = string(a.Name) + " · " + string(a.Kind) + " — " + string(a.Summary);
                catch
                    lines(k) = ids(k) + " · unavailable";
                end
            end
        end

        function renderResult(obj, step)
            obj.TextArea.Value = "No text result recorded.";
            obj.RawArea.Value = "No output artifact recorded.";
            obj.DataTable.Data = cell(0, 2);
            cla(obj.GraphAxes);
            obj.GraphAxes.Visible = 'off';
            ids = string(step.OutputIds);
            if isempty(ids), return; end
            a = obj.Reader.getArtifact(ids(1));
            if isfield(a, 'Status') && string(a.Status) == "omitted"
                obj.TextArea.Value = [string(a.Name) + " was omitted during capture."; string(a.Summary)];
                obj.RawArea.Value = "Artifact unavailable: omitted by capture budget.";
                obj.ResultTabs.SelectedTab = obj.TextTab;
                return;
            end
            v = a.Value;
            obj.RawArea.Value = obj.formatValue(v);
            graphData = struct();
            if isstruct(v) && isscalar(v) && isfield(v, 'Graph') && ...
                    (isa(v.Graph, 'graph') || isa(v.Graph, 'digraph'))
                graphData = v;
                v = v.Graph;
            end
            if isa(v, 'graph') || isa(v, 'digraph')
                obj.GraphAxes.Visible = 'on';
                obj.Viz.styleAxes(obj.GraphAxes);
                maxVisibleNodes = 200;
                if numnodes(v) > maxVisibleNodes
                    visibleGraph = subgraph(v, 1:maxVisibleNodes);
                    scope = "first " + string(maxVisibleNodes) + " of " + string(numnodes(v)) + " nodes";
                else
                    visibleGraph = v;
                    scope = string(numnodes(v)) + " nodes";
                end
                gp = plot(obj.GraphAxes, visibleGraph, 'NodeColor', obj.Viz.Colors.Primary, ...
                    'EdgeColor', obj.Viz.Colors.Grid);
                obj.highlightGraph(gp, graphData, numnodes(visibleGraph));
                title(obj.GraphAxes, string(a.Name) + " · " + scope + " · layout, not embedding coordinates");
                obj.TextArea.Value = [string(a.Summary); "Showing " + scope + "."];
                obj.ResultTabs.SelectedTab = obj.StructureTab;
            elseif istable(v)
                obj.DataTable.Data = v(1:min(height(v), 100), 1:min(width(v), 12));
                obj.TextArea.Value = [string(a.Summary); "Preview: " + string(height(obj.DataTable.Data)) + ...
                    " / " + string(height(v)) + " rows, " + string(width(obj.DataTable.Data)) + ...
                    " / " + string(width(v)) + " columns."];
                obj.ResultTabs.SelectedTab = obj.TableTab;
            elseif isstruct(v) && isscalar(v)
                keys = fieldnames(v);
                totalFields = numel(keys);
                keys = keys(1:min(totalFields, 100));
                vals = strings(numel(keys), 1);
                for k = 1:numel(keys), vals(k) = strjoin(obj.formatValue(v.(keys{k})), newline); end
                obj.DataTable.Data = table(string(keys), vals, 'VariableNames', {'Field', 'Value'});
                obj.TextArea.Value = [string(a.Summary); "Preview: " + string(numel(keys)) + ...
                    " / " + string(totalFields) + " fields."];
                obj.ResultTabs.SelectedTab = obj.TableTab;
            else
                obj.TextArea.Value = obj.formatValue(v);
                obj.ResultTabs.SelectedTab = obj.TextTab;
            end
        end

        function s = formatValue(~, v)
            if isstring(v) || ischar(v)
                s = string(v);
                if numel(s) > 100, s = [s(1:100); "… additional values omitted"]; end
                for k = 1:numel(s)
                    if strlength(s(k)) > 4000
                        s(k) = extractBetween(s(k), 1, 4000) + "…";
                    end
                end
            elseif isnumeric(v) || islogical(v)
                shape = size(v);
                sample = v(1:min(numel(v), 100));
                s = "Size " + join(string(shape), " × ") + newline + ...
                    string(evalc('disp(sample)'));
                if numel(v) > numel(sample), s = s + newline + "… additional values omitted"; end
            elseif istable(v)
                s = "Table: " + string(height(v)) + " rows × " + string(width(v)) + " columns";
            elseif isstruct(v)
                if isempty(v)
                    s = "Empty struct";
                else
                    fields = string(fieldnames(v(1)));
                    s = "Struct: " + string(numel(v)) + " elements; fields: " + ...
                        strjoin(fields(1:min(numel(fields), 30)), ", ");
                end
            elseif iscell(v)
                s = "Cell: " + join(string(size(v)), " × ");
            else
                s = "<" + string(class(v)) + ">";
            end
        end

        function highlightGraph(obj, gp, metadata, visibleCount)
            if isempty(fieldnames(metadata)), return; end
            names = {'VisitedNodes', 'CandidateNodes', 'SelectedNodes'};
            colors = {obj.Viz.Colors.Secondary, obj.Viz.Colors.Boundary, obj.Viz.Colors.Primary};
            handles = gobjects(0);
            labels = strings(0);
            hold(obj.GraphAxes, 'on');
            for k = 1:numel(names)
                name = names{k};
                if ~isfield(metadata, name), continue; end
                nodes = metadata.(name);
                if ~isnumeric(nodes), continue; end
                nodes = unique(nodes(:));
                nodes = nodes(isfinite(nodes) & nodes >= 1 & nodes <= visibleCount & mod(nodes, 1) == 0);
                if ~isempty(nodes)
                    highlight(gp, nodes, 'NodeColor', colors{k}, 'MarkerSize', 6 + 2*k);
                    handles(end+1) = scatter(obj.GraphAxes, NaN, NaN, 36, colors{k}, 'filled'); %#ok<AGROW>
                    labels(end+1) = erase(string(name), "Nodes"); %#ok<AGROW>
                end
            end
            if ~isempty(handles)
                lgd = legend(obj.GraphAxes, handles, cellstr(labels), 'Location', 'best');
                lgd.Color = obj.Viz.Colors.Background;
                lgd.TextColor = obj.Viz.Colors.Text;
                lgd.EdgeColor = obj.Viz.Colors.Grid;
            end
            hold(obj.GraphAxes, 'off');
        end
    end
end
