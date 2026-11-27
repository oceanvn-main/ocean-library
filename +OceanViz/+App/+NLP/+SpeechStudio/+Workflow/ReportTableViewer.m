classdef ReportTableViewer < handle
    % REPORTTABLEVIEWER  Pick a machine-generated report and one of its tables; shows the
    %   table verbatim with its source file, generator line and line number.
    %   Used by ComparisonTab and ValidationTab (Models.ReportTables does the parsing).

    properties (SetAccess = private)
        Grid
        ReportDropDown
        TableDropDown
        Table
        SourceLabel
        Files (1,:) string = strings(1, 0)
        Report = []
    end

    methods
        function obj = ReportTableViewer(parent, files)
            arguments
                parent
                files (1,:) string
            end
            obj.Files = files;
            obj.Grid = uigridlayout(parent, [3 2], 'RowHeight', {'fit', 'fit', '1x'}, ...
                'ColumnWidth', {'1x', '2x'}, 'Padding', [0 0 0 0]);
            names = strings(1, numel(files));
            for k = 1:numel(files)
                [~, n, e] = fileparts(files(k)); names(k) = n + e;
            end
            if isempty(names), names = "(no reports found)"; end
            obj.ReportDropDown = uidropdown(obj.Grid, 'Items', cellstr(names), ...
                'ItemsData', 1:numel(names), 'ValueChangedFcn', @(d, ~) obj.showReport(d.Value));
            obj.ReportDropDown.Layout.Row = 1; obj.ReportDropDown.Layout.Column = 1;
            obj.TableDropDown = uidropdown(obj.Grid, 'Items', {'(none)'}, ...
                'ValueChangedFcn', @(d, ~) obj.showTable(d.Value));
            obj.TableDropDown.Layout.Row = 1; obj.TableDropDown.Layout.Column = 2;
            obj.SourceLabel = uilabel(obj.Grid, 'Text', '', 'WordWrap', 'on', 'FontAngle', 'italic');
            obj.SourceLabel.Layout.Row = 2; obj.SourceLabel.Layout.Column = [1 2];
            obj.Table = uitable(obj.Grid, 'RowName', {}, 'ColumnEditable', false);
            obj.Table.Layout.Row = 3; obj.Table.Layout.Column = [1 2];
            if ~isempty(files), obj.showReport(1); end
        end

        function showReport(obj, k)
            obj.Report = OceanViz.App.NLP.SpeechStudio.Models.ReportTables.read(obj.Files(k));
            t = obj.Report.Tables;
            if isempty(t)
                obj.TableDropDown.Items = {'(no tables)'};
                obj.TableDropDown.ItemsData = {};
                obj.Table.Data = {};
                obj.SourceLabel.Text = char("Source: " + obj.Files(k));
                return;
            end
            labels = arrayfun(@(i) sprintf("%d. %s (line %d)", i, t(i).Heading, t(i).Line), 1:numel(t));
            obj.TableDropDown.Items = cellstr(labels);
            obj.TableDropDown.ItemsData = 1:numel(t);
            obj.ReportDropDown.Value = k;
            obj.showTable(1);
        end

        function showTable(obj, i)
            if isempty(obj.Report) || isempty(obj.Report.Tables), return; end
            t = obj.Report.Tables(i);
            obj.TableDropDown.Value = i;
            obj.Table.ColumnName = cellstr(t.Header);
            obj.Table.Data = cellstr(t.Rows);
            obj.SourceLabel.Text = char(sprintf("Source: %s, line %d. %s", obj.Report.File, t.Line, obj.Report.Generator));
        end

        function n = numTables(obj)
            n = 0;
            if ~isempty(obj.Report), n = numel(obj.Report.Tables); end
        end
    end
end
