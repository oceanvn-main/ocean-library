classdef (Abstract) PanelBase < OceanViz.App.ModelLab.IPanel
    % PANELBASE  Common layout for Model Lab panels: a sidebar of controls on
    %   the left and a grid of image views / plots / text on the right.

    properties (SetAccess = protected)
        Ctx
        Side                    % OceanViz.App.Common.Sidebar
        Main                    % uigridlayout for the views
    end

    methods (Access = protected)
        function layout(obj, parent, ctx, rows, cols)
            obj.Ctx = ctx;
            g = uigridlayout(parent, [1 2], 'ColumnWidth', {260, '1x'}, 'Padding', [6 6 6 6], ...
                'ColumnSpacing', 8, 'BackgroundColor', ctx.Theme.Background);
            obj.Side = OceanViz.App.Common.Sidebar(g, ctx);
            obj.Side.Grid.Layout.Column = 1;
            obj.Main = uigridlayout(g, [rows cols], 'Padding', [0 0 0 0], 'RowSpacing', 6, 'ColumnSpacing', 6, ...
                'BackgroundColor', ctx.Theme.Background);
            obj.Main.Layout.Column = 2;
        end

        function v = view(obj, row, col, titleText)
            ax = uiaxes(obj.Main);
            ax.Layout.Row = row; ax.Layout.Column = col;
            v = OceanViz.App.Common.ImageView(ax, titleText, obj.Ctx);
        end

        function ax = plotAxes(obj, row, col)
            ax = uiaxes(obj.Main);
            ax.Layout.Row = row; ax.Layout.Column = col;
            c = obj.Ctx.Theme;
            ax.Color = c.Background; ax.XColor = c.Text; ax.YColor = c.Text; ax.GridColor = c.Grid;
            ax.Title.Color = c.Text; ax.Title.FontWeight = 'normal';
            grid(ax, 'on');
        end

        function t = textBox(obj, row, col)
            c = obj.Ctx.Theme;
            t = uitextarea(obj.Main, 'Editable', 'off', 'FontName', 'Consolas', 'FontSize', 12, ...
                'BackgroundColor', c.Panel, 'FontColor', c.Text);
            t.Layout.Row = row; t.Layout.Column = col;
        end

        function names = classItems(obj)
            names = ["(unconditional)", obj.Ctx.Model.ClassNames];
        end

        function y = selectedLabel(~, dd)
            % 0 = unconditional, k = class k.
            y = find(dd.Items == string(dd.Value)) - 1;
        end
    end
end
