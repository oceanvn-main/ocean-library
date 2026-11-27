classdef MatrixRenderer < OceanViz.App.Replay.Renderers.IChannelRenderer
    % MATRIXRENDERER  A matrix as a heat map with a colour bar (the value
    %   legend); row/column labels when given and short.

    methods
        function tf = canRender(~, info)
            tf = info.Kind == "matrix";
        end

        function draw(~, ax, value, info, ctx)
            OceanViz.App.Replay.Renderers.IChannelRenderer.prepare(ax);
            imagesc(ax, double(value));
            colormap(ax, parula(256));
            ax.YDir = 'reverse'; axis(ax, 'tight');
            xlabel(ax, 'column'); ylabel(ax, 'row');
            OceanViz.App.Replay.Renderers.IChannelRenderer.finish(ax, ctx.Theme, true);
            colorbar(ax, 'Color', ctx.Theme.Text, 'FontSize', 8);
            if ~isempty(info.RowLabels) && numel(info.RowLabels) <= 40
                ax.YTick = 1:numel(info.RowLabels); ax.YTickLabel = cellstr(info.RowLabels);
            end
            if ~isempty(info.ColLabels) && numel(info.ColLabels) <= 40
                ax.XTick = 1:numel(info.ColLabels); ax.XTickLabel = cellstr(info.ColLabels);
            end
        end
    end
end
