classdef VectorRenderer < OceanViz.App.Replay.Renderers.IChannelRenderer
    % VECTORRENDERER  A vector as bars (short) or a line (long); a categorical
    %   vector (Categories set, e.g. cluster labels) as a coloured strip whose
    %   colour bar names the categories (the legend).

    methods
        function tf = canRender(~, info)
            tf = info.Kind == "vector";
        end

        function draw(~, ax, value, info, ctx)
            OceanViz.App.Replay.Renderers.IChannelRenderer.prepare(ax);
            v = double(value(:)).';
            T = ctx.Theme;
            if ~isempty(info.Categories)
                n = numel(info.Categories);
                image(ax, v, 'CDataMapping', 'direct');
                colormap(ax, lines(n));
                ax.XLim = [0.5 numel(v) + 0.5];
                xlabel(ax, 'element i');
                OceanViz.App.Replay.Renderers.IChannelRenderer.finish(ax, T, true);
                ax.YTick = [];
                colorbar(ax, 'Color', T.Text, 'Ticks', 1:n, 'TickLabels', cellstr(info.Categories), 'FontSize', 8);
                ax.CLim = [0.5 n + 0.5];
            else
                if numel(v) <= 64
                    bar(ax, v, 'FaceColor', T.Primary, 'EdgeColor', 'none');
                else
                    plot(ax, v, 'Color', T.Primary, 'LineWidth', 1); ax.XLim = [1 numel(v)];
                end
                xlabel(ax, 'element'); ylabel(ax, OceanViz.App.Replay.Notation.latex(info), 'Interpreter', 'latex');
                grid(ax, 'on');
                OceanViz.App.Replay.Renderers.IChannelRenderer.finish(ax, T, true);
            end
        end
    end
end
