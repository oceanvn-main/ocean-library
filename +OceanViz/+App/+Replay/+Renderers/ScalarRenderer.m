classdef ScalarRenderer < OceanViz.App.Replay.Renderers.IChannelRenderer
    % SCALARRENDERER  A scalar channel as its number, with its symbol
    %   (the timeline shows its history).

    methods
        function tf = canRender(~, info)
            tf = info.Kind == "scalar";
        end

        function draw(~, ax, value, info, ctx)
            OceanViz.App.Replay.Renderers.IChannelRenderer.prepare(ax);
            text(ax, 0.5, 0.55, sprintf('%.6g', double(value)), 'HorizontalAlignment', 'center', ...
                'FontSize', 20, 'Color', ctx.Theme.Text);
            text(ax, 0.5, 0.2, OceanViz.App.Replay.Notation.latex(info), 'HorizontalAlignment', 'center', 'FontSize', 12, ...
                'Color', ctx.Theme.Text, 'Interpreter', 'latex');
            ax.XLim = [0 1]; ax.YLim = [0 1];
            OceanViz.App.Replay.Renderers.IChannelRenderer.finish(ax, ctx.Theme, false);
        end
    end
end
