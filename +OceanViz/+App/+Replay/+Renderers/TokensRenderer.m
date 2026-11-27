classdef TokensRenderer < OceanViz.App.Replay.Renderers.IChannelRenderer
    % TOKENSRENDERER  A token list as text: "[1] the  [2] cat ...".

    methods
        function tf = canRender(~, info)
            tf = info.Kind == "tokens";
        end

        function draw(~, ax, value, ~, ctx)
            OceanViz.App.Replay.Renderers.IChannelRenderer.prepare(ax);
            t = string(value(:)).';
            parts = compose("[%d] %s", (1:numel(t)).', t.');
            text(ax, 0.02, 0.5, char(strjoin(parts, "   ")), 'Color', ctx.Theme.Text, 'FontSize', 11, ...
                'Interpreter', 'none', 'VerticalAlignment', 'middle');
            ax.XLim = [0 1]; ax.YLim = [0 1];
            OceanViz.App.Replay.Renderers.IChannelRenderer.finish(ax, ctx.Theme, false);
        end
    end
end
