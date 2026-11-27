classdef SequenceRenderer < OceanViz.App.Replay.Renderers.IChannelRenderer
    % SEQUENCERENDERER  A T x D sequence as D lines over position 1..T, with a
    %   legend naming the D columns (d = 1..D).

    methods
        function tf = canRender(~, info)
            tf = info.Kind == "sequence";
        end

        function draw(~, ax, value, info, ctx)
            OceanViz.App.Replay.Renderers.IChannelRenderer.prepare(ax);
            v = double(value);
            D = size(v, 2);
            ax.ColorOrder = lines(max(1, D));
            plot(ax, v, 'LineWidth', 1);
            ax.XLim = [1 max(2, size(v, 1))];
            xlabel(ax, 'position'); ylabel(ax, OceanViz.App.Replay.Notation.latex(info), 'Interpreter', 'latex');
            grid(ax, 'on');
            OceanViz.App.Replay.Renderers.IChannelRenderer.finish(ax, ctx.Theme, true);
            if D <= 8
                lg = legend(ax, compose("d = %d", 1:D), 'Location', 'best');
                OceanViz.App.Replay.Renderers.IChannelRenderer.styleLegend(lg, ctx.Theme);
            end
        end
    end
end
