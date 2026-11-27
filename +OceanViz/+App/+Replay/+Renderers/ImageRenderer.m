classdef ImageRenderer < OceanViz.App.Replay.Renderers.IChannelRenderer
    % IMAGERENDERER  An image (H x W x C): values mapped from the channel's
    %   DisplayRange [lo hi] to black..white (clipped). C = 1 grey with a colour
    %   bar labelled in DATA units (the notation legend), C = 3 RGB, other C: the
    %   channels side by side. Consecutive frames of the same size only replace
    %   the pixels (fast playback).

    methods
        function tf = canRender(~, info)
            tf = info.Kind == "image";
        end

        function draw(~, ax, value, info, ctx)
            img = OceanViz.App.Replay.Renderers.ImageRenderer.toUnit(value, info.DisplayRange);
            C = size(img, 3);
            if C ~= 1 && C ~= 3
                img = OceanViz.App.Common.ImageView.tile(reshape(img, size(img, 1), size(img, 2), 1, C), C);
            end
            h = findobj(ax, 'Type', 'image');
            if isscalar(h) && isequal(size(h.CData), size(img)) && strcmp(ax.Tag, 'replay-image')
                h.CData = img;                                         % same frame size: update in place
                return;
            end
            OceanViz.App.Replay.Renderers.IChannelRenderer.prepare(ax);
            if size(img, 3) == 1
                image(ax, img, 'CDataMapping', 'scaled'); colormap(ax, gray(256)); ax.CLim = [0 1];
            else
                image(ax, img);
            end
            axis(ax, 'image'); ax.YDir = 'reverse';
            OceanViz.App.Replay.Renderers.IChannelRenderer.finish(ax, ctx.Theme, false);
            if size(img, 3) == 1                                       % legend: grey level -> data value
                r = info.DisplayRange;
                cb = colorbar(ax, 'Color', ctx.Theme.Text, 'Ticks', [0 0.5 1], ...
                    'TickLabels', {sprintf('%g', r(1)), sprintf('%g', mean(r)), sprintf('%g', r(2))});
                cb.FontSize = 8;
            end
            ax.Tag = 'replay-image';
        end
    end

    methods (Static)
        function u = toUnit(value, range)
            % [lo hi] -> [0 1], clipped (the stored numbers are untouched).
            lo = range(1); hi = range(2);
            u = min(1, max(0, (double(value) - lo) / max(hi - lo, eps)));
        end
    end
end
