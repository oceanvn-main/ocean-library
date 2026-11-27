classdef PointsRenderer < OceanViz.App.Replay.Renderers.IChannelRenderer
    % POINTSRENDERER  N x 2 (or N x 3) points as a scatter. When the channel
    %   names a LabelsChannel (e.g. data -> assignment), points are coloured by
    %   that channel's NEAREST STORED value at the current coordinates, with a
    %   legend of its categories.

    methods
        function tf = canRender(~, info)
            tf = info.Kind == "points";
        end

        function draw(~, ax, value, info, ctx)
            OceanViz.App.Replay.Renderers.IChannelRenderer.prepare(ax);
            v = double(value); T = ctx.Theme;
            lab = []; names = strings(1, 0);
            if info.LabelsChannel ~= "" && any(ctx.Recording.names() == info.LabelsChannel)
                li = ctx.Recording.info(info.LabelsChannel);
                lab = double(ctx.Recording.get(info.LabelsChannel, OceanViz.App.Replay.Renderers.PointsRenderer.coordsFor(li, ctx.Coords)));
                names = li.Categories;
            end
            hold(ax, 'on');
            if isempty(lab)
                OceanViz.App.Replay.Renderers.PointsRenderer.scatterAny(ax, v, T.Primary, info.Name);
            else
                k = unique(lab).'; col = lines(max([k numel(names)]));
                for c = k
                    nm = "category " + c; if c <= numel(names), nm = names(c); end
                    OceanViz.App.Replay.Renderers.PointsRenderer.scatterAny(ax, v(lab == c, :), col(c, :), nm);
                end
            end
            hold(ax, 'off'); grid(ax, 'on'); axis(ax, 'equal');
            OceanViz.App.Replay.Renderers.IChannelRenderer.finish(ax, T, true);
            if ~isempty(lab) && numel(unique(lab)) <= 12
                OceanViz.App.Replay.Renderers.IChannelRenderer.styleLegend(legend(ax, 'Location', 'best'), T);
            end
        end
    end

    methods (Static)
        function c = coordsFor(info, coords)
            % The fields of coords that a channel depends on (struct()).
            c = struct();
            for a = info.Axes
                if isfield(coords, a), c.(a) = coords.(a); end
            end
        end

        function h = scatterAny(ax, v, color, name)
            if size(v, 2) == 3
                h = scatter3(ax, v(:, 1), v(:, 2), v(:, 3), 14, color, 'filled', 'DisplayName', char(name));
            else
                h = scatter(ax, v(:, 1), v(:, 2), 14, color, 'filled', 'DisplayName', char(name));
            end
        end
    end
end
