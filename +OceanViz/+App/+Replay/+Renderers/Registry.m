classdef Registry
    % REGISTRY  The renderer for a channel, chosen by its kind. The only place
    %   that maps kinds to drawing code (architecture guard: no kind switches
    %   elsewhere in Replay Studio).
    %
    %   r = OceanViz.App.Replay.Renderers.Registry.forInfo(info);
    %   r.draw(ax, value, info, ctx);

    methods (Static)
        function r = forInfo(info)
            persistent renderers
            if isempty(renderers)
                R = 'OceanViz.App.Replay.Renderers.';
                renderers = cellfun(@(n) feval([R n]), {'ScalarRenderer', 'VectorRenderer', 'MatrixRenderer', ...
                    'ImageRenderer', 'SequenceRenderer', 'PointsRenderer', 'TokensRenderer'}, 'UniformOutput', false);
            end
            for i = 1:numel(renderers)
                if renderers{i}.canRender(info), r = renderers{i}; return; end
            end
            error('OceanViz:Replay:noRenderer', 'No renderer for kind "%s".', info.Kind);
        end
    end
end
