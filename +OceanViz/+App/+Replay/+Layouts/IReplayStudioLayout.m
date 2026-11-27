classdef (Abstract) IReplayStudioLayout < handle
    % IREPLAYSTUDIOLAYOUT  How Replay Studio arranges a recording: which tiles
    %   (and in which order), and extra explanatory text for the current point.
    %   DefaultLayout works for ANY recording; domain layouts (Diffusion,
    %   Clustering) add the domain's equations and composite plots.
    %
    %   n     = name()                     shown in the Layout dropdown
    %   tf    = appliesTo(rec)             OceanReplay.Core.Recording
    %   tiles = tiles(rec)                 struct array: Title, Names (channels shown),
    %                                      Draw ([] = the channel's renderer, or
    %                                      @(ax, studio) -> caption for a composite tile)
    %   lines = notes(rec, coords)         cellstr for the info panel
    %   n     = timeline(rec)              optional: the scalar channel the timeline shows
    %                                      first ("" = the first one available)

    methods (Abstract)
        n = name(obj)
        tf = appliesTo(obj, rec)
        tiles = tiles(obj, rec)
        lines = notes(obj, rec, coords)
    end

    methods
        function n = timeline(~, ~)
            n = "";
        end
    end

    methods (Static)
        function t = tile(title, names, draw)
            if nargin < 3, draw = []; end
            t = struct('Title', string(title), 'Names', string(names), 'Draw', {draw});
        end

        function lines = scalarLines(rec, coords, names)
            % "symbol = value   (stored at axis = v)" for scalar channels (nearest stored point).
            lines = {};
            for n = names
                info = rec.info(n);
                c = OceanViz.App.Replay.Renderers.PointsRenderer.coordsFor(info, coords);
                if numel(fieldnames(c)) ~= numel(info.Axes), continue; end
                [v, at] = rec.get(n, c);
                lines{end+1} = sprintf('%s = %.6g   %s', info.Symbol, double(v), ...
                    OceanViz.App.Replay.Layouts.IReplayStudioLayout.where(at)); %#ok<AGROW>
            end
        end

        function s = where(at)
            f = fieldnames(at);
            parts = strings(1, numel(f));
            for i = 1:numel(f), parts(i) = sprintf('%s = %g', f{i}, at.(f{i})); end
            s = char("(stored at " + strjoin(parts, ", ") + ")");
            if isempty(f), s = '(no axis)'; end
        end
    end
end
