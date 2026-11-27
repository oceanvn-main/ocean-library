classdef DefaultLayout < OceanViz.App.Replay.Layouts.IReplayStudioLayout
    % DEFAULTLAYOUT  Works for any recording: one tile per non-scalar channel
    %   and per view (drawn by its kind's renderer); scalar channels are listed
    %   with their current values (their history is on the timeline).

    methods
        function n = name(~), n = "Default"; end

        function tf = appliesTo(~, ~), tf = true; end

        function tiles = tiles(~, rec)
            tiles = OceanViz.App.Replay.Layouts.IReplayStudioLayout.tile("", strings(1, 0)); tiles(1) = [];
            for c = rec.Channels
                if c.Kind ~= "scalar", tiles(end+1) = OceanViz.App.Replay.Layouts.IReplayStudioLayout.tile(c.Name, c.Name); end %#ok<AGROW>
            end
            for v = rec.Views
                tiles(end+1) = OceanViz.App.Replay.Layouts.IReplayStudioLayout.tile(v.Name, v.Name); %#ok<AGROW>
            end
        end

        function lines = notes(~, rec, coords)
            s = OceanReplay.Core.Recording.namesOf(rec.Channels(arrayfun(@(c) c.Kind == "scalar", rec.Channels)));
            lines = [{'CURRENT VALUES (nearest stored point; never blended)'}, ...
                OceanViz.App.Replay.Layouts.IReplayStudioLayout.scalarLines(rec, coords, s)];
        end
    end
end
