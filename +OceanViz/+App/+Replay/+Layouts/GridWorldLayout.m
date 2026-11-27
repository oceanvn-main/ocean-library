classdef GridWorldLayout < OceanViz.App.Replay.Layouts.IReplayStudioLayout
    % GRIDWORLDLAYOUT  A GridSceneRecordable recording: the grid with its agents at the stored commit (one retained
    %   OceanViz.Scene.GridScene, updated in place), plus the commit's world time and episode. Positions are the NEAREST
    %   STORED commit's values (exact inspection); nothing is interpolated.

    properties (Access = private)
        Scene_ = []
    end

    methods
        function n = name(~), n = "GridWorld"; end

        function n = timeline(~, rec)
            n = ""; if any(rec.names() == "tick"), n = "tick"; end
        end

        function tf = appliesTo(~, rec)
            n = rec.names();
            tf = all(ismember(["position" "roster" "terrain"], n)) && rec.info("position").Kind == "points";
        end

        function tiles = tiles(obj, ~)
            tiles = OceanViz.App.Replay.Layouts.IReplayStudioLayout.tile("Grid world: agents at the committed frame", ...
                ["position" "terrain" "roster"], @obj.drawScene);
        end

        function lines = notes(~, rec, coords)
            lines = [{'GRID WORLD (committed frames; nearest stored commit, never blended)'}, ...
                OceanViz.App.Replay.Layouts.IReplayStudioLayout.scalarLines(rec, coords, ...
                intersect(["tick" "episode"], rec.names(), 'stable'))];
        end

        function caption = drawScene(obj, ax, studio)
            rec = studio.Recording;
            c = struct('commit', studio.Coords.commit);
            [P, at] = rec.get("position", c);
            if isempty(obj.Scene_) || ~obj.Scene_.isLive() || obj.Scene_.Axes ~= ax
                ids = string(rec.get("roster")); ids = reshape(ids, 1, []);
                obj.Scene_ = OceanViz.Scene.GridScene(ax, double(rec.get("terrain")), ids, 'Theme', studio.Theme);
            end
            obj.Scene_.update(double(P));
            caption = sprintf('stored at commit = %g', at.commit);
        end
    end
end
