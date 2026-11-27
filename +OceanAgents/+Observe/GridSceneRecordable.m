classdef GridSceneRecordable < OceanReplay.Record.IRecordable
    % GRIDSCENERECORDABLE  Domain view adapter: records the committed scene of a grid AgentWorld into a Replay file
    %   (Graphics_Rendering_and_Replay_Architecture.md, GFX1). It never steps the world itself: the caller's Drive
    %   function runs the episode(s) and calls commit() at every committed boundary (after reset and after each step).
    %
    %   rec = OceanAgents.Observe.GridSceneRecordable(aw, @drive, 'Terrain', terrain);
    %   OceanReplay.Record.Recorder.run(rec, rec.defaultSpec(), "scene.h5");
    %   function out = drive(aw, commit)       % any loop: reset, step ... ; call commit() after each
    %
    % Axis  commit  : 1, 2, ... one value per committed frame, across episodes (open axis).
    % Channels (all raw committed state, copied at the commit so later steps cannot alter them):
    %   position (points, N x 2, cell (row, col))   tick (scalar, world time)   episode (scalar)
    %   ended (vector, N, 0/1)                      terrain (matrix R x C, static)   roster (tokens, static)
    % Privileged research observer: positions come from AgentWorld.researchView, not from an agent's view.

    properties (SetAccess = private)
        World
        Drive
        Terrain double
        Component (1,1) string = "Position"
        NumCommits (1,1) double = 0
    end

    properties (Access = private)
        Recorder_ = []
    end

    methods
        function obj = GridSceneRecordable(world, drive, opts)
            arguments
                world (1,1) OceanAgents.World.AgentWorld
                drive (1,1) function_handle
                opts.Terrain double = []
                opts.Component (1,1) string = "Position"
            end
            obj.World = world; obj.Drive = drive; obj.Terrain = opts.Terrain; obj.Component = opts.Component;
        end

        function axes = axesInfo(~)
            axes = OceanReplay.Core.Axis("commit", [], 'Symbol', "c", 'Description', "committed frame counter (all episodes)");
        end

        function [channels, views] = catalog(obj)
            C = @OceanReplay.Core.ChannelInfo;
            n = numel(obj.World.Roster);
            channels = [C("position", "points", 'Axes', "commit", 'ValueShape', [n 2], 'Units', "cell (row, col)", ...
                    'Description', "agent cells at the committed boundary"), ...
                C("tick", "scalar", 'Axes', "commit", 'Symbol', "t", 'Description', "world time of the commit"), ...
                C("episode", "scalar", 'Axes', "commit", 'Symbol', "e", 'Description', "episode number of the commit"), ...
                C("ended", "vector", 'Axes', "commit", 'ValueShape', n, 'Description', "1 = the agent's episode has ended"), ...
                C("roster", "tokens", 'ValueShape', n, 'Description', "agent IDs, in the row order of position")];
            if ~isempty(obj.Terrain)
                channels(end+1) = C("terrain", "matrix", 'ValueShape', size(obj.Terrain), ...
                    'Description', "cell classes of the grid (0 free, 1 cliff, 2 terminal)");
            end
            views = OceanReplay.Core.ViewInfo.empty;
        end

        function spec = defaultSpec(obj)
            S = OceanReplay.Core.Sampling;
            spec = OceanReplay.Core.Spec().addGroup("scene", 0, ["position" "tick" "episode" "ended"], struct('commit', S.all()));
            static = "roster"; if ~isempty(obj.Terrain), static(end+1) = "terrain"; end
            spec = spec.addGroup("static", 0, static, struct());
        end

        function out = run(obj, recorder)
            obj.Recorder_ = recorder; obj.NumCommits = 0;
            cleanup = onCleanup(@() obj.detach_());
            out = obj.Drive(obj.World, @() obj.commit());
        end

        function commit(obj)
            % Record the world's committed state now. Values are copied here, before any lazy provider runs.
            if isempty(obj.Recorder_), return; end
            v = obj.World.researchView(obj.Component);
            obj.NumCommits = obj.NumCommits + 1;
            state = struct('position', v.Value, 'tick', v.Tick, 'episode', v.EpisodeID, 'ended', double(v.Ended), ...
                'roster', v.Roster);
            if ~isempty(obj.Terrain), state.terrain = obj.Terrain; end
            state = structfun(@(x) {x}, state, 'UniformOutput', false);      % freeze each value for its provider
            f = fieldnames(state);
            for i = 1:numel(f), x = state.(f{i}){1}; state.(f{i}) = @() x; end
            obj.Recorder_.point(struct('commit', obj.NumCommits), state);
        end

        function d = describe(obj)
            d = struct('model', struct('name', "GridScene", 'world', obj.World.Name, 'grid', size(obj.Terrain)), ...
                'references', {{"Planning/System/Game Theory/Graphics_Rendering_and_Replay_Architecture.md"}}, ...
                'notation', struct('commit', "committed frame counter", 'tick', "world time"), 'layout', "GridWorld");
        end
    end

    methods (Access = private)
        function detach_(obj)
            obj.Recorder_ = [];
        end
    end
end
