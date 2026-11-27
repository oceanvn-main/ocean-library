classdef GridScene < handle
    % GRIDSCENE  Retained 2D scene of a grid world on one axes: static terrain and grid lines are drawn once;
    %   every update only rewrites the XData/YData of ONE batched agent scatter (and one selection marker), so the
    %   graphics object count is constant whatever the number of agents (GFX1, basic profile).
    %
    %   s = OceanViz.Scene.GridScene(ax, terrain, ids, 'Theme', theme);
    %   s.update(positions)        positions: N x 2 cells (row, col), row order = ids
    %   s.select("agent2")         selection is by agent identity, not array position
    %   id = s.idAt([row col])     nearest agent to a cell coordinate (hit test)
    %   n = s.NumGraphicsObjects   created by the scene; independent of N
    % Terrain classes: 0 free, 1 cliff, 2 terminal. Reads only what it is given; computes nothing about the world.

    properties (SetAccess = private)
        Axes
        Ids (1,:) string
        Positions double = zeros(0, 2)
        Selected (1,1) string = ""
        Dims (1,2) double
    end

    properties (Access = private)
        Image_
        GridLines_
        Agents_
        Marker_
    end

    properties (Dependent)
        NumGraphicsObjects
    end

    methods
        function obj = GridScene(ax, terrain, ids, opts)
            arguments
                ax (1,1)
                terrain double
                ids (1,:) string
                opts.Theme = []
            end
            obj.Axes = ax; obj.Ids = ids; obj.Dims = size(terrain);
            T = opts.Theme; if isempty(T), T = OceanViz.Core('Theme', 'Dark').Colors; end
            OceanViz.App.Replay.Renderers.IChannelRenderer.prepare(ax);
            [R, C] = size(terrain);
            cmap = [T.Background; 0.80 0.35 0.30; 0.35 0.75 0.45];             % free / cliff / terminal
            rgb = reshape(cmap(terrain + 1, :), R, C, 3);
            hold(ax, 'on');
            obj.Image_ = image(ax, [1 C], [1 R], rgb);
            gx = [repmat((0.5:1:C + 0.5), 2, 1); nan(1, C + 1)]; gy = repmat([0.5; R + 0.5; nan], 1, C + 1);
            hx = [repmat([0.5; C + 0.5; nan], 1, R + 1)]; hy = [repmat((0.5:1:R + 0.5), 2, 1); nan(1, R + 1)];
            obj.GridLines_ = plot(ax, [gx(:); hx(:)], [gy(:); hy(:)], '-', 'Color', T.Grid, 'LineWidth', 0.5);
            n = numel(ids); col = lines(max(n, 1));
            obj.Agents_ = scatter(ax, nan(1, n), nan(1, n), 90, col(1:n, :), 'filled', 'MarkerEdgeColor', T.Text);
            obj.Marker_ = plot(ax, nan, nan, 'o', 'MarkerSize', 16, 'LineWidth', 2, 'Color', T.Secondary);
            hold(ax, 'off');
            axis(ax, 'equal'); axis(ax, 'ij'); xlim(ax, [0.5 C + 0.5]); ylim(ax, [0.5 R + 0.5]);
            OceanViz.App.Replay.Renderers.IChannelRenderer.finish(ax, T, false);
        end

        function n = get.NumGraphicsObjects(~)
            n = 4;                          % terrain image, grid lines, agent scatter, selection marker
        end

        function tf = isLive(obj)
            tf = isvalid(obj.Axes) && isvalid(obj.Agents_) && isvalid(obj.Image_) && obj.Image_.Parent == obj.Axes;
        end

        function update(obj, positions)
            arguments
                obj
                positions (:,2) double
            end
            if size(positions, 1) ~= numel(obj.Ids)
                error('OceanViz:Scene:badPositions', 'Expected %d agent positions, got %d.', numel(obj.Ids), size(positions, 1));
            end
            obj.Positions = positions;
            obj.Agents_.XData = positions(:, 2).'; obj.Agents_.YData = positions(:, 1).';
            obj.refreshMarker_();
        end

        function select(obj, id)
            arguments
                obj
                id (1,1) string
            end
            if id ~= "" && ~any(obj.Ids == id)
                error('OceanViz:Scene:unknownId', 'No agent "%s" in the scene.', id);
            end
            obj.Selected = id; obj.refreshMarker_();
        end

        function id = idAt(obj, cell)
            % The agent nearest to cell (row, col); "" when the scene has no positions yet.
            id = "";
            if isempty(obj.Positions), return; end
            [~, k] = min(sum((obj.Positions - reshape(cell, 1, 2)).^2, 2));
            id = obj.Ids(k);
        end
    end

    methods (Access = private)
        function refreshMarker_(obj)
            k = find(obj.Ids == obj.Selected, 1);
            if isempty(k) || isempty(obj.Positions)
                obj.Marker_.XData = nan; obj.Marker_.YData = nan;
            else
                obj.Marker_.XData = obj.Positions(k, 2); obj.Marker_.YData = obj.Positions(k, 1);
            end
        end
    end
end
