classdef QuadTree < handle
    % QUADTREE  2D spatial index using recursive quadrant subdivision.
    %
    %   Partitions a 2D plane into a tree of rectangular cells so that
    %   nearest-neighbor and range queries run in O(log N) average time
    %   instead of O(N) brute-force scanning.
    %
    %   Usage (bulk-build):
    %     positions = rand(1000, 2) * 100;       % Nx2
    %     ids       = (1:1000)';                  % Nx1
    %     qt = OceanMath.Geometry.QuadTree(positions, ids);
    %
    %   Nearest-neighbor:
    %     nearId = qt.findNearest([50, 50], 10);  % within radius 10
    %
    %   Range query:
    %     ids = qt.rangeQuery([50, 50], 20);      % all within radius 20
    %
    %   Bounding-box query (for curve segments):
    %     ids = qt.queryBox([xMin, yMin, xMax, yMax]);
    %
    % See also: OceanViz.Designer.Interaction.InteractionHandler,
    %           OceanViz.Designer.TrackRenderer

    properties (SetAccess = private)
        Root_           % Root node (struct)
        Count_   (1,1) double = 0   % Total points stored
        Bounds_  (1,4) double       % [xMin, yMin, xMax, yMax]
    end

    properties (Constant)
        MAX_LEAF_SIZE = 8;   % Max points per leaf before split
        MAX_DEPTH     = 20;  % Safety cap to prevent infinite recursion
    end

    methods

        function obj = QuadTree(positions, ids)
            % QUADTREE  Construct from Nx2 positions and Nx1 integer IDs.
            %
            %   qt = OceanMath.Geometry.QuadTree(positions, ids)
            %
            %   positions: Nx2 matrix of [X, Y] coordinates
            %   ids:       Nx1 vector of unique point identifiers
            arguments
                positions (:,2) double
                ids       (:,1) double
            end

            obj.Count_ = size(positions, 1);

            if obj.Count_ == 0
                obj.Bounds_ = [0 0 1 1];
                obj.Root_ = obj.makeLeaf_([0 0 1 1], zeros(0,2), zeros(0,1));
                return;
            end

            % Compute tight bounding box with small epsilon padding
            xMin = min(positions(:,1));  xMax = max(positions(:,1));
            yMin = min(positions(:,2));  yMax = max(positions(:,2));
            pad = max(1e-6, max(xMax - xMin, yMax - yMin) * 0.001);
            obj.Bounds_ = [xMin - pad, yMin - pad, xMax + pad, yMax + pad];

            % Recursively build the tree
            obj.Root_ = obj.buildNode_(obj.Bounds_, positions, ids, 0);
        end

        % ============================================================== %
        %  Nearest-Neighbor Query
        % ============================================================== %

        function [nearId, nearDist] = findNearest(obj, queryPt, threshold)
            % FINDNEAREST  Return the ID of the closest point within threshold.
            %
            %   nearId = qt.findNearest([x, y], radius)
            %   Returns 0 if no point lies within threshold.
            arguments
                obj
                queryPt   (1,2) double
                threshold (1,1) double = inf
            end

            if obj.Count_ == 0
                nearId = 0; nearDist = inf; return;
            end

            % Use a mutable struct to track best candidate across recursion
            best = struct('id', 0, 'dist', threshold);
            best = obj.searchNearest_(obj.Root_, queryPt, best);
            nearId   = best.id;
            nearDist = best.dist;
        end

        % ============================================================== %
        %  Range Query (all points within radius)
        % ============================================================== %

        function ids = rangeQuery(obj, center, radius)
            % RANGEQUERY  Return IDs of all points within radius of center.
            %
            %   ids = qt.rangeQuery([x, y], radius)
            arguments
                obj
                center (1,2) double
                radius (1,1) double
            end

            if obj.Count_ == 0
                ids = []; return;
            end

            ids = obj.searchRange_(obj.Root_, center, radius);
        end

        % ============================================================== %
        %  Box Query (all points inside axis-aligned rectangle)
        % ============================================================== %

        function ids = queryBox(obj, box)
            % QUERYBOX  Return IDs of all points inside an axis-aligned box.
            %
            %   ids = qt.queryBox([xMin, yMin, xMax, yMax])
            arguments
                obj
                box (1,4) double   % [xMin, yMin, xMax, yMax]
            end

            if obj.Count_ == 0
                ids = []; return;
            end

            ids = obj.searchBox_(obj.Root_, box);
        end

        % ============================================================== %
        %  Diagnostics
        % ============================================================== %

        function describe(obj)
            % DESCRIBE  Print tree statistics.
            [leafCount, maxDepth, totalPts] = obj.treeStats_(obj.Root_, 0);
            fprintf('QuadTree:\n');
            fprintf('  Points:    %d\n', totalPts);
            fprintf('  Leaves:    %d\n', leafCount);
            fprintf('  Max depth: %d\n', maxDepth);
            fprintf('  Bounds:    [%.2f, %.2f] to [%.2f, %.2f]\n', ...
                obj.Bounds_(1), obj.Bounds_(2), obj.Bounds_(3), obj.Bounds_(4));
        end

    end

    % ================================================================== %
    %  PRIVATE — Tree Construction
    % ================================================================== %

    methods (Access = private)

        function node = buildNode_(obj, bounds, positions, ids, depth)
            % BUILDNODE_  Recursively partition space into quadrants.
            n = size(positions, 1);

            % Base cases: make a leaf
            if n <= obj.MAX_LEAF_SIZE || depth >= obj.MAX_DEPTH
                node = obj.makeLeaf_(bounds, positions, ids);
                return;
            end

            % Compute midpoint of current bounds
            xMid = (bounds(1) + bounds(3)) / 2;
            yMid = (bounds(2) + bounds(4)) / 2;

            % Classify each point into one of 4 quadrants
            isRight = positions(:,1) >= xMid;
            isTop   = positions(:,2) >= yMid;

            % SW = ~Right & ~Top,  SE = Right & ~Top
            % NW = ~Right & Top,   NE = Right & Top
            maskSW = ~isRight & ~isTop;
            maskSE =  isRight & ~isTop;
            maskNW = ~isRight &  isTop;
            maskNE =  isRight &  isTop;

            % Build child bounds
            bSW = [bounds(1), bounds(2), xMid,     yMid];
            bSE = [xMid,     bounds(2), bounds(3), yMid];
            bNW = [bounds(1), yMid,     xMid,     bounds(4)];
            bNE = [xMid,     yMid,     bounds(3), bounds(4)];

            % Recursively build children
            node = struct( ...
                'isLeaf', false, ...
                'bounds', bounds, ...
                'positions', [], ...
                'ids', [], ...
                'SW', obj.buildNode_(bSW, positions(maskSW,:), ids(maskSW), depth+1), ...
                'SE', obj.buildNode_(bSE, positions(maskSE,:), ids(maskSE), depth+1), ...
                'NW', obj.buildNode_(bNW, positions(maskNW,:), ids(maskNW), depth+1), ...
                'NE', obj.buildNode_(bNE, positions(maskNE,:), ids(maskNE), depth+1) );
        end

        function node = makeLeaf_(~, bounds, positions, ids)
            % MAKELEAF_  Create a terminal leaf node storing raw points.
            node = struct( ...
                'isLeaf', true, ...
                'bounds', bounds, ...
                'positions', positions, ...
                'ids', ids, ...
                'SW', [], 'SE', [], 'NW', [], 'NE', []);
        end

    end

    % ================================================================== %
    %  PRIVATE — Nearest-Neighbor Search
    % ================================================================== %

    methods (Access = private)

        function best = searchNearest_(obj, node, queryPt, best)
            % SEARCHNEAREST_  Recursively prune and search for closest point.

            % PRUNE: If the closest possible point in this cell's bounding
            % box is already farther than our current best, skip entirely.
            if obj.boxDist_(node.bounds, queryPt) >= best.dist
                return;
            end

            if node.isLeaf
                % Brute-force scan the (small) leaf
                for k = 1:size(node.positions, 1)
                    d = sqrt(sum((node.positions(k,:) - queryPt).^2));
                    if d < best.dist
                        best.dist = d;
                        best.id   = node.ids(k);
                    end
                end
                return;
            end

            % Visit children ordered by distance to query point
            % (nearest child first for faster pruning)
            children = {node.SW, node.SE, node.NW, node.NE};
            childDists = zeros(1, 4);
            for c = 1:4
                childDists(c) = obj.boxDist_(children{c}.bounds, queryPt);
            end
            [~, order] = sort(childDists);

            for c = order
                best = obj.searchNearest_(children{c}, queryPt, best);
            end
        end

    end

    % ================================================================== %
    %  PRIVATE — Range Search
    % ================================================================== %

    methods (Access = private)

        function ids = searchRange_(obj, node, center, radius)
            % SEARCHRANGE_  Collect all points within radius of center.

            % PRUNE: If the entire bounding box is outside the search circle
            if obj.boxDist_(node.bounds, center) >= radius
                ids = []; return;
            end

            if node.isLeaf
                % Brute-force scan the (small) leaf
                if isempty(node.positions)
                    ids = []; return;
                end
                dists = sqrt(sum((node.positions - center).^2, 2));
                ids = node.ids(dists < radius);
                return;
            end

            % Recurse into all children
            ids = [ ...
                obj.searchRange_(node.SW, center, radius); ...
                obj.searchRange_(node.SE, center, radius); ...
                obj.searchRange_(node.NW, center, radius); ...
                obj.searchRange_(node.NE, center, radius) ];
        end

    end

    % ================================================================== %
    %  PRIVATE — Box Query
    % ================================================================== %

    methods (Access = private)

        function ids = searchBox_(obj, node, box)
            % SEARCHBOX_  Collect all points inside an axis-aligned box.

            % PRUNE: If the bounding box doesn't overlap the query box
            if ~obj.boxesOverlap_(node.bounds, box)
                ids = []; return;
            end

            if node.isLeaf
                if isempty(node.positions)
                    ids = []; return;
                end
                inside = node.positions(:,1) >= box(1) & ...
                         node.positions(:,1) <= box(3) & ...
                         node.positions(:,2) >= box(2) & ...
                         node.positions(:,2) <= box(4);
                ids = node.ids(inside);
                return;
            end

            ids = [ ...
                obj.searchBox_(node.SW, box); ...
                obj.searchBox_(node.SE, box); ...
                obj.searchBox_(node.NW, box); ...
                obj.searchBox_(node.NE, box) ];
        end

    end

    % ================================================================== %
    %  PRIVATE — Geometry Helpers
    % ================================================================== %

    methods (Access = private, Static)

        function d = boxDist_(bounds, pt)
            % BOXDIST_  Minimum distance from a point to an AABB.
            %   Returns 0 if the point is inside the box.
            dx = max([bounds(1) - pt(1), 0, pt(1) - bounds(3)]);
            dy = max([bounds(2) - pt(2), 0, pt(2) - bounds(4)]);
            d = sqrt(dx*dx + dy*dy);
        end

        function tf = boxesOverlap_(a, b)
            % BOXESOVERLAP_  Test if two AABBs overlap.
            tf = a(1) <= b(3) && a(3) >= b(1) && ...
                 a(2) <= b(4) && a(4) >= b(2);
        end

    end

    % ================================================================== %
    %  PRIVATE — Diagnostics
    % ================================================================== %

    methods (Access = private)

        function [leafCount, maxDepth, totalPts] = treeStats_(obj, node, depth)
            if node.isLeaf
                leafCount = 1;
                maxDepth  = depth;
                totalPts  = size(node.positions, 1);
                return;
            end

            [lSW, dSW, pSW] = obj.treeStats_(node.SW, depth+1);
            [lSE, dSE, pSE] = obj.treeStats_(node.SE, depth+1);
            [lNW, dNW, pNW] = obj.treeStats_(node.NW, depth+1);
            [lNE, dNE, pNE] = obj.treeStats_(node.NE, depth+1);

            leafCount = lSW + lSE + lNW + lNE;
            maxDepth  = max([dSW, dSE, dNW, dNE]);
            totalPts  = pSW + pSE + pNW + pNE;
        end

    end
end
