classdef KDTree < handle
    % KDTREE Generic N-dimensional KD-Tree for nearest-neighbour queries.
    %
    % OVERVIEW:
    %   Builds a balanced binary KD-Tree by recursively splitting on the
    %   dimension with the largest range (max-spread heuristic). Supports
    %   k-NN search and radius range queries in O(log N) average time for
    %   low-to-moderate dimensionality.
    %
    %   Metrics supported: 'euclidean' | 'manhattan' | 'chebyshev'
    %   Custom metrics (function handle) are accepted but disable AABB pruning.
    %
    % USAGE:
    %   kd = OceanAlgo.Spatial.KDTree(X);               % default: euclidean
    %   kd = OceanAlgo.Spatial.KDTree(X, 'Metric', 'manhattan', 'LeafSize', 20);
    %
    %   [ids, dists] = kd.findKNearest(q, 5);           % 5-NN of row vector q
    %   [ids, dists] = kd.rangeQuery(q, 3.0);           % all within radius 3
    %   kd.describe();
    %
    % REFERENCE:
    %   Bentley (1975). Multidimensional binary search trees used for
    %   associative searching. CACM, 18(9), 509-517.
    %
    % Author: Ocean Library
    % Date:   2026-05-12

    % =====================================================================
    properties (SetAccess = private)
        Points_   double    % (N x d) stored points
        Root_               % root node struct
        N_  (1,1) double    % total points
        D_  (1,1) double    % dimensionality
        Metric_ string      % resolved metric name
        DistFn_             % @(row_a, row_b) -> scalar
    end

    properties
        LeafSize (1,1) double = 10   % max points per leaf before splitting
        MaxDepth (1,1) double = 64   % recursion safety cap
    end

    % =====================================================================
    methods

        function obj = KDTree(points, options)
            % KDTREE Constructor.
            %
            % Args:
            %   points   (N x d double): Point set to index.
            %   Metric   (string|fhandle): Distance metric. Default 'euclidean'.
            %   LeafSize (double): Max points per leaf. Default 10.
            %   MaxDepth (double): Max recursion depth. Default 64.
            arguments
                points          (:,:) double
                options.Metric              = 'euclidean'
                options.LeafSize (1,1) double = 10
                options.MaxDepth (1,1) double = 64
            end

            [obj.N_, obj.D_] = size(points);
            obj.Points_   = points;
            obj.LeafSize  = options.LeafSize;
            obj.MaxDepth  = options.MaxDepth;
            obj.Metric_   = obj.resolveMetricName_(options.Metric);
            obj.DistFn_   = obj.makeDistFn_(options.Metric);

            if obj.N_ == 0
                obj.Root_ = [];
                return;
            end
            obj.Root_ = obj.buildNode_((1:obj.N_)', 0);
        end

        % -----------------------------------------------------------------
        function [ids, dists] = findKNearest(obj, query, k)
            % FINDKNEAREST Return the k nearest neighbours of query point.
            %
            % Args:
            %   query (1 x d double): Query point.
            %   k     (double):       Number of neighbours. Default 1.
            %
            % Returns:
            %   ids   (k x 1): Row indices into the original point set.
            %   dists (k x 1): Corresponding distances (ascending order).
            %         Padded with 0/Inf if N < k.
            arguments
                obj
                query (1,:) double
                k     (1,1) double = 1
            end
            obj.assertDim_(query);
            kHeap = min(k, obj.N_);               % cannot find more than N points
            heap = struct('dists', inf(1,kHeap), 'ids', zeros(1,kHeap), ...
                          'k', kHeap, 'count', 0);
            heap = obj.searchKNN_(obj.Root_, query, heap);

            n = heap.count;
            [sd, ord] = sort(heap.dists(1:n));
            ids   = heap.ids(ord)';
            dists = sd';
            if n < k                              % pad if fewer than k
                ids(end+1:k)   = 0;
                dists(end+1:k) = inf;
            end
        end

        % -----------------------------------------------------------------
        function [ids, dists] = rangeQuery(obj, query, radius)
            % RANGEQUERY Return all points within radius of query.
            %
            % Args:
            %   query  (1 x d double): Centre point.
            %   radius (double):       Search radius.
            %
            % Returns:
            %   ids   (M x 1): Row indices of qualifying points.
            %   dists (M x 1): Their distances to query (unsorted).
            arguments
                obj
                query  (1,:) double
                radius (1,1) double
            end
            obj.assertDim_(query);
            [ids, dists] = obj.searchRange_(obj.Root_, query, radius);
        end

        % -----------------------------------------------------------------
        function describe(obj)
            % DESCRIBE Print tree diagnostics.
            if isempty(obj.Root_)
                fprintf('KDTree: empty (N=0)\n'); return;
            end
            [nL, maxD, ~] = obj.treeStats_(obj.Root_, 0);
            fprintf('KDTree:\n');
            fprintf('  Points    : %d\n', obj.N_);
            fprintf('  Dims      : %d\n', obj.D_);
            fprintf('  Metric    : %s\n', obj.Metric_);
            fprintf('  Leaf size : %d\n', obj.LeafSize);
            fprintf('  Leaves    : %d\n', nL);
            fprintf('  Max depth : %d\n', maxD);
        end

    end % public methods

    % =====================================================================
    %  Private — Build
    % =====================================================================
    methods (Access = private)

        function node = buildNode_(obj, idx, depth)
            % BUILDNODE_  Recursively build a KD-Tree node.
            n   = numel(idx);
            pts = obj.Points_(idx, :);
            lo  = min(pts, [], 1);
            hi  = max(pts, [], 1);
            bounds = [lo; hi];   % 2 x d

            if n <= obj.LeafSize || depth >= obj.MaxDepth
                node = obj.leaf_(bounds, idx);
                return;
            end

            % Split on dimension with largest range
            [~, dim] = max(hi - lo);
            col = pts(:, dim);

            % Guard: all values identical on split dim → make leaf
            if col(1) == col(end) && min(col) == max(col)
                node = obj.leaf_(bounds, idx); %All belongs to the same block
                return;
            end

            med       = median(col);
            leftMask  = col <= med;
            rightMask = ~leftMask;

            % Guard: all points on one side (degenerate split)
            if all(leftMask) || all(rightMask)
                node = obj.leaf_(bounds, idx);
                return;
            end

            node = struct('isLeaf', false, 'bounds', bounds, ...
                'idx', [], 'dim', dim, 'val', med, ...
                'left',  obj.buildNode_(idx(leftMask),  depth+1), ...
                'right', obj.buildNode_(idx(rightMask), depth+1));
        end

        function node = leaf_(~, bounds, idx)
            node = struct('isLeaf', true, 'bounds', bounds, ...
                'idx', idx, 'dim', 0, 'val', 0, 'left', [], 'right', []);
        end

    end

    % =====================================================================
    %  Private — k-NN Search
    % =====================================================================
    methods (Access = private)

        function heap = searchKNN_(obj, node, q, heap)
            %walk the tree, prunes dead branches, updates the headp with any point closer than the current k-th
            if isempty(node), return; end

            % Prune once k slots are filled
            if heap.count >= heap.k
                if obj.minBoxDist_(node.bounds, q) >= heap.dists(1)
                    return;
                end
            end

            if node.isLeaf
                pts = obj.Points_(node.idx, :);
                for i = 1:size(pts, 1)
                    d    = obj.DistFn_(pts(i,:), q); %Calcualte dist to query
                    heap = obj.heapInsert_(heap, node.idx(i), d); % store
                end
                return;
            end

            % Visit near child first for faster pruning
            if q(node.dim) <= node.val
                heap = obj.searchKNN_(node.left,  q, heap);
                heap = obj.searchKNN_(node.right, q, heap);
            else
                heap = obj.searchKNN_(node.right, q, heap);
                heap = obj.searchKNN_(node.left,  q, heap);
            end
        end

        function heap = heapInsert_(~, heap, id, d)
            % Maintain descending-sorted dists (worst at index 1).
            if heap.count < heap.k
                heap.count = heap.count + 1;
                heap.ids(heap.count)   = id;
                heap.dists(heap.count) = d;
                if heap.count == heap.k   % first time full: sort
                    [heap.dists, ord] = sort(heap.dists, 'descend');
                    heap.ids          = heap.ids(ord);
                end
            elseif d < heap.dists(1)
                heap.dists(1) = d;
                heap.ids(1)   = id;
                [heap.dists, ord] = sort(heap.dists, 'descend');
                heap.ids          = heap.ids(ord);
            end
        end

    end

    % =====================================================================
    %  Private — Range Search
    % =====================================================================
    methods (Access = private)

        function [ids, dists] = searchRange_(obj, node, q, r)
            if isempty(node), ids = []; dists = []; return; end

            if obj.minBoxDist_(node.bounds, q) >= r
                ids = []; dists = []; return;
            end

            if node.isLeaf
                pts = obj.Points_(node.idx, :);
                d   = zeros(numel(node.idx), 1);
                for i = 1:numel(node.idx)
                    d(i) = obj.DistFn_(pts(i,:), q);
                end
                mask  = d < r;
                ids   = node.idx(mask);
                dists = d(mask);
                return;
            end

            %Interal node which overlap with search radius r
            [iL, dL] = obj.searchRange_(node.left,  q, r); 
            [iR, dR] = obj.searchRange_(node.right, q, r);
            ids   = [iL; iR];
            dists = [dL; dR];
        end

    end

    % =====================================================================
    %  Private — Geometry & Metric Helpers
    % =====================================================================
    methods (Access = private)

        function d = minBoxDist_(obj, bounds, q)
            %How far outside the box we are on the X,Y,Z,... coordindates. Raw distance
            % Minimum distance from query q to AABB bounds (2 x d).
            delta = max(0, max(bounds(1,:) - q, q - bounds(2,:)));
            switch obj.Metric_
                case 'euclidean',  d = sqrt(sum(delta .^ 2));
                case 'manhattan',  d = sum(delta);
                case 'chebyshev',  d = max(delta);
                otherwise,         d = 0;   % no pruning for custom
            end
        end

        function fn = makeDistFn_(~, metric)
            if isa(metric, 'function_handle')
                fn = metric; return;
            end
            switch lower(metric)
                case 'euclidean',  fn = @(a,b) sqrt(sum((a-b).^2));
                case 'manhattan',  fn = @(a,b) sum(abs(a-b));
                case 'chebyshev',  fn = @(a,b) max(abs(a-b));
                otherwise
                    error('KDTree:UnknownMetric', ...
                        'Unknown metric "%s". Use euclidean/manhattan/chebyshev or a function handle.', metric);
            end
        end

        function m = resolveMetricName_(~, metric)
            if isa(metric, 'function_handle'), m = 'custom'; return; end
            m = lower(metric);
        end

        function assertDim_(obj, q)
            if numel(q) ~= obj.D_
                error('KDTree:DimMismatch', ...
                    'Query has %d dims but tree was built on %d dims.', numel(q), obj.D_);
            end
        end

    end

    % =====================================================================
    %  Private — Diagnostics
    % =====================================================================
    methods (Access = private)

        function [nLeaves, maxDepth, nPts] = treeStats_(obj, node, depth)
            if node.isLeaf
                nLeaves = 1; maxDepth = depth; nPts = numel(node.idx);
                return;
            end
            [lL,dL,pL] = obj.treeStats_(node.left,  depth+1);
            [lR,dR,pR] = obj.treeStats_(node.right, depth+1);
            nLeaves  = lL + lR;
            maxDepth = max(dL, dR);
            nPts     = pL + pR;
        end

    end
end
