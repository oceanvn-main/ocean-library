classdef GraphMapper < OceanViz.Adapters.ISpatialMapper
    % GRAPHMAPPER  Maps state indices to arbitrary node positions in a graph.
    %
    % Unlike GridMapper2D (which computes positions from a formula), this
    % mapper uses user-supplied node coordinates. Each state corresponds
    % to a node in a graph with a known physical or visual position.
    %
    % Suitable for:
    %   - Random walks on arbitrary graphs (social, road, knowledge graphs)
    %   - Visualizing Markov chains where states have no geometric regularity
    %   - Any network topology (erdos-renyi, scale-free, geographic, etc.)
    %
    % Layout algorithms supported (optional automatic positioning):
    %   'manual'       - User provides node positions directly (default)
    %   'spring'       - Force-directed layout (Fruchterman-Reingold)
    %   'circular'     - Evenly spaced on a unit circle
    %   'spectral'     - Eigenvectors of the graph Laplacian (for Markov chains)
    %
    % Usage:
    %   % Manual positions (most common):
    %   positions = [0 0; 3 1; 5 4; 1 3];   % (K x 2) matrix
    %   mapper    = OceanViz.Adapters.GraphMapper(positions);
    %   coords    = mapper.mapState(2);       % [3, 1]
    %
    %   % Auto layout from a transition matrix (spring):
    %   mapper = OceanViz.Adapters.GraphMapper.fromAdjacency(P, 'Layout', 'spring');
    %
    % See also: OceanViz.Adapters.ISpatialMapper
    %           OceanViz.Adapters.GridMapper2D

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        % NodePositions_ : (K x D) matrix; row k = coordinates of state k
        NodePositions_ (:,:) double
        Layout_        (1,:) char = 'manual'
        
        % AdjacencyMatrix_ : (K x K) matrix of weights (optional)
        AdjacencyMatrix_ (:,:) double
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = GraphMapper(nodePositions, options)
            % GRAPHMAPPER  Construct from a (K x D) position matrix.
            %
            % INPUTS:
            %   nodePositions - (K x D) double matrix.
            %       K = number of states, D = 2 or 3 (spatial dimensions).
            %       Row k is the coordinate of state k.
            %
            % Name-Value:
            %   Layout (char): layout algorithm label (for display). Default: 'manual'.
            arguments
                nodePositions (:,:) double
                options.Layout (1,:) char = 'manual'
                options.Adjacency (:,:) double = []
            end
            if size(nodePositions, 1) < 1
                error('OceanViz:Adapters:GraphMapper:Empty', ...
                    'nodePositions must have at least one row.');
            end
            if ~ismember(size(nodePositions, 2), [2, 3])
                error('OceanViz:Adapters:GraphMapper:BadDimension', ...
                    'nodePositions must be (K x 2) or (K x 3). Got K x %d.', ...
                    size(nodePositions, 2));
            end
            obj.NodePositions_ = nodePositions;
            obj.Layout_        = options.Layout;
            obj.AdjacencyMatrix_ = options.Adjacency;
        end

        % ---- ISpatialMapper Implementation ---- %

        function coords = mapState(obj, stateIndex)
            % MAPSTATE  Return the node's physical coordinates. O(1).
            if ~obj.isValidState(stateIndex)
                error('OceanViz:Adapters:GraphMapper:InvalidState', ...
                    'State %d is out of range [1, %d].', stateIndex, obj.getNumStates());
            end
            coords = obj.NodePositions_(stateIndex, :);
        end

        function coordSeq = mapSequence(obj, stateSeq)
            % MAPSEQUENCE  Vectorized batch lookup — zero loop overhead.
            idx      = double(stateSeq(:));
            coordSeq = obj.NodePositions_(idx, :);
        end

        function D = getDimensions(obj)
            D = size(obj.NodePositions_, 2);
        end

        function N = getNumStates(obj)
            N = size(obj.NodePositions_, 1);
        end

        function A = getAdjacency(obj)
            A = obj.AdjacencyMatrix_;
        end

        function disp(obj)
            fprintf('GraphMapper  [%d nodes, %dD, layout: %s]\n', ...
                obj.getNumStates(), obj.getDimensions(), obj.Layout_);
        end
    end

    % ------------------------------------------------------------------ %
    methods (Static)
        function obj = fromAdjacency(A, options)
            % FROMADJACENCY  Auto-compute node positions from an adjacency matrix.
            %
            % INPUTS:
            %   A - (K x K) adjacency or transition matrix (non-negative)
            %
            % Name-Value:
            %   Layout (char): 'circular', 'spring', 'spectral'. Default: 'circular'.
            %
            % The 'spectral' layout uses the two smallest non-trivial
            % eigenvectors of the graph Laplacian for a layout that
            % preserves the connectivity structure of the graph.
            arguments
                A       (:,:) double {mustBeNonnegative}
                options.Layout (1,:) char = 'circular'
            end

            K = size(A, 1);
            if size(A, 2) ~= K
                error('OceanViz:Adapters:GraphMapper:NotSquare', ...
                    'Adjacency matrix must be square.');
            end

            switch lower(options.Layout)
                case 'circular'
                    angles   = linspace(0, 2*pi, K+1)'; angles(end) = [];
                    positions = [cos(angles), sin(angles)];

                case 'spectral'
                    % Graph Laplacian: L = D - A (where D is degree matrix)
                    % Use eigenvectors 2 & 3 (skip the trivial first one)
                    deg = sum(A, 2);
                    L   = diag(deg) - A;
                    [V, ~] = eig(L);
                    % Sort by eigenvalue (ascending), skip eigvec 1 (constant)
                    positions = V(:, 2:3);

                case 'spring'
                    % Fruchterman-Reingold force-directed layout
                    % Initialize randomly, then iterate attractive/repulsive forces
                    positions = OceanViz.Adapters.GraphMapper.fruchtermanReingold_(A, K);

                otherwise
                    error('OceanViz:Adapters:GraphMapper:UnknownLayout', ...
                        'Unknown layout: "%s". Use circular, spectral, or spring.', ...
                        options.Layout);
            end

            obj = OceanViz.Adapters.GraphMapper(positions, ...
                'Layout', options.Layout, 'Adjacency', A);
        end
    end

    % ------------------------------------------------------------------ %
    methods (Static, Access = private)
        function pos = fruchtermanReingold_(A, K)
            % FRUCHTERMANREINGOLD_  Force-directed layout algorithm.
            % Minimizes edge crossings and normalizes edge lengths.
            rng(0);   % deterministic seed for reproducibility
            pos = rand(K, 2) * 2 - 1;   % uniform in [-1, 1]

            area   = 1;
            k_opt  = sqrt(area / K);     % optimal pairwise distance
            nIter  = 100;
            temp   = 1.0;                % simulated annealing temperature
            cool   = temp / nIter;       % linear cooling schedule

            for iter = 1:nIter
                disp_ = zeros(K, 2);

                % Repulsive forces (all pairs)
                for i = 1:K
                    for j = 1:K
                        if i ~= j
                            delta = pos(i,:) - pos(j,:);
                            dist  = max(norm(delta), 1e-6);
                            disp_(i,:) = disp_(i,:) + (delta/dist) * (k_opt^2 / dist);
                        end
                    end
                end

                % Attractive forces (connected pairs only)
                for i = 1:K
                    for j = 1:K
                        if A(i,j) > 0
                            delta = pos(i,:) - pos(j,:);
                            dist  = max(norm(delta), 1e-6);
                            disp_(i,:) = disp_(i,:) - (delta/dist) * (dist^2 / k_opt);
                        end
                    end
                end

                % Apply displacement, clamped by temperature
                for i = 1:K
                    d = norm(disp_(i,:));
                    if d > 0
                        pos(i,:) = pos(i,:) + (disp_(i,:)/d) * min(d, temp);
                    end
                end

                temp = max(temp - cool, 0);
            end
        end
    end
end
