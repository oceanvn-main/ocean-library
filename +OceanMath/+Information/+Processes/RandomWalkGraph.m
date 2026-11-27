classdef RandomWalkGraph < OceanMath.Information.Processes.IStochasticProcess
    % RANDOMWALKGRAPH  A stochastic process defined by a graph adjacency matrix.
    %
    % Models a random walk on an arbitrary undirected or directed graph.
    % The transition matrix P is derived automatically from the adjacency
    % weights: each row is normalized so that probabilities sum to 1.
    %
    %   P(i, j) = W(i,j) / sum(W(i,:))
    %
    % where W is the adjacency (weight) matrix. This means:
    %   - If W is unweighted {0,1}: uniform probability over neighbors.
    %   - If W is weighted: proportional probability over edge weights.
    %
    % Importantly, RandomWalkGraph knows NOTHING about geometry (grids,
    % coordinates, etc.). It only understands graph connectivity.
    % Spatial meaning is provided externally by an ISpatialMapper.
    %
    % Factory Constructors (preferred over direct construction):
    %   obj = RandomWalkGraph.fromAdjacency(W)       % arbitrary graph
    %   obj = RandomWalkGraph.fromGrid(numCols, numRows)  % 2D grid topology
    %   obj = RandomWalkGraph.fromGrid2D(numCols)    % square 2D grid shorthand
    %
    % Usage:
    %   % Build from a 3x3 grid topology:
    %   rwg = OceanMath.Information.Processes.RandomWalkGraph.fromGrid2D(3);
    %   sim = OceanMath.Simulation.MarkovSimulator(rwg);
    %
    %   % Build from arbitrary adjacency:
    %   W   = [0 1 0; 1 0 1; 0 1 0];
    %   rwg = OceanMath.Information.Processes.RandomWalkGraph.fromAdjacency(W);
    %
    % See also: OceanMath.Information.Processes.IStochasticProcess
    %           OceanMath.Information.Processes.MarkovChain
    %           OceanViz.Adapters.GraphMapper

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        % Normalized row-stochastic transition matrix derived from adjacency
        P_           (:,:) double

        % Raw adjacency / weight matrix (stored for reference, not computation)
        Adjacency_   (:,:) double

        % Stationary distribution (computed lazily on first request)
        Stationary_  = []
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = RandomWalkGraph(adjacency)
            % RANDOMWALKGRAPH  Construct from a (K x K) adjacency matrix.
            %
            % INPUTS:
            %   adjacency - (K x K) non-negative matrix.
            %       adjacency(i,j) > 0 means there is an edge from i to j.
            %       Zero-columns in a row mean state i is a dead end (error).
            arguments
                adjacency (:,:) double {mustBeNonnegative}
            end
            [K, Kc] = size(adjacency);
            if K ~= Kc
                error('OceanMath:Processes:RandomWalkGraph:NotSquare', ...
                    'Adjacency matrix must be square. Got [%d x %d].', K, Kc);
            end
            if K < 2
                error('OceanMath:Processes:RandomWalkGraph:TooSmall', ...
                    'Graph must have at least 2 nodes. Got %d.', K);
            end

            rowSums = sum(adjacency, 2);
            if any(rowSums == 0)
                deadEnds = find(rowSums == 0)';
                error('OceanMath:Processes:RandomWalkGraph:DeadEnd', ...
                    'State(s) [%s] have no outgoing edges (dead ends).', ...
                    num2str(deadEnds));
            end

            obj.Adjacency_ = adjacency;
            obj.P_         = adjacency ./ rowSums;   % row-normalize
        end

        % ---- IStochasticProcess Implementation ---- %

        function P = getTransitionMatrix(obj)
            P = obj.P_;
        end

        function p = getTransitionProbability(obj, stateI, stateJ)
            p = obj.P_(stateI, stateJ);
        end

        function K = getNumStates(obj)
            K = size(obj.P_, 1);
        end

        function dist = getStationaryDistribution(obj)
            % Lazy-computed: only computed once, then cached.
            if isempty(obj.Stationary_)
                obj.Stationary_ = obj.computeStationary_();
            end
            dist = obj.Stationary_;
        end

        function tf = isErgodic(obj)
            % ISERGODIC  True iff the chain is irreducible AND aperiodic.
            %
            % IMPORTANT: Both conditions are required for ergodicity.
            %   - Irreducibility: every state can reach every other state.
            %     (checked by DFS from every node on the directed graph)
            %   - Aperiodicity: the GCD of all return-cycle lengths = 1.
            %     (checked by BFS-based GCD cycle detection)
            %
            % Counter-example caught by this fix:
            %   A simple 1<->2 bipartite graph is irreducible but has
            %   period 2, so DFS alone would wrongly return true.
            tf = obj.isIrreducible_() && (obj.getPeriod() == 1);
        end

        function d = getPeriod(obj)
            % GETPERIOD  Return the period of the Markov chain.
            %
            % Period d = GCD of all cycle lengths reachable from any node.
            % For a strongly connected graph, all states share the same period.
            %   d = 1  → aperiodic (chain mixes to a unique stationary dist)
            %   d > 1  → periodic  (chain oscillates with period d)
            %
            % Algorithm: BFS from node 1, assign distances. For every edge
            % (u->v) where v is already visited, the implied cycle length is
            % d[u] + 1 - d[v]. Period = GCD of all such lengths.
            d = obj.computePeriod_();
        end

        function disp(obj)
            K      = obj.getNumStates();
            nEdges = sum(obj.Adjacency_(:) > 0);
            d      = obj.getPeriod();
            fprintf('RandomWalkGraph  [%d nodes, %d directed edges]\n', K, nEdges);
            fprintf('  Avg degree:   %.2f\n', nEdges / K);
            fprintf('  Irreducible:  %s\n', mat2str(obj.isIrreducible_()));
            fprintf('  Period:       %d%s\n', d, obj.periodLabel_(d));
            fprintf('  Ergodic:      %s\n', mat2str(obj.isErgodic()));
        end
    end

    % ================================================================== %
    methods (Static)
        function obj = fromAdjacency(W, options)
            % FROMADJACENCY  Construct from an arbitrary weight matrix.
            %
            % INPUTS:
            %   W - (K x K) non-negative adjacency or weight matrix.
            %
            % Name-Value:
            %   Symmetric (logical): force symmetry by using (W + W')/2.
            %       Default: false.
            arguments
                W        (:,:) double {mustBeNonnegative}
                options.Symmetric (1,1) logical = false
            end
            if options.Symmetric
                W = (W + W') / 2;
            end
            obj = OceanMath.Information.Processes.RandomWalkGraph(W);
        end

        function obj = fromGrid(numCols, numRows, options)
            % FROMGRID  Build a uniform random walk on a 2D rectangular grid.
            %
            % States are indexed row-major (left-to-right, top-to-bottom).
            % Adjacent cells (Up, Down, Left, Right) are connected by unit edges.
            % Boundary cells connect only to valid in-grid neighbors, so boundary
            % states naturally have fewer neighbors and the walk reflects inward.
            %
            % INPUTS:
            %   numCols - positive integer, number of grid columns
            %   numRows - positive integer, number of grid rows
            %
            % Name-Value Options:
            %   SelfLoops (double): Weight assigned to staying in the current cell. 
            %       > 0 makes the graph aperiodic (ergodic). Default: 0.
            %   Diagonals (logical): If true, connects the 4 diagonal neighbors 
            %       (8-connectivity). Makes the graph aperiodic. Default: false.
            arguments
                numCols (1,1) double {mustBePositive, mustBeInteger}
                numRows (1,1) double {mustBePositive, mustBeInteger}
                options.SelfLoops (1,1) double {mustBeNonnegative} = 0.0
                options.Diagonals (1,1) logical = false
            end
            N = numCols * numRows;
            W = zeros(N);

            for s = 1:N
                c = mod(s-1, numCols) + 1;              % column (1-indexed)
                r = floor((s-1) / numCols) + 1;         % row (1-indexed)

                % -- Cardinal directions (4-way) --
                if c > 1,       W(s, s - 1)       = 1; end   % Left
                if c < numCols, W(s, s + 1)       = 1; end   % Right
                if r > 1,       W(s, s - numCols) = 1; end   % Up
                if r < numRows, W(s, s + numCols) = 1; end   % Down

                % -- Diagonal directions (8-way) --
                if options.Diagonals
                    if c > 1 && r > 1,             W(s, s - 1 - numCols) = 1; end % Up-Left
                    if c < numCols && r > 1,       W(s, s + 1 - numCols) = 1; end % Up-Right
                    if c > 1 && r < numRows,       W(s, s - 1 + numCols) = 1; end % Down-Left
                    if c < numCols && r < numRows, W(s, s + 1 + numCols) = 1; end % Down-Right
                end
            end

            % -- Self-Loops (Stay in place) --
            if options.SelfLoops > 0
                W = W + options.SelfLoops * eye(N);
            end

            obj = OceanMath.Information.Processes.RandomWalkGraph(W);
        end

        function obj = fromGrid2D(numCols, options)
            % FROMGRID2D  Shorthand for a square numCols x numCols grid.
            arguments
                numCols (1,1) double {mustBePositive, mustBeInteger}
                options.SelfLoops (1,1) double {mustBeNonnegative} = 0.0
                options.Diagonals (1,1) logical = false
            end
            obj = OceanMath.Information.Processes.RandomWalkGraph.fromGrid( ...
                numCols, numCols, ...
                'SelfLoops', options.SelfLoops, ...
                'Diagonals', options.Diagonals);
        end
    end

    % ================================================================== %
    methods (Access = private)
        function dist = computeStationary_(obj)
            % COMPUTESTATIONARY_  Solve mu*P = mu via left eigenvector.
            import OceanMath.Information.Distributions.DiscreteDistribution;
            [V, D] = eig(obj.P_');
            eigenvalues = diag(D);
            [~, idx]   = min(abs(eigenvalues - 1));
            mu = abs(real(V(:, idx)));
            mu = mu / sum(mu);
            support = (1:obj.getNumStates())';
            dist    = DiscreteDistribution.fromPMF(support, mu);
        end

        function tf = isIrreducible_(obj)
            % ISIRREDUCIBLE_  True iff every node can reach every other node.
            % Uses DFS from every node on the binary adjacency graph.
            K = obj.getNumStates();
            A = double(obj.Adjacency_ > 0);
            tf = true;
            for s = 1:K
                if sum(obj.dfsReach_(A, s, K)) < K
                    tf = false;
                    return;
                end
            end
        end

        function d = computePeriod_(obj)
            % COMPUTEPERIOD_  BFS-based GCD cycle-length period detection.
            %
            % Algorithm:
            %   1. BFS from node 1, assign distance dist[v] = BFS level.
            %   2. For each edge (u -> v) where dist[v] already assigned:
            %      The edge closes a cycle of length (dist[u] + 1 - dist[v]).
            %      Compute the running GCD of all such cycle lengths.
            %   3. Period = final GCD. If no cross-edges found, period = 1.
            %
            % Correctness: for a strongly connected graph, all nodes share
            % the same period, so BFS from node 1 is sufficient.
            K    = obj.getNumStates();
            A    = double(obj.Adjacency_ > 0);
            dist = -ones(1, K);     % -1 = unvisited
            dist(1) = 0;
            queue   = 1;
            d       = 0;            % running GCD of cycle lengths; 0 = identity

            while ~isempty(queue)
                u     = queue(1);
                queue = queue(2:end);
                nbrs  = find(A(u,:) > 0);
                for v = nbrs
                    if dist(v) == -1
                        % Tree edge: first visit, assign distance
                        dist(v) = dist(u) + 1;
                        queue(end+1) = v; %#ok<AGROW>
                    else
                        % Cross/back edge: closes a cycle
                        cycleLen = dist(u) + 1 - dist(v);
                        if cycleLen > 0
                            d = gcd(d, cycleLen);
                        end
                    end
                end
            end

            % If GCD is still 0 (no cross-edges seen from node 1), period = 1
            if d == 0
                d = 1;
            end
        end

        function visited = dfsReach_(~, A, startNode, K)
            % DFSREACH_  Depth-first search from startNode on adjacency A.
            visited            = false(1, K);
            visited(startNode) = true;
            stack = startNode;
            while ~isempty(stack)
                node  = stack(end);
                stack = stack(1:end-1);
                nbrs  = find(A(node,:) > 0);
                for nb = nbrs
                    if ~visited(nb)
                        visited(nb)  = true;
                        stack(end+1) = nb; %#ok<AGROW>
                    end
                end
            end
        end

        function lbl = periodLabel_(~, d)
            % PERIODLABEL_  Human-readable annotation for the period value.
            if d == 1
                lbl = '  (aperiodic)';
            else
                lbl = sprintf('  (periodic — oscillates every %d steps)', d);
            end
        end
    end
end
