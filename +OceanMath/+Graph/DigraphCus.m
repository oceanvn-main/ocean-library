classdef DigraphCus < handle
    % DIGRAPHCUS  Generic weighted graph (digraph-style architecture).
    %
    % A pure graph data structure — no domain-specific logic.
    % Use wrappers (e.g., TransitNetwork) for domain vocabularies.
    %
    % Internal structure:
    %   Nodes_  : (1 x V) cell array of Node value objects
    %   Edges_  : (1 x E) cell array of Edge value objects
    %   W_      : (V x V) weight matrix (single source of truth)
    %             W_(i,j) > 0 means edge exists with that weight; 0 = no edge.
    %
    % Supports undirected (default) and directed graphs.
    %
    % Usage:
    %   g = OceanMath.Graph.DigraphCus();
    %   g.addNode('A', [0, 0]);
    %   g.addNode('B', [3, 4]);
    %   g.addEdge(1, 2, 5);
    %
    %   % Algorithms live externally:
    %   [path, dist] = OceanMath.Algorithms.Graph.Dijkstra.solve(g.getAdjacencyMatrix(), 1, 2);
    %
    % See also: OceanMath.Graph.Node, OceanMath.Graph.Edge,
    %           OceanMath.Algorithms.Graph.Dijkstra, OceanMath.Algorithms.Graph.NetworkAnalytics

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        Nodes_    % cell array of Node objects
        Edges_    % cell array of Edge objects
        W_        % (V x V) weight matrix
        Directed_ (1,1) logical = false
        NodeCount_ (1,1) double = 0
        EdgeCount_ (1,1) double = 0
        EdgeIdMap_        % O(1) dictionary mapping 'src_tgt' -> edgeIdx
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = DigraphCus(options)
            % DIGRAPHCUS  Construct an empty graph.
            %
            %   g = DigraphCus()                  % undirected
            %   g = DigraphCus('Directed', true)  % directed
            arguments
                options.Directed (1,1) logical = false
            end
            obj.Directed_ = options.Directed;
            obj.Nodes_ = {};
            obj.Edges_ = {};
            obj.W_ = [];
            obj.EdgeIdMap_ = dictionary(string.empty, double.empty);
        end

        % ============================================================== %
        %  Node Operations
        % ============================================================== %

        function reserveNodes(obj, N)
            % RESERVENODES  Pre-allocate matrix storage to avoid O(N^3) bottleneck.
            if size(obj.W_, 1) < N
                newW = zeros(N, N);
                if obj.NodeCount_ > 0
                    c = obj.NodeCount_;
                    newW(1:c, 1:c) = obj.W_(1:c, 1:c);
                end
                obj.W_ = newW;
                obj.Nodes_{N} = [];
            end
        end

        function id = addNode(obj, name, position, metadata)
            % ADDNODE  Add a vertex to the graph.
            arguments
                obj
                name      (1,:) char
                position  (1,:) double = [0, 0]
                metadata  (1,1) struct = struct()
            end
            obj.NodeCount_ = obj.NodeCount_ + 1;
            id = obj.NodeCount_;
            node = OceanMath.Graph.Node(id, name, position, metadata);
            obj.Nodes_{id} = node;

            % Expand weight matrix only if not pre-allocated
            if id > size(obj.W_, 1)
                newW = zeros(id, id);
                if id > 1
                    c = id - 1;
                    newW(1:c, 1:c) = obj.W_(1:c, 1:c); % Copy old matrix
                end
                obj.W_ = newW;
            end
        end

        function node = getNode(obj, id)
            % GETNODE  Retrieve a Node by its integer ID.
            obj.validateNodeId_(id);
            node = obj.Nodes_{id};
        end

        function node = getNodeByName(obj, name)
            % GETNODEBYNAME  Find a node by its Name string.
            %   Returns empty [] if not found.
            arguments
                obj
                name (1,:) char
            end
            for i = 1:obj.NodeCount_
                if ~isempty(obj.Nodes_{i}) && strcmp(obj.Nodes_{i}.Name, name)
                    node = obj.Nodes_{i};
                    return;
                end
            end
            node = [];
        end

        %Soft deletion. 
        function removeNode(obj, id)
            % REMOVENODE  Remove a node and all its incident edges.
            obj.validateNodeId_(id);

            % Remove incident edges (iterate in reverse)
            toRemove = [];
            %Find all edge touching this node. Throught metadata Edges
            for i = 1:obj.EdgeCount_
                if ~isempty(obj.Edges_{i})
                    e = obj.Edges_{i};
                    if e.Source == id || e.Target == id
                        toRemove = [toRemove, i]; %#ok<AGROW>
                    end
                end
            end
            for i = toRemove
                obj.Edges_{i} = [];
            end

            % Zero out row and column in weight matrix
            obj.W_(id, :) = 0;
            obj.W_(:, id) = 0;

            % Mark node as removed (set to empty)
            obj.Nodes_{id} = [];
        end

        function updateNodeName(obj, id, newName)
            % UPDATENODENAME  Rename an existing node.
            arguments
                obj
                id (1,1) double {mustBePositive, mustBeInteger}
                newName (1,:) char
            end
            obj.validateNodeId_(id);
            old = obj.Nodes_{id};
            obj.Nodes_{id} = OceanMath.Graph.Node(old.Id, newName, old.Position, old.Metadata);
        end

        function updateNodePosition(obj, id, newPosition)
            % UPDATENODEPOSITION  Move a node to a new position.
            %   Creates a new Node object (since Node is immutable).
            arguments
                obj
                id          (1,1) double {mustBePositive, mustBeInteger}
                newPosition (1,:) double
            end
            obj.validateNodeId_(id);
            old = obj.Nodes_{id};
            obj.Nodes_{id} = OceanMath.Graph.Node(old.Id, old.Name, newPosition, old.Metadata);
        end

        function N = getNodeCount(obj)
            N = obj.NodeCount_;
        end

        function names = getNodeNames(obj)
            % GETNODENAMES  Return cell array of all node names.
            names = cell(1, obj.NodeCount_);
            for i = 1:obj.NodeCount_
                if ~isempty(obj.Nodes_{i})
                    names{i} = obj.Nodes_{i}.Name;
                else
                    names{i} = '';
                end
            end
        end

        % ============================================================== %
        %  Edge Operations
        % ============================================================== %

        function edgeIdx = addEdge(obj, srcId, tgtId, weight, options)
            % ADDEDGE  Connect two nodes with a weighted edge.
            arguments
                obj
                srcId   (1,1) double {mustBePositive, mustBeInteger}
                tgtId   (1,1) double {mustBePositive, mustBeInteger}
                weight  (1,1) double {mustBePositive}
                options.Label (1,:) char = ''
            end
            obj.validateNodeId_(srcId);
            obj.validateNodeId_(tgtId);
            % Create an immutable Edge value object and appends to bhe Edge_{} cell array
            edge = OceanMath.Graph.Edge(srcId, tgtId, weight, ...
                'Label', options.Label);
            obj.EdgeCount_ = obj.EdgeCount_ + 1;
            edgeIdx = obj.EdgeCount_;
            obj.Edges_{edgeIdx} = edge;

            % Update weight matrix
            obj.W_(srcId, tgtId) = weight;
            obj.EdgeIdMap_(sprintf('%d_%d', srcId, tgtId)) = edgeIdx;
            
            if ~obj.Directed_
                obj.W_(tgtId, srcId) = weight;
                obj.EdgeIdMap_(sprintf('%d_%d', tgtId, srcId)) = edgeIdx;
            end
        end

        function removeEdge(obj, srcId, tgtId)
            % REMOVEEDGE  Remove the edge between two nodes.
            obj.validateNodeId_(srcId);
            obj.validateNodeId_(tgtId);

            % Zero out in weight matrix
            obj.W_(srcId, tgtId) = 0;
            if ~obj.Directed_
                obj.W_(tgtId, srcId) = 0;
            end

            % Remove from edge list
            for i = 1:obj.EdgeCount_
                if ~isempty(obj.Edges_{i})
                    e = obj.Edges_{i};
                    if e.Source == srcId && e.Target == tgtId
                        obj.Edges_{i} = []; % Remove edge from list
                        obj.EdgeIdMap_(sprintf('%d_%d', srcId, tgtId)) = [];
                        return;
                    end
                    % For undirected, also check reverse
                    if ~obj.Directed_ && e.Source == tgtId && e.Target == srcId
                        obj.Edges_{i} = [];
                        obj.EdgeIdMap_(sprintf('%d_%d', tgtId, srcId)) = [];
                        return;
                    end
                end
            end
        end

        function updateEdgeWeight(obj, edgeIdx, newWeight)
            % UPDATEEDGEWEIGHT  Update the weight of an existing edge.
            arguments
                obj
                edgeIdx   (1,1) double {mustBePositive, mustBeInteger}
                newWeight (1,1) double {mustBePositive}
            end
            e = obj.Edges_{edgeIdx};
            obj.W_(e.Source, e.Target) = newWeight; % Update weight matrix
            if ~obj.Directed_
                obj.W_(e.Target, e.Source) = newWeight;
            end
        end

        function E = getEdgeCount(obj)
            E = obj.EdgeCount_;
        end

        function e = getEdge(obj, idx)
            % GETEDGE  Retrieve an Edge by its index.
            e = obj.Edges_{idx};
        end

        function idx = getEdgeId(obj, srcId, tgtId)
            % GETEDGEID  O(1) hash map lookup for edge index.
            key = sprintf('%d_%d', srcId, tgtId);
            if isKey(obj.EdgeIdMap_, key)
                idx = obj.EdgeIdMap_(key);
            else
                idx = 0;
            end
        end

        % ============================================================== %
        %  Graph Queries
        % ============================================================== %

        function nbrs = neighbors(obj, nodeId)
            % NEIGHBORS  Return IDs of all nodes connected to nodeId.
            obj.validateNodeId_(nodeId);
            row = obj.W_(nodeId, :); %O(V) scan of one of the row matrix
            nbrs = find(row > 0);
        end

        function d = degree(obj, nodeId)
            % DEGREE  Number of edges connected to nodeId.
            d = numel(obj.neighbors(nodeId));
        end

        function w = getEdgeWeight(obj, srcId, tgtId)
            % GETEDGEWEIGHT  Return the weight between two nodes (0 = no edge).
            obj.validateNodeId_(srcId);
            obj.validateNodeId_(tgtId);
            w = obj.W_(srcId, tgtId);
        end

        function tf = hasEdge(obj, srcId, tgtId)
            % HASEDGE  Check if an edge exists between two nodes. O(1).
            tf = obj.W_(srcId, tgtId) > 0;
        end

        function A = getAdjacencyMatrix(obj)
            % GETADJACENCYMATRIX  Return the (V x V) weight matrix.
            A = obj.W_;
        end

        function positions = getNodePositions(obj)
            % GETNODEPOSITIONS  Return (V x D) matrix of node coordinates.
            if obj.NodeCount_ == 0
                positions = [];
                return;
            end
            % Find max dimensionality to safely pad 2D vectors imported from JSON
            d = 2; % Minimum 2D
            for i = 1:obj.NodeCount_
                if ~isempty(obj.Nodes_{i}) && numel(obj.Nodes_{i}.Position) > d
                    d = numel(obj.Nodes_{i}.Position);
                end
            end
            positions = zeros(obj.NodeCount_, d); % V x D output matrix
            for i = 1:obj.NodeCount_
                if ~isempty(obj.Nodes_{i}) % Row Index = Node ID
                    pos = obj.Nodes_{i}.Position;
                    positions(i, 1:numel(pos)) = pos;
                end
            end
        end

        function tf = isDirected(obj)
            % ISDIRECTED  Return whether the graph is directed.
            tf = obj.Directed_;
        end



        % ============================================================== %
        %  Serialization
        % ============================================================== %

        function s = toStruct(obj)
            % TOSTRUCT  Export graph as a plain struct for saving.
            nodeData = cell(1, obj.NodeCount_);
            for i = 1:obj.NodeCount_
                if ~isempty(obj.Nodes_{i})
                    n = obj.Nodes_{i};
                    nodeData{i} = struct('Id', n.Id, 'Name', n.Name, ...
                        'Position', n.Position, 'Metadata', n.Metadata);
                else
                    nodeData{i} = struct('Id', i, 'Name', '', ...
                        'Position', [0 0], 'Metadata', struct());
                end
            end
            edgeData = {};
            for i = 1:obj.EdgeCount_
                if ~isempty(obj.Edges_{i})
                    e = obj.Edges_{i};
                    edgeData{end+1} = struct('Source', e.Source, 'Target', e.Target, ...
                        'Weight', e.Weight, 'Label', e.Label); %#ok<AGROW>
                end
            end
            s = struct('Nodes', {nodeData}, 'Edges', {edgeData}, ...
                'WeightMatrix', obj.W_, 'Directed', obj.Directed_);
        end

        % ============================================================== %
        %  Display
        % ============================================================== %

        function describe(obj)
            % DESCRIBE  Print summary statistics.
            fprintf('DigraphCus:\n');
            fprintf('  Nodes:    %d\n', obj.NodeCount_);
            fprintf('  Edges:    %d\n', obj.EdgeCount_);
            fprintf('  Directed: %s\n', mat2str(obj.Directed_));
        end
    end

    % ------------------------------------------------------------------ %
    methods (Access = private)
        function validateNodeId_(obj, id)
            if id < 1 || id > obj.NodeCount_ || mod(id, 1) ~= 0
                error('OceanMath:Graph:DigraphCus:InvalidNode', ...
                    'Node ID must be integer in [1, %d]. Got %g.', ...
                    obj.NodeCount_, id);
            end
        end
    end
end
