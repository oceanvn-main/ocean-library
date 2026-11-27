classdef NNGraph < handle
    % NNGRAPH  Mutable neural network topology + state container.
    %   OBSERVER PATTERN
    
    %   Central orchestrator. Holds all NNLayer, NNNode, and NNEdge objects
    %   for one network instance.
    %
    %   DESIGN PRINCIPLE (inspired by Flax NNX):
    %     - TOPOLOGY (shape) lives in NNTopologySpec (immutable value class).
    %     - PARAMETERS (weights, biases) live in NNNode/NNEdge (mutable handles).
    %     - NNGraph is the LIVE, MUTABLE VIEW built from a spec.
    %
    %   PHASE 2 — COMMAND / UNDO STACK:
    %     All mutations now go through CommandHistory. The public API is
    %     unchanged — callers still call disableNode(), setWeight() etc.
    %     The history enables undo/redo without changing any caller code.
    %
    %     graph.disableNode('L2_N2');          % executes + records
    %     graph.undo();                         % restores node to Active
    %     graph.redo();                         % re-disables
    %     graph.getHistory()                    % cell of describe() strings
    %
    %   EVENTS (Observer Pattern):
    %     TopologyChanged    — fired when nodes are disabled/enabled/swapped
    %     WeightsChanged     — fired when any weight or bias changes
    %     ActivationsUpdated — fired after forwardPass() completes
    %
    %   WEIGHT INITIALIZATION:
    %     ReLU layers  → He Normal:    W ~ N(0, sqrt(2/d_in))
    %     Other layers → Glorot Unif: W ~ U(-sqrt(6/(d_in+d_out)), ...)
    %     Biases always initialized to 0.
    %
    %   FORWARD PASS CONVENTION:
    %     X is (N × d_in) — N samples, d_in features (row = one sample).
    %     Internally transposed to (d_in × N) for efficient W * A computation.
    %     Returns A as (N × d_out).
    %
    %   See also: OceanDL.Graph.NNTopologySpec
    %             OceanDL.Graph.ParameterStore
    %             OceanDL.Commands.CommandHistory

    % ------------------------------------------------------------------ %
    events
        TopologyChanged     % node enabled/disabled/swapped
        WeightsChanged      % edge weight or node bias modified
        ActivationsUpdated  % forwardPass() finished updating node values
    end

    properties (Access = private)
        Layers_         % cell array of NNLayer  (index = layer number, 1-based)
        Edges_          % cell array of NNEdge   (all edges, active + inactive)
        Spec_           % NNTopologySpec — the immutable blueprint
        CommandHistory_ % OceanDL.Commands.CommandHistory
    end

    % ================================================================== %
    % --- Static Factories --------------------------------------------- %
    % ================================================================== %
    methods (Static)
        function graph = fromSpec(spec, initParams)
            % FROMSPEC  Build a live NNGraph from a NNTopologySpec.
            %
            %   graph = NNGraph.fromSpec(spec)
            %       Builds the graph with He/Glorot weight initialization.
            %
            %   graph = NNGraph.fromSpec(spec, paramVector)
            %       Builds then loads pre-computed parameters (from ParameterStore).
            arguments
                spec        OceanDL.Graph.NNTopologySpec
                initParams  (:,1) double = []
            end

            import OceanDL.Graph.*

            graph         = NNGraph();
            graph.Spec_   = spec;
            nL            = numel(spec.LayerSizes);
            graph.Layers_ = cell(1, nL);
            graph.Edges_  = {};

            % ---- Build layers + nodes -------------------------------- %
            for L = 1:nL
                if     L == 1,  ltype = 'input';
                elseif L == nL, ltype = 'output';
                else            ltype = 'hidden';
                end

                layer = NNLayer(int32(L), spec.LayerNames{L}, ltype);

                for n = 1:spec.LayerSizes(L)
                    node = NNNode(int32(L), int32(n), spec.Activations{L}, 0.0);
                    layer.addNode(node);
                end

                graph.Layers_{L} = layer;
            end

            % ---- Build edges (dense, adjacent layers only) ----------- %
            for L = 1:nL-1
                fromNodes = graph.Layers_{L}.getNodes();
                toNodes   = graph.Layers_{L+1}.getNodes();
                dIn       = numel(fromNodes);
                dOut      = numel(toNodes);

                % Weight initialization
                actNext   = lower(spec.Activations{L+1});
                if strcmp(actNext, 'relu')
                    % He Normal  —  optimal for ReLU
                    W = randn(dOut, dIn) * sqrt(2 / dIn);
                else
                    % Glorot Uniform  —  optimal for sigmoid / tanh / linear / softmax
                    bound = sqrt(6 / (dIn + dOut));
                    W     = (rand(dOut, dIn) * 2 - 1) * bound;
                end

                for i = 1:dOut
                    for j = 1:dIn
                        e = NNEdge(fromNodes{j}, toNodes{i}, W(i,j));
                        graph.Edges_{end+1} = e;
                    end
                end
            end

            % ---- Optionally load pre-computed parameters ------------- %
            if ~isempty(initParams)
                ps = ParameterStore();
                ps.setFromFlat(graph, initParams);
            end
        end
    end

    % ================================================================== %
    % --- Public Topology Queries -------------------------------------- %
    % ================================================================== %
    methods
        function n = numLayers(obj)
            n = numel(obj.Layers_);
        end

        function n = numNodes(obj, L)
            n = obj.Layers_{L}.numNodes();
        end

        function n = numActiveNodes(obj, L)
            n = obj.Layers_{L}.numActiveNodes();
        end

        function layer = getLayer(obj, L)
            layer = obj.Layers_{L};
        end

        function node = getNode(obj, L, n)
            node = obj.Layers_{L}.getNodeAt(n);
        end

        function node = getNodeByID(obj, id)
            % GETNODEBYID  Find a node by its string ID (e.g. "L2_N3").
            for L = 1:numel(obj.Layers_)
                nodes = obj.Layers_{L}.getNodes();
                for ni = 1:numel(nodes)
                    if strcmp(nodes{ni}.ID, id)
                        node = nodes{ni};
                        return;
                    end
                end
            end
            error('OceanDL:Graph:NNGraph:NodeNotFound', ...
                'No node with ID "%s" in this graph.', id);
        end

        function edges = getEdges(obj)
            % GETEDGES  All edges (including deactivated ones).
            edges = obj.Edges_;
        end

        function edges = getActiveEdges(obj)
            % GETACTIVEEDGES  Only edges where both endpoints are enabled.
            mask  = cellfun(@(e) e.isActive(), obj.Edges_);
            edges = obj.Edges_(mask);
        end

        function spec = toSpec(obj)
            % TOSPEC  Export the immutable topology blueprint (no weights).
            spec = obj.Spec_;
        end

        % ---- Weight matrix assembly ---------------------------------- %

        function W = getWeightMatrix(obj, L)
            % GETWEIGHTMATRIX  Build the d_{L+1} x d_L weight matrix for layer L→L+1.
            %   ALL nodes included (active and disabled), for ParameterStore.
            dIn  = obj.Layers_{L}.numNodes();
            dOut = obj.Layers_{L+1}.numNodes();
            W    = zeros(dOut, dIn);

            for ei = 1:numel(obj.Edges_)
                e = obj.Edges_{ei};
                if double(e.FromNode.LayerIndex) == L && ...
                   double(e.ToNode.LayerIndex)   == L+1
                    W(e.ToNode.NodeIndex, e.FromNode.NodeIndex) = e.Weight_;
                end
            end
        end

        function setWeightMatrix(obj, L, W)
            % SETWEIGHTMATRIX  Scatter a (d_{L+1} x d_L) matrix into edges.
            for ei = 1:numel(obj.Edges_)
                e = obj.Edges_{ei};
                if double(e.FromNode.LayerIndex) == L && ...
                   double(e.ToNode.LayerIndex)   == L+1
                    e.Weight_ = W(e.ToNode.NodeIndex, e.FromNode.NodeIndex);
                end
            end
        end

        function b = getBiasVector(obj, L)
            % GETBIASVECTOR  Return the (d_L x 1) bias column for layer L.
            %
            %   Used by VectorizedEngine to assemble the bias term in
            %   Z = W * A + b without iterating over individual nodes.
            %
            %   All nodes in the layer are included (active and disabled),
            %   consistent with getWeightMatrix convention.
            %
            %   b = graph.getBiasVector(2)   % biases for layer 2
            nodes = obj.Layers_{L}.getNodes();
            d     = numel(nodes);
            b     = zeros(d, 1);
            for ni = 1:d
                b(ni) = nodes{ni}.Bias_;
            end
        end

        % ---- ParameterStore convenience wrappers -------------------- %

        function v = flattenParams(obj)
            ps = OceanDL.Graph.ParameterStore();
            v  = ps.flatten(obj);
        end

        function setFlatParams(obj, v)
            ps = OceanDL.Graph.ParameterStore();
            ps.setFromFlat(obj, v);
            notify(obj, 'WeightsChanged');
        end

        function n = paramCount(obj)
            ps = OceanDL.Graph.ParameterStore();
            n  = ps.paramCount(obj);
        end
    end

    % ================================================================== %
    % --- State Mutation (Command wrappers in Phase 2) ----------------- %
    % ================================================================== %
    methods
        function disableNode(obj, nodeID)
            % DISABLENODE  Deactivate a node (recorded — undoable).
            import OceanDL.Commands.DisableNodeCommand;
            cmd = DisableNodeCommand(nodeID);
            obj.CommandHistory_.execute(cmd, obj);
            notify(obj, 'TopologyChanged');
        end

        function enableNode(obj, nodeID)
            % ENABLENODE  Re-activate a disabled node (recorded — undoable).
            import OceanDL.Commands.EnableNodeCommand;
            cmd = EnableNodeCommand(nodeID);
            obj.CommandHistory_.execute(cmd, obj);
            notify(obj, 'TopologyChanged');
        end

        function highlightNode(obj, nodeID)
            % HIGHLIGHTNODE  Highlight a node (not recorded — cosmetic only).
            node = obj.getNodeByID(nodeID);
            node.highlight();
            notify(obj, 'TopologyChanged');
        end

        function deselectNode(obj, nodeID)
            % DESELECTNODE  Deselect a node (not recorded — cosmetic only).
            node = obj.getNodeByID(nodeID);
            node.deselect();
            notify(obj, 'TopologyChanged');
        end

        function setWeight(obj, fromID, toID, w)
            % SETWEIGHT  Update a single edge weight (recorded — undoable).
            import OceanDL.Commands.SetWeightCommand;
            cmd = SetWeightCommand(fromID, toID, w);
            obj.CommandHistory_.execute(cmd, obj);
            notify(obj, 'WeightsChanged');
        end

        function setBias(obj, nodeID, b)
            % SETBIAS  Update a single node bias (recorded — undoable).
            import OceanDL.Commands.SetBiasCommand;
            cmd = SetBiasCommand(nodeID, b);
            obj.CommandHistory_.execute(cmd, obj);
            notify(obj, 'WeightsChanged');
        end

        function swapNodes(obj, idA, idB)
            % SWAPNODES  Swap two neurons in the same layer (recorded — undoable).
            %
            %   Permutation experiment: after swapping, the network computes
            %   an IDENTICAL function (same biases and weights, just relabeled).
            import OceanDL.Commands.SwapNodesCommand;
            cmd = SwapNodesCommand(idA, idB);
            obj.CommandHistory_.execute(cmd, obj);
            notify(obj, 'TopologyChanged');
        end

        % ---- Undo / Redo -------------------------------------------- %

        function undo(obj)
            % UNDO  Reverse the most recent recorded mutation.
            if obj.CommandHistory_.canUndo()
                obj.CommandHistory_.undo(obj);
                notify(obj, 'TopologyChanged');
            end
        end

        function redo(obj)
            % REDO  Re-apply the most recently undone mutation.
            if obj.CommandHistory_.canRedo()
                obj.CommandHistory_.redo(obj);
                notify(obj, 'TopologyChanged');
            end
        end

        function prune(obj, epsilon)
            % PRUNE  Zero-out small weights (recorded — undoable).
            if nargin < 2, epsilon = 0.05; end
            import OceanDL.Commands.PruningCommand;
            cmd = PruningCommand(epsilon);
            obj.CommandHistory_.execute(cmd, obj);
            notify(obj, 'WeightsChanged');
        end

        function tf = canUndo(obj)
            % CANUNDO  True iff there is at least one action to undo.
            tf = obj.CommandHistory_.canUndo();
        end

        function tf = canRedo(obj)
            % CANREDO  True iff there is at least one action to redo.
            tf = obj.CommandHistory_.canRedo();
        end

        function log = getHistory(obj)
            % GETHISTORY  Return ordered cell array of command descriptions.
            %   log{1} = oldest, log{end} = most recent.
            log = obj.CommandHistory_.getLog();
        end

        function clearHistory(obj)
            % CLEARHISTORY  Empty the command history (e.g. after loading).
            obj.CommandHistory_.clear();
        end
    end

    % ================================================================== %
    % --- Forward Pass ------------------------------------------------- %
    % ================================================================== %
    methods
        function A = forwardPass(obj, X)
            % FORWARDPASS  Compute activations through all active nodes.
            %
            %   X — (N × d_in) input matrix (N samples, d_in features)
            %   A — (N × d_out) output matrix
            %
            %   Updates NNNode.PreActivation_ and NNNode.Value_ for every
            %   active node at every layer (mean over the batch for display).
            %   Fires ActivationsUpdated event on completion.
            %
            %   DISABLED NODES are excluded from the weight matrix assembly.
            %   Their contribution is simply zeroed out in W.

            nL = obj.numLayers();

            % Work in column-major: A_col is (d × N)
            A_col = X';  % (d_in × N)

            % Update input layer values (mean over batch)
            inNodes = obj.Layers_{1}.getActiveNodes();
            for ni = 1:numel(inNodes)
                if ni <= size(A_col, 1)
                    inNodes{ni}.PreActivation_ = mean(A_col(ni,:));
                    inNodes{ni}.Value_         = mean(A_col(ni,:));
                end
            end

            for L = 2:nL
                layer     = obj.Layers_{L};
                prevLayer = obj.Layers_{L-1};

                activeFrom = prevLayer.getActiveNodes();
                activeTo   = layer.getActiveNodes();
                dOut       = numel(activeTo);

                if dOut == 0
                    % Entire layer disabled — propagate zero tensor
                    A_col = zeros(0, size(A_col, 2));
                    continue;
                end

                % Build reduced weight matrix (active nodes only)
                dIn = numel(activeFrom);
                W   = zeros(dOut, dIn);
                b   = zeros(dOut, 1);

                for i = 1:dOut
                    b(i) = activeTo{i}.Bias_;
                    for j = 1:dIn
                        % Find the edge fromNode=activeFrom{j}, toNode=activeTo{i}
                        w_ij = obj.findWeight_(activeFrom{j}, activeTo{i});
                        W(i, j) = w_ij;
                    end
                end

                % Subset A_col to active-from columns (active input nodes might
                % be a subset of the previous layer's full width)
                fromIdx = cellfun(@(n) double(n.NodeIndex), activeFrom);
                if size(A_col, 1) ~= dIn
                    % Previous layer was also partially disabled — A_col already
                    % has reduced rows matching active-from count. Use as is.
                    A_prev = A_col;
                else
                    A_prev = A_col(fromIdx, :);
                end

                % Pre-activation: Z = W * A_prev + b  (dOut × N)
                N = size(A_prev, 2);
                if N == 0, N = 1; end
                Z = W * A_prev + repmat(b, 1, N);

                % Activation function (applied at the layer level)
                act = lower(obj.Spec_.Activations{L});
                switch act
                    case 'relu'
                        A_col = max(0, Z);
                    case 'sigmoid'
                        A_col = 1 ./ (1 + exp(-Z));
                    case 'tanh'
                        A_col = tanh(Z);
                    case 'linear'
                        A_col = Z;
                    case 'softmax'
                        % Numerically stable: subtract max per column
                        expZ  = exp(Z - max(Z, [], 1));
                        A_col = expZ ./ sum(expZ, 1);
                    otherwise
                        error('OceanDL:Graph:NNGraph:UnknownActivation', ...
                            'Layer %d: unknown activation "%s".', L, act);
                end

                % Store mean values on nodes (for visualization)
                for i = 1:dOut
                    activeTo{i}.PreActivation_ = mean(Z(i,:));
                    activeTo{i}.Value_         = mean(A_col(i,:));
                end
            end

            A = A_col';  % Return (N × d_out)
            notify(obj, 'ActivationsUpdated');
        end
    end

    % ================================================================== %
    % --- Phase 6: Snapshot & Weight Diagnostics ----------------------- %
    % ================================================================== %
    methods
        function memento = saveSnapshot(obj)
            % SAVESNAPSHOT  Capture a full parameter snapshot of this graph.
            %
            % RETURNS:
            %   memento - NNGraphMemento value object (immutable)
            memento = OceanDL.Graph.NNGraphMemento(obj);
        end

        function loadSnapshot(obj, memento)
            % LOADSNAPSHOT  Restore graph parameters from a saved snapshot.
            %
            % INPUTS:
            %   memento - NNGraphMemento returned by saveSnapshot()
            %
            % Restores: edge weights, node biases, node enable/disable masks.
            % Fires WeightsChanged + TopologyChanged when done.
            if ~isa(memento, 'OceanDL.Graph.NNGraphMemento')
                error('OceanDL:Graph:NNGraph:BadMemento', ...
                    'loadSnapshot expects an NNGraphMemento object.');
            end
            if ~memento.isCompatible(obj)
                error('OceanDL:Graph:NNGraph:IncompatibleSnapshot', ...
                    'Snapshot topology does not match current graph.');
            end

            nL = double(memento.NumLayers);

            % Restore edge weights
            for L = 1:nL-1
                obj.setWeightMatrix(L, memento.Weights{L});
            end

            % Restore biases + enable states
            for L = 1:nL
                layer = obj.Layers_{L};
                nodes = layer.getNodes();
                for ni = 1:numel(nodes)
                    nodes{ni}.Bias_ = memento.Biases{L}(ni);
                    if memento.EnableMasks{L}(ni)
                        nodes{ni}.enable();
                    else
                        nodes{ni}.disable();
                    end
                end
            end

            notify(obj, 'WeightsChanged');
            notify(obj, 'TopologyChanged');
        end

        function data = getWeightHistData(obj, options)
            % GETWEIGHTHISTDATA  Collect weight distributions for all layer pairs.
            %
            % Name-Value:
            %   Snapshot (NNGraphMemento, optional) — if provided, the
            %       returned struct includes a .SnapWeights field for
            %       before/after comparison in WeightHistogramStrategy.
            %
            % RETURNS:
            %   data.NumPairs    - number of layer-to-layer pairs
            %   data.Labels      - cell of 'L1→L2 (relu)' strings
            %   data.Weights     - cell of flat edge weight vectors
            %   data.SnapWeights - cell of snapshot weight vectors (if Snapshot provided)
            if nargin < 2, options = struct(); end
            hasSnap = isfield(options, 'Snapshot') && ...
                      ~isempty(options.Snapshot) && ...
                      isa(options.Snapshot, 'OceanDL.Graph.NNGraphMemento');

            nL = obj.numLayers();
            nP = nL - 1;

            data.NumPairs    = nP;
            data.Labels      = cell(nP, 1);
            data.Weights     = cell(nP, 1);
            if hasSnap
                data.SnapWeights = cell(nP, 1);
            end

            spec = obj.Spec_;
            for p = 1:nP
                W = obj.getWeightMatrix(p);
                data.Weights{p} = W(:);
                data.Labels{p}  = sprintf('L%d→L%d  (%s)', ...
                    p, p+1, spec.Activations{p+1});
                if hasSnap
                    Ws = options.Snapshot.Weights{p};
                    data.SnapWeights{p} = Ws(:);
                end
            end
        end
    end

    % ================================================================== %
    % --- Display ------------------------------------------------------ %
    % ================================================================== %
    methods
        function disp(obj)
            nL = obj.numLayers();
            fprintf('NNGraph  [%d layers, %d total params]\n', ...
                nL, obj.paramCount());
            fprintf('  Spec : %s\n', num2str(obj.Spec_.LayerSizes, '%d '));
            for L = 1:nL
                layer = obj.Layers_{L};
                fprintf('  L%-2d  %-12s  (%s)  %d/%d nodes active\n', ...
                    L, layer.Name, layer.Type, ...
                    layer.numActiveNodes(), layer.numNodes());
            end
            allE    = numel(obj.Edges_);
            activeE = numel(obj.getActiveEdges());
            fprintf('  Edges: %d total, %d active\n', allE, activeE);
        end
    end

    % ================================================================== %
    % --- Private Helpers ---------------------------------------------- %
    % ================================================================== %
    methods (Access = private)
        function obj = NNGraph()
            % Default constructor — use fromSpec() instead.
            obj.Layers_         = {};
            obj.Edges_          = {};
            obj.CommandHistory_ = OceanDL.Commands.CommandHistory();
        end

        function w = findWeight_(obj, fromNode, toNode)
            % FINDWEIGHT_  Retrieve the weight between two node handles.
            %   O(edges) — acceptable for small/medium networks.
            for ei = 1:numel(obj.Edges_)
                e = obj.Edges_{ei};
                if e.FromNode == fromNode && e.ToNode == toNode
                    w = e.Weight_;
                    return;
                end
            end
            w = 0;  % Edge deactivated or doesn't exist — contribute zero
        end
    end

    % Friend access — visible ONLY to the listed Command classes
    methods (Access = {?OceanDL.Commands.SwapNodesCommand, ?OceanDL.Commands.SetWeightCommand, ?OceanDL.Commands.SetBiasCommand})

        function swapNodes_(obj, idA, idB)
            % SWAPNODES_  Raw swap implementation — called by SwapNodesCommand.
            %   (not recorded — the command itself handles history)
            nodeA = obj.getNodeByID(idA);
            nodeB = obj.getNodeByID(idB);

            if nodeA.LayerIndex ~= nodeB.LayerIndex
                error('OceanDL:Graph:NNGraph:InvalidSwap', ...
                    'swapNodes requires both nodes in the same layer. Got L%d and L%d.', ...
                    nodeA.LayerIndex, nodeB.LayerIndex);
            end

            % 1. Swap scalar biases
            tmp_bias    = nodeA.Bias_;
            nodeA.Bias_ = nodeB.Bias_;
            nodeB.Bias_ = tmp_bias;

            % 2. Redirect incoming edges (rows of W_L) — two-pass to avoid aliasing
            toA = {}; toB = {};
            for ei = 1:numel(obj.Edges_)
                e = obj.Edges_{ei};
                if e.ToNode == nodeA, toA{end+1} = e; end  %#ok<AGROW>
                if e.ToNode == nodeB, toB{end+1} = e; end  %#ok<AGROW>
            end
            for ei = 1:numel(toA), toA{ei}.ToNode = nodeB; end
            for ei = 1:numel(toB), toB{ei}.ToNode = nodeA; end

            % 3. Redirect outgoing edges (columns of W_{L+1})
            fromA = {}; fromB = {};
            for ei = 1:numel(obj.Edges_)
                e = obj.Edges_{ei};
                if e.FromNode == nodeA, fromA{end+1} = e; end  %#ok<AGROW>
                if e.FromNode == nodeB, fromB{end+1} = e; end  %#ok<AGROW>
            end
            for ei = 1:numel(fromA), fromA{ei}.FromNode = nodeB; end
            for ei = 1:numel(fromB), fromB{ei}.FromNode = nodeA; end

            % 4. Swap NodeIndex + update IDs
            L               = nodeA.LayerIndex;
            tmp_idx         = nodeA.NodeIndex;
            nodeA.NodeIndex = nodeB.NodeIndex;
            nodeB.NodeIndex = tmp_idx;
            nodeA.ID = sprintf("L%d_N%d", L, nodeA.NodeIndex);
            nodeB.ID = sprintf("L%d_N%d", L, nodeB.NodeIndex);

            % 5. Update layer cell ordering
            obj.Layers_{L}.swapNodeOrder( ...
                double(nodeA.NodeIndex), double(nodeB.NodeIndex));
        end

        function edge = getEdge_(obj, fromID, toID)
            % GETEDGE_  Find a specific NNEdge by its node IDs.
            %   Called by SetWeightCommand — returns the handle directly.
            for ei = 1:numel(obj.Edges_)
                e = obj.Edges_{ei};
                if strcmp(e.FromNode.ID, fromID) && strcmp(e.ToNode.ID, toID)
                    edge = e;
                    return;
                end
            end
            error('OceanDL:Graph:NNGraph:EdgeNotFound', ...
                'No edge from "%s" to "%s".', fromID, toID);
        end
    end
end

