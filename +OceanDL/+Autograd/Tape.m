classdef Tape < handle
    % TAPE  Reverse-mode autograd tape for spatial (CNN) modules.
    %
    %   ARCHITECTURE (see project memory / plan for the full research
    %   synthesis behind this design):
    %     - Graph construction is PyTorch-style: nodes are recorded as ops
    %       actually execute during one real forward pass (define-by-run),
    %       NOT a static pre-declared graph and NOT a JAX-style functional
    %       trace. This lets existing buffer-on-self modules (MaxPool2D,
    %       ReLUActivation, the future Conv2D, ...) plug in UNCHANGED --
    %       an op just calls module.forward()/module.backward() and hands
    %       the result to Tape.record().
    %     - The query API is TensorFlow2-GradientTape-style: gradient(target,
    %       sources) accepts ANY watched/recorded node id as both target and
    %       source, not just leaves. This is what Grad-CAM needs (gradient of
    %       the class score w.r.t. a MIDDLE conv layer's output feature map,
    %       not the input), and what PyTorch's default (leaves-only unless
    %       .retain_grad() is called) makes awkward.
    %
    %   NODE MODEL:
    %     Every node has: Id (char), ParentIds (cell of ids consumed to
    %     produce this node), Value (cached forward output -- so a caller
    %     can read any intermediate activation by id, e.g. for activation
    %     maps or Grad-CAM's target feature map), OpName, IsLeaf, and
    %     BackwardFn -- a function handle
    %         parentGrads = BackwardFn(dOut)
    %     that maps the accumulated upstream gradient dOut (same shape as
    %     Value) to a cell array of gradients, one per ParentIds entry, in
    %     the same order. Leaf/"watch" nodes have BackwardFn = [] -- they
    %     are valid gradient SOURCES but propagate nothing further back.
    %
    %   GRADIENT ACCUMULATION:
    %     A node consumed by more than one downstream op (a skip connection,
    %     or a feature map read both by the rest of the forward pass AND by
    %     an interpretability tap) accumulates (sums) the gradient from every
    %     consumer before it is itself propagated further -- the standard
    %     multi-use-tensor accumulation rule.
    %
    %   USAGE:
    %     tape = OceanDL.Autograd.Tape();
    %     xId  = tape.watch(X, 'x');
    %     yId  = tape.record('relu', {xId}, max(0,X), @(dY) {dY .* (X > 0)});
    %     grads = tape.gradient(yId, {xId});     % {dX}, seeded with ones(size(Y))
    %     dX    = tape.gradient(yId, xId);        % non-cell in/out for a single source
    %
    %   A Tape is single-use per forward pass (mirrors "one forward, one
    %   backward" of the Pooling modules' buffer-on-self contract) -- call
    %   reset() before recording a new, unrelated forward pass on the same
    %   object, or simply construct a fresh Tape().
    %
    % See also: OceanDL.Pooling.IPoolingLayer
    %           OceanDL.Conv.CNNEngine

    properties (Access = private)
        Nodes_      = {}   % cell array of node structs, in creation (= topological) order
        IdToIndex_          % containers.Map: char id -> 1-based index into Nodes_
        Counter_    = 0     % monotonically increasing, used to auto-generate ids
    end

    methods
        function obj = Tape()
            obj.reset();
        end

        function reset(obj)
            % RESET  Clear all recorded nodes so the Tape can be reused for
            %   a new, unrelated forward pass.
            obj.Nodes_ = {};
            obj.IdToIndex_ = containers.Map('KeyType', 'char', 'ValueType', 'double');
            obj.Counter_ = 0;
        end

        function id = watch(obj, value, name)
            % WATCH  Register a leaf node (e.g. the input image, or any
            %   value whose gradient you may want later) with no producing
            %   op. Returns its node id.
            %
            %   id = tape.watch(X)          % auto-generated id
            %   id = tape.watch(X, 'input') % explicit, human-readable id
            arguments
                obj
                value
                name (1,:) char = ''
            end
            id = obj.newId(name);
            node = struct('Id', id, 'ParentIds', {{}}, 'Value', value, ...
                'OpName', 'watch', 'IsLeaf', true, 'BackwardFn', []);
            obj.append(node);
        end

        function id = record(obj, opName, parentIds, value, backwardFn, name)
            % RECORD  Register a new node produced by an op that consumed
            %   the nodes in parentIds.
            %
            %   id = tape.record(opName, parentIds, value, backwardFn)
            %   id = tape.record(opName, parentIds, value, backwardFn, name)
            %
            %   parentIds  — cell array of ids previously returned by
            %                watch()/record() on THIS tape.
            %   value      — the op's forward output (cached for later reads).
            %   backwardFn — @(dOut) -> cell array of parent gradients, same
            %                order as parentIds. dOut has the same size as
            %                value.
            arguments
                obj
                opName (1,:) char
                parentIds cell
                value
                backwardFn
                name (1,:) char = ''
            end
            for k = 1:numel(parentIds)
                if ~isKey(obj.IdToIndex_, parentIds{k})
                    error('OceanDL:Autograd:Tape:UnknownParent', ...
                        'record(''%s'', ...): parent id ''%s'' was not found on this tape.', ...
                        opName, parentIds{k});
                end
            end
            id = obj.newId(name);
            node = struct('Id', id, 'ParentIds', {parentIds}, 'Value', value, ...
                'OpName', opName, 'IsLeaf', false, 'BackwardFn', backwardFn);
            obj.append(node);
        end

        function val = getValue(obj, id)
            % GETVALUE  Return the cached forward-pass value for node id.
            val = obj.Nodes_{obj.indexOf(id)}.Value;
        end

        function tf = hasNode(obj, id)
            tf = isKey(obj.IdToIndex_, id);
        end

        function n = numNodes(obj)
            n = numel(obj.Nodes_);
        end

        function grads = gradient(obj, targetId, sourceIds, seed)
            % GRADIENT  Reverse-mode walk from targetId back to sourceIds.
            %
            %   grads = tape.gradient(targetId, sourceIds)
            %   grads = tape.gradient(targetId, sourceIds, seed)
            %
            %   sourceIds — a single id (char) or a cell array of ids.
            %   seed      — upstream gradient seed for targetId, default
            %               ones(size(targetValue)) (standard "scalar loss"
            %               convention generalizes elementwise to any shape).
            %
            %   Returns a value matching sourceIds' shape: a single array if
            %   sourceIds was a single id, otherwise a cell array in the same
            %   order. A source not reached by targetId (the target simply
            %   does not depend on it) returns zeros(size(sourceValue)),
            %   matching the mathematical fact that its gradient is zero.
            arguments
                obj
                targetId (1,:) char
                sourceIds
                seed = []
            end
            targetIdx = obj.indexOf(targetId);
            targetNode = obj.Nodes_{targetIdx};
            if isempty(seed)
                seed = ones(size(targetNode.Value));
            end

            gradMap = containers.Map('KeyType', 'char', 'ValueType', 'any');
            gradMap(targetId) = seed;

            for idx = targetIdx : -1 : 1
                node = obj.Nodes_{idx};
                if ~isKey(gradMap, node.Id)
                    continue  % not an ancestor of targetId -- nothing flows through it
                end
                if node.IsLeaf || isempty(node.BackwardFn)
                    continue  % leaf: valid gradient source, nothing further to propagate
                end

                dOut = gradMap(node.Id);
                parentGrads = node.BackwardFn(dOut);
                if ~iscell(parentGrads)
                    parentGrads = {parentGrads};
                end
                if numel(parentGrads) ~= numel(node.ParentIds)
                    error('OceanDL:Autograd:Tape:BackwardArityMismatch', ...
                        'BackwardFn for op ''%s'' (node ''%s'') returned %d gradient(s), expected %d (one per parent).', ...
                        node.OpName, node.Id, numel(parentGrads), numel(node.ParentIds));
                end

                for p = 1:numel(node.ParentIds)
                    pid = node.ParentIds{p};
                    g = parentGrads{p};
                    if isKey(gradMap, pid)
                        gradMap(pid) = gradMap(pid) + g;
                    else
                        gradMap(pid) = g;
                    end
                end
            end

            wasCell = iscell(sourceIds);
            if ~wasCell
                sourceIds = {sourceIds};
            end

            grads = cell(size(sourceIds));
            for k = 1:numel(sourceIds)
                sid = sourceIds{k};
                srcIdx = obj.indexOf(sid);
                if isKey(gradMap, sid)
                    grads{k} = gradMap(sid);
                else
                    grads{k} = zeros(size(obj.Nodes_{srcIdx}.Value));
                end
            end
            if ~wasCell
                grads = grads{1};
            end
        end
    end

    methods (Access = private)
        function id = newId(obj, name)
            obj.Counter_ = obj.Counter_ + 1;
            if isempty(name)
                id = sprintf('n%d', obj.Counter_);
            else
                id = name;
            end
            if isKey(obj.IdToIndex_, id)
                error('OceanDL:Autograd:Tape:DuplicateId', ...
                    'Node id ''%s'' already exists on this tape (ids must be unique -- call reset() to start a new forward pass).', id);
            end
        end

        function append(obj, node)
            obj.Nodes_{end+1} = node;
            obj.IdToIndex_(node.Id) = numel(obj.Nodes_);
        end

        function idx = indexOf(obj, id)
            if ~isKey(obj.IdToIndex_, id)
                error('OceanDL:Autograd:Tape:UnknownNode', ...
                    'Node id ''%s'' was not found on this tape.', id);
            end
            idx = obj.IdToIndex_(id);
        end
    end
end
