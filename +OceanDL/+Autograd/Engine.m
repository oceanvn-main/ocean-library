classdef Engine
    % ENGINE  Reverse-mode traversal of the Tensor graph (PyTorch's autograd engine).
    %
    %   backward(root)  -- the chain rule, organised so that every node's
    %   vector-Jacobian product runs exactly ONCE, after ALL gradient
    %   contributions to its output have been summed:
    %
    %   1. DISCOVER the sub-graph that can carry gradient to root: an
    %      ITERATIVE depth-first walk over GradFn -> Inputs -> GradFn ...
    %      (iterative, not recursive, so an RNN unrolled over thousands of
    %      steps cannot hit MATLAB's recursion limit).
    %   2. ORDER: Function.Id is a creation counter, and an op is always
    %      created after the ops that produced its inputs, so decreasing Id
    %      is a valid reverse topological order -- no explicit sort of edges.
    %   3. SWEEP: for each node, g = sum of the gradients flowing into its
    %      output (multi-use accumulation: a weight used at every time step
    %      of an RNN receives the SUM over steps -- this IS backpropagation
    %      through time), call its VJP, reduce each input gradient to the
    %      input's shape (Broadcast.reduceTo) and route it:
    %        leaf tensor      -> accumulate into .Grad   (PyTorch: .grad +=)
    %        non-leaf tensor  -> add to its producer's pending gradient
    %   4. RELEASE saved buffers unless RetainGraph (PyTorch default).
    %
    %   grad(outputs, inputs) -- same sweep, but returns dL/d(inputs) for any
    %   tensors (leaf OR intermediate) without touching any .Grad
    %   (torch.autograd.grad; what Grad-CAM needs).
    %
    %   LIMITATION (documented, not hidden): VJPs operate on raw arrays, so
    %   the backward pass is not itself recorded -- first derivatives only
    %   (no create_graph / Hessian-vector products).

    methods (Static)
        function backward(root, seed, opts)
            arguments
                root OceanDL.Autograd.Tensor
                seed = []
                opts.RetainGraph (1,1) logical = false
            end
            OceanDL.Autograd.Engine.run(root, seed, opts.RetainGraph, {});
        end

        function grads = grad(output, inputs, opts)
            % GRAD  dL/d(inputs) without accumulating into .Grad.
            %   inputs: a Tensor or a cell array of Tensors. Returns the same
            %   container shape; unreachable inputs get zeros.
            arguments
                output OceanDL.Autograd.Tensor
                inputs
                opts.Seed = []
                opts.RetainGraph (1,1) logical = false
            end
            wasCell = iscell(inputs);
            if ~wasCell
                inputs = {inputs};
            end
            grads = OceanDL.Autograd.Engine.run(output, opts.Seed, opts.RetainGraph, inputs);
            if ~wasCell
                grads = grads{1};
            end
        end
    end

    methods (Static, Access = private)
        function captured = run(root, seed, retainGraph, targets)
            import OceanDL.Autograd.Broadcast
            capturing = ~isempty(targets);
            captured = cell(size(targets));
            for k = 1:numel(targets)
                captured{k} = zeros(size(targets{k}.Data), 'like', targets{k}.Data);
            end

            if ~root.RequiresGrad
                error('OceanDL:Autograd:Engine:NoGrad', ...
                    'Tensor does not require grad and has no GradFn (was it computed under GradMode.disable, or from constants only?).');
            end
            if isempty(seed)
                if numel(root.Data) ~= 1
                    error('OceanDL:Autograd:Engine:NonScalarRoot', ...
                        'backward() of a non-scalar tensor (size [%s]) needs an explicit seed (dL/dRoot).', ...
                        num2str(size(root.Data)));
                end
                seed = ones(size(root.Data), 'like', root.Data);
            end

            % Root is itself a leaf (e.g. backward on a Parameter): trivial.
            if root.IsLeaf
                [captured, handled] = OceanDL.Autograd.Engine.deliver(root, seed, targets, captured, capturing);
                if ~handled && ~capturing
                    root.accumulateGrad(seed);
                end
                return;
            end

            % 1. Discover (iterative DFS).
            nodes = containers.Map('KeyType', 'double', 'ValueType', 'any');
            stack = {root.GradFn};
            while ~isempty(stack)
                fn = stack{end};
                stack(end) = [];
                if isKey(nodes, fn.Id)
                    continue;
                end
                if fn.Released
                    error('OceanDL:Autograd:Engine:GraphReleased', ...
                        ['Trying to backward through the graph a second time (node %s #%d). ', ...
                         'Its saved buffers were freed; pass ''RetainGraph'', true to the first backward().'], ...
                        fn.opName(), fn.Id);
                end
                nodes(fn.Id) = fn;
                for k = 1:numel(fn.Inputs)
                    t = fn.Inputs{k};
                    if fn.NeedsInputGrad(k) && ~t.IsLeaf
                        stack{end+1} = t.GradFn; %#ok<AGROW>
                    end
                end
            end

            % 2. Order: decreasing creation Id.
            ids = sort(cell2mat(keys(nodes)), 'descend');

            % 3. Sweep.
            pending = containers.Map('KeyType', 'double', 'ValueType', 'any');
            pending(root.GradFn.Id) = seed;
            if root.RetainsGrad && ~capturing
                root.accumulateGrad(seed);
            end
            for id = ids
                if ~isKey(pending, id)
                    continue;
                end
                fn = nodes(id);
                g = pending(id);
                remove(pending, id);

                % A captured intermediate is complete once its producer is reached.
                if capturing
                    for k = 1:numel(targets)
                        if ~targets{k}.IsLeaf && targets{k}.GradFn.Id == fn.Id
                            captured{k} = captured{k} + g;
                        end
                    end
                end

                inGrads = fn.backward(g);
                for k = 1:numel(fn.Inputs)
                    if ~fn.NeedsInputGrad(k) || isempty(inGrads{k})
                        continue;
                    end
                    t = fn.Inputs{k};
                    gk = Broadcast.reduceTo(inGrads{k}, size(t.Data));
                    if t.IsLeaf
                        [captured, handled] = OceanDL.Autograd.Engine.deliver(t, gk, targets, captured, capturing);
                        if ~capturing && ~handled
                            t.accumulateGrad(gk);
                        end
                    else
                        pid = t.GradFn.Id;
                        if isKey(pending, pid)
                            pending(pid) = pending(pid) + gk;
                        else
                            pending(pid) = gk;
                        end
                        if t.RetainsGrad && ~capturing
                            t.accumulateGrad(gk);
                        end
                    end
                end
                if ~retainGraph
                    fn.release();
                end
            end
        end

        function [captured, handled] = deliver(t, g, targets, captured, capturing)
            % Route a leaf gradient to grad()'s capture list (if requested).
            handled = false;
            if ~capturing
                return;
            end
            for k = 1:numel(targets)
                if targets{k}.Uid == t.Uid
                    captured{k} = captured{k} + g;
                    handled = true;
                end
            end
        end
    end
end
