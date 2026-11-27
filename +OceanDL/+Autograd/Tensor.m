classdef Tensor < handle & matlab.mixin.indexing.RedefinesParen
    % TENSOR  An array that remembers how it was computed (PyTorch torch.Tensor).
    %
    %   A Tensor wraps a plain numeric array (Data, any shape, single or
    %   double) plus three autograd fields:
    %     RequiresGrad  should gradients flow into / through this tensor?
    %     GradFn        the Function that produced it ([] for a LEAF: an
    %                   input, a constant or a Parameter)
    %     Grad          dL/dTensor, filled by backward() for leaves (and for
    %                   intermediates that called retainGrad())
    %
    %   Ordinary MATLAB syntax builds the graph -- no manual recording:
    %     W = Tensor(randn(4,3), 'RequiresGrad', true);
    %     b = Tensor(zeros(4,1), 'RequiresGrad', true);
    %     x = randn(3, 8);                       % plain arrays are constants
    %     h = tanh(W * x + b);                   % MatMul, Add (broadcast), Tanh
    %     L = mean(h(:) .^ 2);                   % Index, Pow, Mean
    %     L.backward();                          % W.Grad, b.Grad now hold dL/dW, dL/db
    %
    %   SEMANTICS (chosen to be unsurprising in MATLAB AND like PyTorch):
    %     - MATLAB operator meaning is kept: * is matrix product (scalar *
    %       is elementwise), .* is elementwise, sum(x) sums the first
    %       non-singleton dimension, sum(x,'all') sums everything.
    %     - numel/size/end/isempty describe Data (like torch), not the handle.
    %     - Indexing x(i,j), x(:), x(:, ids) is differentiable (repeated
    %       indices accumulate gradient -- embedding lookups rely on this).
    %     - No in-place assignment x(i) = v: it would silently invalidate
    %       buffers saved for backward. Build new tensors with cat() instead.
    %     - Gradients ACCUMULATE across backward() calls until zeroGrad(),
    %       exactly like PyTorch (needed for gradient accumulation / BPTT
    %       over several chunks).
    %     - Precision follows Data: single stays single through the graph.
    %
    % See also: OceanDL.Autograd.Function, OceanDL.Autograd.Engine,
    %           OceanDL.nn.Parameter, OceanDL.Autograd.F

    properties
        Data                                   % the numeric array
        Grad = []                              % dL/dThis (leaves / retained)
        Name char = ''
    end

    properties (SetAccess = protected)
        RequiresGrad (1,1) logical = false
        GradFn = []                            % producing Function, [] for a leaf
        RetainsGrad (1,1) logical = false
        Uid (1,1) double = 0                   % unique identity (handle == is avoided: size is overloaded)
    end

    properties (Dependent)
        IsLeaf
    end

    methods
        function obj = Tensor(data, opts)
            arguments
                data = []
                opts.RequiresGrad (1,1) logical = false
                opts.GradFn = []
                opts.Name char = ''
            end
            if isa(data, 'OceanDL.Autograd.Tensor')
                data = data.Data;
            end
            if ~(isnumeric(data) || islogical(data))
                error('OceanDL:Autograd:Tensor:BadData', ...
                    'Tensor data must be numeric or logical, got %s.', class(data));
            end
            if islogical(data)
                data = double(data);
            end
            obj.Data = data;
            obj.GradFn = opts.GradFn;
            obj.RequiresGrad = opts.RequiresGrad || ~isempty(opts.GradFn);
            obj.Name = opts.Name;
            obj.Uid = OceanDL.Autograd.Function.nextId();
        end

        function tf = get.IsLeaf(obj)
            tf = isempty(obj.GradFn);
        end

        % ------------------------------------------------------------ %
        %  Autograd API
        % ------------------------------------------------------------ %
        function backward(obj, seed, opts)
            % BACKWARD  Accumulate dL/dLeaf into every reachable leaf's Grad.
            %   L.backward()                      scalar L
            %   Y.backward(dLdY)                  non-scalar Y
            %   L.backward([], 'RetainGraph', true)
            arguments
                obj
                seed = []
                opts.RetainGraph (1,1) logical = false
            end
            OceanDL.Autograd.Engine.backward(obj, seed, 'RetainGraph', opts.RetainGraph);
        end

        function requiresGrad(obj, tf)
            % REQUIRESGRAD  Mark a LEAF as trainable (x.requires_grad_()).
            arguments
                obj
                tf (1,1) logical = true
            end
            if ~obj.IsLeaf
                error('OceanDL:Autograd:Tensor:NonLeafRequiresGrad', ...
                    'RequiresGrad can only be set on leaf tensors; use detach() first.');
            end
            obj.RequiresGrad = tf;
        end

        function retainGrad(obj)
            % RETAINGRAD  Keep .Grad on an intermediate (non-leaf) tensor too.
            obj.RetainsGrad = true;
        end

        function t = detach(obj)
            % DETACH  Same data, cut from the graph (stop-gradient).
            t = OceanDL.Autograd.Tensor(obj.Data);
        end

        function zeroGrad(obj)
            obj.Grad = [];
        end

        function accumulateGrad(obj, g)
            % ACCUMULATEGRAD  Grad += g (called by Engine).
            if isempty(obj.Grad)
                obj.Grad = g;
            else
                obj.Grad = obj.Grad + g;
            end
        end

        function v = item(obj)
            if numel(obj.Data) ~= 1
                error('OceanDL:Autograd:Tensor:NotScalar', 'item() needs a one-element tensor.');
            end
            v = obj.Data;
        end

        function v = double(obj)
            v = double(obj.Data);
        end

        function v = single(obj)
            v = single(obj.Data);
        end

        % ------------------------------------------------------------ %
        %  Shape queries (describe Data)
        % ------------------------------------------------------------ %
        function varargout = size(obj, varargin)
            [varargout{1:max(nargout, 1)}] = size(obj.Data, varargin{:});
        end

        function n = ndims(obj)
            n = ndims(obj.Data);
        end

        % ------------------------------------------------------------ %
        %  Arithmetic (MATLAB operator overloads)
        % ------------------------------------------------------------ %
        function y = plus(a, b),    y = OceanDL.Autograd.Functions.Add().apply(a, b); end
        function y = minus(a, b),   y = OceanDL.Autograd.Functions.Sub().apply(a, b); end
        function y = times(a, b),   y = OceanDL.Autograd.Functions.Mul().apply(a, b); end
        function y = rdivide(a, b), y = OceanDL.Autograd.Functions.Div().apply(a, b); end
        function y = uminus(a),     y = OceanDL.Autograd.Functions.Neg().apply(a); end
        function y = uplus(a),      y = a; end

        function y = mtimes(a, b)
            % Scalar * X keeps MATLAB's meaning: elementwise scaling.
            if numel(OceanDL.Autograd.Tensor.raw(a)) == 1 || ...
               numel(OceanDL.Autograd.Tensor.raw(b)) == 1
                y = OceanDL.Autograd.Functions.Mul().apply(a, b);
            else
                y = OceanDL.Autograd.Functions.MatMul().apply(a, b);
            end
        end

        function y = mrdivide(a, b)
            if numel(OceanDL.Autograd.Tensor.raw(b)) ~= 1
                error('OceanDL:Autograd:Tensor:Unsupported', ...
                    'Matrix right division A/B is not differentiable here; use A ./ B or an explicit solve.');
            end
            y = OceanDL.Autograd.Functions.Div().apply(a, b);
        end

        function y = power(a, p)
            if isa(p, 'OceanDL.Autograd.Tensor')
                error('OceanDL:Autograd:Tensor:Unsupported', ...
                    'Only constant exponents are supported: x.^p with numeric p.');
            end
            y = OceanDL.Autograd.Functions.Pow(p).apply(a);
        end

        function y = transpose(a),  y = OceanDL.Autograd.Functions.Transpose().apply(a); end
        function y = ctranspose(a), y = OceanDL.Autograd.Functions.Transpose().apply(a); end

        % Comparisons are not differentiable: they return plain logical arrays.
        function m = gt(a, b), m = OceanDL.Autograd.Tensor.raw(a) >  OceanDL.Autograd.Tensor.raw(b); end
        function m = lt(a, b), m = OceanDL.Autograd.Tensor.raw(a) <  OceanDL.Autograd.Tensor.raw(b); end
        function m = ge(a, b), m = OceanDL.Autograd.Tensor.raw(a) >= OceanDL.Autograd.Tensor.raw(b); end
        function m = le(a, b), m = OceanDL.Autograd.Tensor.raw(a) <= OceanDL.Autograd.Tensor.raw(b); end

        % ------------------------------------------------------------ %
        %  Elementwise math
        % ------------------------------------------------------------ %
        function y = exp(a),     y = OceanDL.Autograd.Functions.Exp().apply(a); end
        function y = log(a),     y = OceanDL.Autograd.Functions.Log().apply(a); end
        function y = sqrt(a),    y = OceanDL.Autograd.Functions.Sqrt().apply(a); end
        function y = tanh(a),    y = OceanDL.Autograd.Functions.Tanh().apply(a); end
        function y = abs(a),     y = OceanDL.Autograd.Functions.Abs().apply(a); end
        function y = sigmoid(a), y = OceanDL.Autograd.Functions.Sigmoid().apply(a); end
        function y = relu(a),    y = OceanDL.Autograd.Functions.ReLU().apply(a); end
        function y = gelu(a),    y = OceanDL.Autograd.Functions.GELU().apply(a); end

        function C = pagemtimes(A, varargin)
            % PAGEMTIMES  Differentiable batched matrix product, same call forms
            %   as the built-in: pagemtimes(A, B) or pagemtimes(A, tA, B, tB)
            %   with tA / tB = 'none' | 'transpose'.
            if numel(varargin) == 1
                B = varargin{1}; tA = 'none'; tB = 'none';
            else
                tA = varargin{1}; B = varargin{2}; tB = varargin{3};
            end
            C = OceanDL.Autograd.Functions.BatchMatMul(strcmpi(tA, 'transpose'), strcmpi(tB, 'transpose')).apply(A, B);
        end

        % ------------------------------------------------------------ %
        %  Reductions and shape ops
        % ------------------------------------------------------------ %
        function y = sum(a, dim)
            if nargin < 2, dim = []; end
            y = OceanDL.Autograd.Functions.Sum(dim).apply(a);
        end

        function y = mean(a, dim)
            if nargin < 2, dim = []; end
            y = OceanDL.Autograd.Functions.Mean(dim).apply(a);
        end

        function [y, idx] = max(a, other, dim)
            % MAX  Reduction max(a, [], dim) only (routes gradient to the argmax).
            if nargin < 3 || ~isempty(other)
                error('OceanDL:Autograd:Tensor:Unsupported', ...
                    'Use max(x, [], dim); elementwise max(a, b) is not supported (use relu(a - b) + b).');
            end
            fn = OceanDL.Autograd.Functions.Max(dim);
            y = fn.apply(a);
            idx = fn.ArgMax;
        end

        function y = reshape(a, varargin)
            if isscalar(varargin)
                sz = varargin{1};
            else
                sz = cellfun(@OceanDL.Autograd.Tensor.sizeArg, varargin, 'UniformOutput', false);
                known = prod([sz{~cellfun(@isempty, sz)}]);
                sz(cellfun(@isempty, sz)) = {numel(a.Data) / known};
                sz = [sz{:}];
            end
            y = OceanDL.Autograd.Functions.Reshape(sz).apply(a);
        end

        function y = permute(a, order)
            y = OceanDL.Autograd.Functions.Permute(order).apply(a);
        end

        function y = softmax(a, dim)
            y = OceanDL.Autograd.Functions.Softmax(dim).apply(a);
        end

        function y = logSoftmax(a, dim)
            y = OceanDL.Autograd.Functions.LogSoftmax(dim).apply(a);
        end

        % ------------------------------------------------------------ %
        %  Concatenation (RedefinesParen requires cat; [a b] and [a; b] use it)
        % ------------------------------------------------------------ %
        function y = cat(dim, varargin)
            y = OceanDL.Autograd.Functions.Cat(dim).apply(varargin{:});
        end

        function y = horzcat(varargin)
            y = cat(2, varargin{:});
        end

        function y = vertcat(varargin)
            y = cat(1, varargin{:});
        end

        % ------------------------------------------------------------ %
        %  Display
        % ------------------------------------------------------------ %
        function disp(obj)
            fn = 'leaf';
            if ~obj.IsLeaf
                fn = ['GradFn=' obj.GradFn.opName()];
            end
            fprintf('  Tensor [%s %s]  RequiresGrad=%d  %s\n', ...
                strjoin(string(size(obj.Data)), 'x'), class(obj.Data), obj.RequiresGrad, fn);
            if numel(obj.Data) <= 64
                disp(obj.Data);
            end
        end
    end

    methods (Access = protected)
        function varargout = parenReference(obj, indexOp)
            y = OceanDL.Autograd.Functions.Index(indexOp(1).Indices).apply(obj);
            if isscalar(indexOp)
                varargout{1} = y;
            else
                [varargout{1:nargout}] = y.(indexOp(2:end));
            end
        end

        function obj = parenAssign(obj, indexOp, varargin) %#ok<INUSD>
            error('OceanDL:Autograd:Tensor:NoInPlace', ...
                ['In-place assignment x(...) = v is not supported on Tensors: it would invalidate ', ...
                 'values saved for backward. Build a new tensor (cat, masks) or edit x.Data outside the graph.']);
        end

        function obj = parenDelete(obj, indexOp) %#ok<INUSD>
            error('OceanDL:Autograd:Tensor:NoInPlace', 'Deleting elements of a Tensor is not supported.');
        end

        function n = parenListLength(~, ~, ~)
            n = 1;
        end
    end

    methods (Static)
        function t = wrap(x)
            % WRAP  Tensor as-is; anything else becomes a constant leaf.
            if isa(x, 'OceanDL.Autograd.Tensor')
                t = x;
            else
                t = OceanDL.Autograd.Tensor(x);
            end
        end

        function v = raw(x)
            % RAW  The numeric data of a Tensor or of a plain array.
            if isa(x, 'OceanDL.Autograd.Tensor')
                v = x.Data;
            else
                v = x;
            end
        end

        function obj = empty()
            obj = OceanDL.Autograd.Tensor([]);
        end
    end

    methods (Static, Access = private)
        function s = sizeArg(x)
            s = OceanDL.Autograd.Tensor.raw(x);
        end
    end
end
