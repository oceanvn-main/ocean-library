classdef LinearHead
    % LINEARHEAD  Affine map y = W x + b on feature-major column vectors.
    %
    % Used for the cross-encoder classifier (1 x d, with bias) and the
    % ColBERT projection (dim x d, NO bias). W is the PyTorch nn.Linear
    % weight (out, in) used AS-IS: y_col = W * x_col + b is algebraically
    % the same as PyTorch's y_row = x_row * W' + b (no transpose crosses the
    % boundary, same rule as MultiHeadAttention).
    %
    %   Y = head.forward(X)     X (in x T)  ->  Y (out x T)
    %
    % See also: OceanNLP.Embedding.Modules.PoolerHead

    properties (SetAccess = private)
        W (:,:) double = []
        b (:,1) double = []      % empty = no bias
    end

    properties (Dependent, SetAccess = private)
        InDim
        OutDim
        HasBias
    end

    methods
        function obj = LinearHead(W, b)
            arguments
                W (:,:) double
                b (:,1) double = zeros(0, 1)
            end
            if ~isempty(b) && numel(b) ~= size(W, 1)
                error('OceanNLP:Embedding:Modules:LinearHead:BiasSizeMismatch', ...
                    'b has %d entries but W has %d output rows.', numel(b), size(W, 1));
            end
            obj.W = W;
            obj.b = b;
        end

        function Y = forward(obj, X)
            arguments
                obj
                X (:,:) double
            end
            if size(X, 1) ~= obj.InDim
                error('OceanNLP:Embedding:Modules:LinearHead:InDimMismatch', ...
                    'X has %d rows but this head expects %d.', size(X, 1), obj.InDim);
            end
            Y = obj.W * X;
            if obj.HasBias
                Y = Y + obj.b;
            end
        end

        function n = get.InDim(obj),   n = size(obj.W, 2); end
        function n = get.OutDim(obj),  n = size(obj.W, 1); end
        function tf = get.HasBias(obj), tf = ~isempty(obj.b); end

        function meta = getMetadata(obj)
            meta = struct();
            if obj.HasBias
                meta.SymbolicLatex = "\mathbf{y} = W\mathbf{x} + \mathbf{b}";
            else
                meta.SymbolicLatex = "\mathbf{y} = W\mathbf{x}";
            end
            meta.Symbols = containers.Map();
            meta.Symbols('W') = sprintf("weight matrix, %d x %d (out x in), PyTorch nn.Linear layout", obj.OutDim, obj.InDim);
            meta.Symbols('x') = "input column vector";
            if obj.HasBias, meta.Symbols('b') = "bias column vector"; end
            meta.Assumptions = {'feature-major columns; W used as-is with no transpose'};
            meta.IsSymbolic = true;
            meta.Reference = "affine (fully connected) layer";
        end

        function spec = getPipelineSpec(~)
            spec = struct();
            spec.Inputs = struct('Name', "X", 'Domain', "token-state-matrix", 'Dtype', "double", 'Shape', "in x T");
            spec.Outputs = struct('Name', "Y", 'Domain', "token-state-matrix", 'Dtype', "double", 'Shape', "out x T");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "linear");
            spec.ComposesFrom = {};
        end
    end
end
