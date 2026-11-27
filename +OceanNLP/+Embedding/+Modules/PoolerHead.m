classdef PoolerHead
    % POOLERHEAD  BERT pooler: p = tanh(W h_[CLS] + b).
    %
    % Reads the FIRST column (the [CLS] position) of a (d x T) hidden-state
    % matrix and returns a (d x 1) vector. W is (d x d), PyTorch layout,
    % used as-is (see LinearHead).
    %
    % See also: OceanNLP.Embedding.Modules.LinearHead, OceanNLP.Embedding.CrossEncoder

    properties (SetAccess = private)
        W (:,:) double = []
        b (:,1) double = []
    end

    methods
        function obj = PoolerHead(W, b)
            arguments
                W (:,:) double
                b (:,1) double
            end
            if size(W, 1) ~= size(W, 2) || numel(b) ~= size(W, 1)
                error('OceanNLP:Embedding:Modules:PoolerHead:ShapeMismatch', ...
                    'Pooler needs a square W (d x d) and b of length d. Got W %dx%d, b %d.', ...
                    size(W, 1), size(W, 2), numel(b));
            end
            obj.W = W;
            obj.b = b;
        end

        function p = forward(obj, X)
            arguments
                obj
                X (:,:) double
            end
            if size(X, 1) ~= size(obj.W, 2) || size(X, 2) < 1
                error('OceanNLP:Embedding:Modules:PoolerHead:InDimMismatch', ...
                    'X must be (%d x T) with T >= 1. Got %dx%d.', size(obj.W, 2), size(X, 1), size(X, 2));
            end
            p = tanh(obj.W * X(:, 1) + obj.b);
        end

        function meta = getMetadata(~)
            meta = struct();
            meta.SymbolicLatex = "\mathbf{p} = \tanh(W\,\mathbf{h}_{\mathrm{[CLS]}} + \mathbf{b})";
            meta.Symbols = containers.Map();
            meta.Symbols('h_{[CLS]}') = "final hidden state at the [CLS] position";
            meta.Symbols('W') = "pooler weight (d x d)";
            meta.Symbols('b') = "pooler bias (d x 1)";
            meta.Assumptions = {'reads only column 1 ([CLS]) of the hidden-state matrix'};
            meta.IsSymbolic = true;
            meta.Reference = "BERT pooler (Devlin et al., 2018)";
        end

        function spec = getPipelineSpec(~)
            spec = struct();
            spec.Inputs = struct('Name', "X", 'Domain', "token-state-matrix", 'Dtype', "double", 'Shape', "d x T");
            spec.Outputs = struct('Name', "p", 'Domain', "dense-vector-batch", 'Dtype', "double", 'Shape', "d x 1");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "pooler");
            spec.ComposesFrom = {};
        end
    end
end
