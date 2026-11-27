classdef Mean < OceanDL.Autograd.Function
    % MEAN  Y = mean(A, dim)   (dim = [] -> first non-singleton dim; 'all' -> everything)
    %
    %   VJP:  dL/dA = g / n  expanded to size(A),  n = number of averaged elements.

    properties (SetAccess = private)
        Dim = []
    end

    methods
        function obj = Mean(dim)
            if nargin > 0
                obj.Dim = dim;
            end
        end

        function Y = forward(obj, A)
            d = OceanDL.Autograd.Functions.Sum.resolveDim(obj.Dim, A);
            Y = mean(A, d);
            obj.Saved.InSize = size(A);
            obj.Saved.Like = A([]);
            obj.Saved.N = numel(A) / numel(Y);
        end

        function grads = backward(obj, g)
            grads = {g / obj.Saved.N + zeros(obj.Saved.InSize, 'like', obj.Saved.Like)};
        end
    end
end
