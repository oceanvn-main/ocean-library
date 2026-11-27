classdef Sum < OceanDL.Autograd.Function
    % SUM  Y = sum(A, dim)   (dim = [] -> first non-singleton dim; 'all' -> everything)
    %
    %   VJP: every input element contributed with weight 1 to exactly one
    %   output element, so the gradient is g COPIED back along the summed
    %   dimension(s):  dL/dA = g  expanded to size(A)
    %   (implemented as g + zeros(size(A)), i.e. implicit expansion).

    properties (SetAccess = private)
        Dim = []
    end

    methods
        function obj = Sum(dim)
            if nargin > 0
                obj.Dim = dim;
            end
        end

        function Y = forward(obj, A)
            obj.Saved.InSize = size(A);
            obj.Saved.Like = A([]);
            Y = sum(A, OceanDL.Autograd.Functions.Sum.resolveDim(obj.Dim, A));
        end

        function grads = backward(obj, g)
            grads = {g + zeros(obj.Saved.InSize, 'like', obj.Saved.Like)};
        end
    end

    methods (Static)
        function d = resolveDim(dim, A)
            % MATLAB's default reduction dimension: first non-singleton.
            if isempty(dim)
                d = find(size(A) ~= 1, 1);
                if isempty(d)
                    d = 1;
                end
            else
                d = dim;
            end
        end
    end
end
