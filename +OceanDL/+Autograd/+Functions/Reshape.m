classdef Reshape < OceanDL.Autograd.Function
    % RESHAPE  Y = reshape(A, sz)   (column-major, MATLAB's element order)
    %
    %   VJP:  dL/dA = reshape(g, size(A))  -- reshaping only relabels
    %   positions, every element keeps its value and its gradient.

    properties (SetAccess = private)
        OutSize (1,:) double
    end

    methods
        function obj = Reshape(sz)
            obj.OutSize = sz;
        end

        function Y = forward(obj, A)
            obj.Saved.InSize = size(A);
            Y = reshape(A, obj.OutSize);
        end

        function grads = backward(obj, g)
            grads = {reshape(g, obj.Saved.InSize)};
        end
    end
end
