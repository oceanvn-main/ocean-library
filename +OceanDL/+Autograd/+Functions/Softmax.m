classdef Softmax < OceanDL.Autograd.Function
    % SOFTMAX  Y = exp(A - max A) ./ sum(exp(A - max A), dim)
    %
    %   Shifting by the max leaves Y unchanged (it cancels in the ratio) but
    %   keeps exp() from overflowing.
    %   VJP (full Jacobian-vector product, not the diagonal shortcut):
    %     dY_i/dA_j = Y_i (delta_ij - Y_j)
    %     => dL/dA = Y .* (g - sum(g .* Y, dim))

    properties (SetAccess = private)
        Dim (1,1) double
    end

    methods
        function obj = Softmax(dim)
            obj.Dim = dim;
        end

        function Y = forward(obj, A)
            E = exp(A - max(A, [], obj.Dim));
            Y = E ./ sum(E, obj.Dim);
            obj.Saved.Y = Y;
        end

        function grads = backward(obj, g)
            Y = obj.Saved.Y;
            grads = {Y .* (g - sum(g .* Y, obj.Dim))};
        end
    end
end
