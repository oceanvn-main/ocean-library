classdef Permute < OceanDL.Autograd.Function
    % PERMUTE  Y = permute(A, order)
    %
    %   VJP:  dL/dA = ipermute(g, order)   (the inverse axis permutation)

    properties (SetAccess = private)
        Order (1,:) double
    end

    methods
        function obj = Permute(order)
            obj.Order = order;
        end

        function Y = forward(obj, A)
            Y = permute(A, obj.Order);
        end

        function grads = backward(obj, g)
            grads = {ipermute(g, obj.Order)};
        end
    end
end
