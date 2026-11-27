classdef Neg < OceanDL.Autograd.Function
    % NEG  Y = -A
    %
    %   VJP:  dL/dA = -g

    methods
        function Y = forward(obj, A)
            Y = -A;
        end

        function grads = backward(obj, g)
            grads = {-g};
        end
    end
end
