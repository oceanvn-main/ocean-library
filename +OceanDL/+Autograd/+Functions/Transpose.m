classdef Transpose < OceanDL.Autograd.Function
    % TRANSPOSE  Y = A.'  (2-D)
    %
    %   VJP:  dL/dA = g.'

    methods
        function Y = forward(obj, A)
            Y = A.';
        end

        function grads = backward(obj, g)
            grads = {g.'};
        end
    end
end
