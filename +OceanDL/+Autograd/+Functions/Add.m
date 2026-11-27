classdef Add < OceanDL.Autograd.Function
    % ADD  Y = A + B  (implicit expansion allowed)
    %
    %   VJP:  dL/dA = g,  dL/dB = g   (Engine sums broadcast copies back)

    methods
        function Y = forward(obj, A, B)
            Y = A + B;
        end

        function grads = backward(obj, g)
            grads = {g, g};
        end
    end
end
