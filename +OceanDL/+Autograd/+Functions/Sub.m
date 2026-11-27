classdef Sub < OceanDL.Autograd.Function
    % SUB  Y = A - B
    %
    %   VJP:  dL/dA = g,  dL/dB = -g

    methods
        function Y = forward(obj, A, B)
            Y = A - B;
        end

        function grads = backward(obj, g)
            grads = {g, -g};
        end
    end
end
