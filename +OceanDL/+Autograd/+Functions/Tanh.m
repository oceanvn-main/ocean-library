classdef Tanh < OceanDL.Autograd.Function
    % TANH  Y = tanh(A)
    %
    %   VJP:  dL/dA = g .* (1 - Y.^2)       (tanh' = 1 - tanh^2)

    methods
        function Y = forward(obj, A)
            Y = tanh(A);
            obj.Saved.Y = Y;
        end

        function grads = backward(obj, g)
            grads = {g .* (1 - obj.Saved.Y.^2)};
        end
    end
end
