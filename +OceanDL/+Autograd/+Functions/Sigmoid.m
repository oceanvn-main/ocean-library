classdef Sigmoid < OceanDL.Autograd.Function
    % SIGMOID  Y = 1 ./ (1 + exp(-A))
    %
    %   VJP:  dL/dA = g .* Y .* (1 - Y)     (sigma' = sigma (1 - sigma))
    %   exp(-A) may overflow to Inf for A << 0; then Y = 0 exactly, which is the correct limit.

    methods
        function Y = forward(obj, A)
            Y = 1 ./ (1 + exp(-A));
            obj.Saved.Y = Y;
        end

        function grads = backward(obj, g)
            grads = {g .* obj.Saved.Y .* (1 - obj.Saved.Y)};
        end
    end
end
