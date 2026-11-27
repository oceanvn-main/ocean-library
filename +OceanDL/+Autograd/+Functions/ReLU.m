classdef ReLU < OceanDL.Autograd.Function
    % RELU  Y = max(A, 0)
    %
    %   VJP:  dL/dA = g .* [A > 0]   (subgradient 0 chosen at A = 0, as in PyTorch)

    methods
        function Y = forward(obj, A)
            obj.Saved.Mask = A > 0;
            Y = A .* obj.Saved.Mask;
        end

        function grads = backward(obj, g)
            grads = {g .* obj.Saved.Mask};
        end
    end
end
