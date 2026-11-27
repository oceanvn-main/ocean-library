classdef Abs < OceanDL.Autograd.Function
    % ABS  Y = |A|
    %
    %   VJP:  dL/dA = g .* sign(A)   (0 at A = 0)

    methods
        function Y = forward(obj, A)
            obj.Saved.S = sign(A);
            Y = abs(A);
        end

        function grads = backward(obj, g)
            grads = {g .* obj.Saved.S};
        end
    end
end
