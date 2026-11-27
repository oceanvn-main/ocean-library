classdef Exp < OceanDL.Autograd.Function
    % EXP  Y = exp(A)
    %
    %   VJP:  dL/dA = g .* exp(A) = g .* Y

    methods
        function Y = forward(obj, A)
            Y = exp(A);
            obj.Saved.Y = Y;
        end

        function grads = backward(obj, g)
            grads = {g .* obj.Saved.Y};
        end
    end
end
