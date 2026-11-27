classdef Sqrt < OceanDL.Autograd.Function
    % SQRT  Y = sqrt(A)
    %
    %   VJP:  dL/dA = g ./ (2 sqrt(A)) = g ./ (2 Y)

    methods
        function Y = forward(obj, A)
            Y = sqrt(A);
            obj.Saved.Y = Y;
        end

        function grads = backward(obj, g)
            grads = {g ./ (2 * obj.Saved.Y)};
        end
    end
end
