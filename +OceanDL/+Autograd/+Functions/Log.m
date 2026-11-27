classdef Log < OceanDL.Autograd.Function
    % LOG  Y = log(A)
    %
    %   VJP:  dL/dA = g ./ A

    methods
        function Y = forward(obj, A)
            obj.Saved.A = A;
            Y = log(A);
        end

        function grads = backward(obj, g)
            grads = {g ./ obj.Saved.A};
        end
    end
end
