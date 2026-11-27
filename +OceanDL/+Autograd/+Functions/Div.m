classdef Div < OceanDL.Autograd.Function
    % DIV  Y = A ./ B
    %
    %   VJP:  dL/dA = g ./ B,  dL/dB = -g .* A ./ B.^2 = -g .* Y ./ B

    methods
        function Y = forward(obj, A, B)
            Y = A ./ B;
            obj.Saved.B = B;
            obj.Saved.Y = Y;
        end

        function grads = backward(obj, g)
            grads = {g ./ obj.Saved.B, -g .* obj.Saved.Y ./ obj.Saved.B};
        end
    end
end
