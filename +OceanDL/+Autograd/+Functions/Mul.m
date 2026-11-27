classdef Mul < OceanDL.Autograd.Function
    % MUL  Y = A .* B  (elementwise, implicit expansion allowed)
    %
    %   VJP:  dL/dA = g .* B,  dL/dB = g .* A

    methods
        function Y = forward(obj, A, B)
            obj.Saved.A = A;
            obj.Saved.B = B;
            Y = A .* B;
        end

        function grads = backward(obj, g)
            grads = {g .* obj.Saved.B, g .* obj.Saved.A};
        end
    end
end
