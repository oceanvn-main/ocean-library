classdef Pow < OceanDL.Autograd.Function
    % POW  Y = A .^ p   for a CONSTANT exponent p
    %
    %   VJP:  dL/dA = g .* p .* A.^(p-1)

    properties (SetAccess = private)
        Exponent (1,1) double
    end

    methods
        function obj = Pow(p)
            obj.Exponent = p;
        end

        function Y = forward(obj, A)
            obj.Saved.A = A;
            Y = A .^ obj.Exponent;
        end

        function grads = backward(obj, g)
            p = obj.Exponent;
            grads = {g .* p .* obj.Saved.A .^ (p - 1)};
        end
    end
end
