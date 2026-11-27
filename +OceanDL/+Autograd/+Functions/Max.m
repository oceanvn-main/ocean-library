classdef Max < OceanDL.Autograd.Function
    % MAX  Y = max(A, [], dim)   (reduction; max-pooling over time, etc.)
    %
    %   VJP: only the winning element influenced Y, so the gradient is
    %   routed to the argmax position and is zero elsewhere:
    %     dL/dA = g .* [k == argmax]    along dim   (ties -> first index, as MATLAB)
    %   The one-hot mask is built by comparing an index ramp 1..n (laid
    %   along dim) with the argmax indices -- no loops, any rank.

    properties (SetAccess = private)
        Dim (1,1) double
        ArgMax = []          % indices along Dim from the last forward()
    end

    methods
        function obj = Max(dim)
            obj.Dim = dim;
        end

        function Y = forward(obj, A)
            [Y, obj.ArgMax] = max(A, [], obj.Dim);
            n = size(A, obj.Dim);
            ramp = reshape(1:n, [ones(1, obj.Dim - 1), n, 1]);
            obj.Saved.Mask = (ramp == obj.ArgMax);
        end

        function grads = backward(obj, g)
            grads = {g .* obj.Saved.Mask};
        end
    end
end
