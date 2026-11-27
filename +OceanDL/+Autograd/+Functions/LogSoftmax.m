classdef LogSoftmax < OceanDL.Autograd.Function
    % LOGSOFTMAX  Y = A - logsumexp(A, dim),  logsumexp computed stably:
    %     logsumexp(A) = m + log(sum(exp(A - m))),  m = max(A)
    %
    %   The numerically safe building block of cross-entropy
    %   (log(softmax(A)) would underflow to log(0) = -Inf for confident logits).
    %   VJP:  dY_i/dA_j = delta_ij - softmax_j
    %     => dL/dA = g - softmax(A) .* sum(g, dim) = g - exp(Y) .* sum(g, dim)

    properties (SetAccess = private)
        Dim (1,1) double
    end

    methods
        function obj = LogSoftmax(dim)
            obj.Dim = dim;
        end

        function Y = forward(obj, A)
            m = max(A, [], obj.Dim);
            Y = A - (m + log(sum(exp(A - m), obj.Dim)));
            obj.Saved.Y = Y;
        end

        function grads = backward(obj, g)
            grads = {g - exp(obj.Saved.Y) .* sum(g, obj.Dim)};
        end
    end
end
