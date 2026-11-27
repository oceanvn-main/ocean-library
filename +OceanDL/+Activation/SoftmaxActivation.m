classdef SoftmaxActivation < OceanDL.Activation.IActivation
    % SOFTMAXACTIVATION  Numerically stable softmax over the neuron dimension.
    %
    %   Converts a vector of raw logits into a probability distribution.
    %   Applied column-wise: each column of Z is one sample.
    %   Subtracts the per-column max before exponentiation (numerically stable).
    %
    %   grad() returns the diagonal of the Jacobian per column:
    %     d_softmax_i / d_z_i = A_i * (1 - A_i)
    %   This is the term needed in backprop when using cross-entropy loss
    %   (the full Jacobian is never needed — CE gradient combines to (A - Y)).
    %
    %   Usage:
    %     act = OceanDL.Activation.SoftmaxActivation();
    %     A   = act.forward(Z);   % (d x N), columns sum to 1
    %     dA  = act.grad(Z);      % diagonal Jacobian, same shape

    methods
        function A = forward(~, Z)
            % Subtract column max for numerical stability
            Z_stable = Z - max(Z, [], 1);
            expZ = exp(Z_stable);
            A    = expZ ./ sum(expZ, 1);
        end

        function dA = grad(~, Z)
            % Diagonal Jacobian: A_i * (1 - A_i)
            Z_stable = Z - max(Z, [], 1);
            expZ = exp(Z_stable);
            A    = expZ ./ sum(expZ, 1);
            dA   = A .* (1 - A);
        end

        function s = name(~)
            s = 'softmax';
        end
    end
end
