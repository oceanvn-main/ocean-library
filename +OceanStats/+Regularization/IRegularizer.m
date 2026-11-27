classdef IRegularizer < handle
    % IREGULARIZER  Abstract interface for model regularization penalties.
    %
    %   Defines the contract for evaluating a penalty r(w), its gradient
    %   (or subgradient), and its proximity operator.
    %   This decouples the choice of regularization (L1, L2, Elastic Net)
    %   from the overall structural risk minimization objective, allowing
    %   composite objectives of the form: f(w) + lambda * r(w).
    %
    %   Author: Ocean Library | 2026-03-13

    methods (Abstract)
        % EVALUATE  Calculate the penalty value r(w).
        %   Inputs:
        %       w : (p x 1) vector of weights
        %   Outputs:
        %       val : scalar penalty value
        val = evaluate(obj, w)

        % GRADIENT  Calculate the gradient or subgradient of r(w).
        %   Inputs:
        %       w : (p x 1) vector of weights
        %   Outputs:
        %       grad : (p x 1) gradient vector
        grad = gradient(obj, w)

        % PROXIMITY  Apply the proximity operator for iterative shrinking.
        %   Solves: argmin_x { 0.5/(t) * ||x - w||_2^2 + lambda * r(x) }
        %   Inputs:
        %       w      : (p x 1) current weights
        %       t      : scalar step size (mu)
        %       lambda : scalar regularization strength (gamma)
        %   Outputs:
        %       w_prox : (p x 1) new proximity-adjusted weights
        w_prox = proximity(obj, w, t, lambda)

        % GETNAME  Return regularizer name.
        name = getName(obj)
    end
end
