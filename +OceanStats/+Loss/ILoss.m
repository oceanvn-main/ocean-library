classdef (Abstract) ILoss < handle
    % ILOSS  Abstract interface for all loss functions.
    %
    %   OVERVIEW:
    %     Defines the five-method contract for all loss functions.
    %     Enables gradient-based algorithms (GBM, LogitBoost) to swap
    %     their objective function at runtime without any code change.
    %
    %   CONTRACT:
    %     L     = forward(obj, Y_true, Y_pred)              — scalar mean loss
    %     g     = gradient(obj, Y_true, Y_pred)             — N×1 first derivative
    %     h     = hessian(obj, Y_true, Y_pred)              — N×1 second derivative
    %
    %   NOTE ON Y_pred:
    %     For regression losses, Y_pred is the raw model output (same units as Y).
    %     For classification losses, Y_pred is the raw log-odds / score (before
    %     applying sigmoid/softmax). This is the standard convention from
    %     XGBoost, LightGBM, and Gradient Boosting literature.
    %
    %   USAGE:
    %     loss = OceanStats.Loss.Regression.MSELoss();
    %     L    = loss.forward(Y_true, Y_pred);
    %     g    = loss.gradient(Y_true, Y_pred);
    %     h    = loss.hessian(Y_true, Y_pred);
    %
    %   EXTENSION:
    %     1. Subclass ILoss in +OceanStats/+Loss/+Regression/ or +Classification/
    %     2. Implement forward(), gradient(), hessian()
    %     3. Optionally override name() for display
    %
    %   See also: OceanStats.Loss.Regression.MSELoss
    %             OceanStats.Loss.Regression.PseudoHuberLoss
    %             OceanStats.Loss.Regression.ClassicHuberLoss
    %             OceanStats.Loss.Classification.CrossEntropyLoss
    %             OceanStats.Loss.Classification.ExponentialLoss

    methods (Abstract)
        % FORWARD  Compute the loss value(s).
        %
        %   L = forward(obj, Y_true, Y_pred)
        %
        %   Inputs:
        %     Y_true — N×1 ground truth values (regression) or {-1,+1} / {0,1} labels
        %     Y_pred — N×1 predicted scores (raw model output, NOT probabilities)
        %
        %   Output:
        %     L — scalar mean loss, or N×1 per-sample losses
        L = forward(obj, Y_true, Y_pred)

        % GRADIENT  First derivative of loss w.r.t. Y_pred.
        %
        %   g = gradient(obj, Y_true, Y_pred)
        %
        %   Output:
        %     g — N×1 vector of first-order derivatives  (residuals / pseudo-residuals)
        g = gradient(obj, Y_true, Y_pred)

        % HESSIAN  Second derivative of loss w.r.t. Y_pred.
        %
        %   h = hessian(obj, Y_true, Y_pred)
        %
        %   Output:
        %     h — N×1 vector of second-order derivatives  (used by Newton-step methods)
        h = hessian(obj, Y_true, Y_pred)

        % NAME  Human-readable label for this loss.
        %
        %   Output:
        %     n — string (e.g. "MSE", "Huber (delta=1.0)")
        n = name(obj)
    end

    methods
        % CHECKDIMS  Validate that Y_true and Y_pred are conformant N×1 vectors.
        % Called internally by concrete subclasses as a guard.
        function checkDims(~, Y_true, Y_pred)
            if ~isequal(size(Y_true), size(Y_pred))
                error('OceanStats:Loss:ILoss:DimMismatch', ...
                    'Y_true (%s) and Y_pred (%s) must have the same size.', ...
                    mat2str(size(Y_true)), mat2str(size(Y_pred)));
            end
        end
    end
end
