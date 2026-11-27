classdef CompositeObjective < handle
    % COMPOSITEOBJECTIVE  Wrapper for f(w) + lambda * r(w)
    %
    %   Encapsulates an ILoss and an IRegularizer. Evaluates the 
    %   combined objective function, its gradient, and provides 
    %   access to the proximity operator.
    %   This allows solvers (like SGD or Proximal Gradient) to remain
    %   completely agnostic to the underlying mathematical choices.
    %
    %   USAGE:
    %     objFunc = CompositeObjective(mseLoss, l1Reg, 0.1);
    %     val = objFunc.evaluate(Y_true, X*w, w);
    %     w_new = objFunc.proximity(w, stepSize);
    %
    %   Author: Ocean Library | 2026-03-13

    properties
        Loss        OceanStats.Loss.ILoss
        Regularizer OceanStats.Regularization.IRegularizer
        Lambda      (1,1) double {mustBeNonnegative} = 0.0
    end

    methods
        function obj = CompositeObjective(loss, regularizer, lambda)
            arguments
                loss        OceanStats.Loss.ILoss
                regularizer OceanStats.Regularization.IRegularizer = []
                lambda      (1,1) double {mustBeNonnegative} = 0.0
            end
            obj.Loss = loss;
            if isempty(regularizer)
                % Default to L2 with lambda=0 (effectively unregularized)
                obj.Regularizer = OceanStats.Regularization.L2Regularizer();
                obj.Lambda = 0.0;
            else
                obj.Regularizer = regularizer;
                obj.Lambda = lambda;
            end
        end

        function val = evaluate(obj, Y_true, Y_pred, w) 
            % EVALUATE  Calculate total composite loss f(w) + lambda * r(w), w is beta
            lossVal = obj.Loss.forward(Y_true, Y_pred);
            regVal  = obj.Regularizer.evaluate(w);
            val     = mean(lossVal) + obj.Lambda * regVal;
        end

        function grad = gradient(obj, Y_true, Y_pred, X, w)
            % GRADIENT  Calculate total gradient w.r.t w
            %   Assumes linear model Y_pred = X * w for the chain rule
            %   grad = X' * dL/dY_pred / N + lambda * dr/dw
            N = length(Y_true);
            dL_dY = obj.Loss.gradient(Y_true, Y_pred); % (N x 1)
            
            lossGrad = (X' * dL_dY) / N;               % (p x 1)
            regGrad  = obj.Regularizer.gradient(w);    % (p x 1)
            
            grad = lossGrad + obj.Lambda * regGrad;
        end

        function w_prox = proximity(obj, w, t)
            % PROXIMITY  Apply regularizer's proximity operator
            %   Used by Proximal Gradient methods.
            w_prox = obj.Regularizer.proximity(w, t, obj.Lambda);
        end
    end
end
