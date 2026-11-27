classdef (Abstract) ITreeLoss < OceanStats.Loss.ILoss
    % ITREELOSS  Abstract interface for tree-based loss functions.
    %
    %   OVERVIEW:
    %     Extends ILoss to include Gradient Boosting specific initialization
    %     and leaf optimization routines that non-tree models (like SVMs)
    %     do not naturally require.
    %
    %   CONTRACT:
    %     In addition to ILoss (forward, gradient, hessian, name), must provide:
    %     gamma = initPrediction(obj, Y_true)
    %     gamma = computeLeafConstant(obj, Y_true, F_prev, leafMask)
    %
    %   USAGE:
    %     loss = OceanStats.Loss.Regression.MSELoss();
    %     gamma_0 = loss.initPrediction(Y_train);
    %
    %   See also: OceanStats.Loss.ILoss
    %             OceanStats.Loss.Regression.MSELoss
    %             OceanStats.Loss.Classification.CrossEntropyLoss

    methods (Abstract)
        % INITPREDICTION  Compute the optimal constant initial prediction F_0.
        %
        %   gamma = initPrediction(obj, Y_true)
        %
        %   Minimizes sum_i L(y_i, gamma) over the scalar gamma.
        %   Called once before the GBM training loop begins.
        %
        %   Output:
        %     gamma — scalar, the optimal constant starting prediction
        gamma = initPrediction(obj, Y_true)

        % COMPUTELEAFCONSTANT  Optimal leaf output under the TRUE loss (Algorithm 10.3, Step 2c).
        %
        %   gamma = computeLeafConstant(obj, Y_true, F_prev, leafMask)
        %
        %   After the surrogate (squared-error) tree defines the regions,
        %   this re-optimizes every leaf's output using the REAL loss function.
        %   This is what separates robust GBM (Huber/MAE) from vanilla squared-error boosting.
        %
        %   Inputs:
        %     Y_true   — N×1 ground truth (full training set)
        %     F_prev   — N×1 current ensemble predictions (before this tree)
        %     leafMask — N×1 logical, true for samples in this leaf
        %
        %   Output:
        %     gamma — scalar optimal leaf constant
        gamma = computeLeafConstant(obj, Y_true, F_prev, leafMask)
    end
end
