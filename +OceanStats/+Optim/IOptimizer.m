classdef (Abstract) IOptimizer < handle
    % IOPTIMIZER  Abstract interface for all parameter optimizers.
    %
    %   Each concrete optimizer manages its own internal state (e.g. moment
    %   buffers). One instance is created per trainable parameter tensor
    %   (one for W, one for b per layer).
    %
    %   CONTRACT:
    %     param_new = opt.step(param, grad)   — apply one update, return new param
    %     opt.reset()                         — clear internal state (re-use for new run)
    %     s = opt.describe()                  — human-readable hyperparameter summary
    %
    %   USAGE:
    %     opt = OceanStats.Optim.SGDOptimizer('LR', 0.01);
    %     W   = opt.step(W, dW);
    %
    %   See also: OceanStats.Optim.SGDOptimizer
    %             OceanStats.Optim.SGDMomentumOptimizer
    %             OceanStats.Optim.RMSPropOptimizer
    %             OceanStats.Optim.AdamOptimizer

    methods (Abstract)
        param = step(obj, param, grad)
        % STEP  Apply one optimizer update.
        %   param — current parameter (any numeric array)
        %   grad  — gradient w.r.t. param (same shape)
        %   Returns updated param (same shape).

        reset(obj)
        % RESET  Clear internal state for a fresh training run.

        s = describe(obj)
        % DESCRIBE  Return a human-readable string of hyperparameters.
    end
end
