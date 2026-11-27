classdef (Abstract) ITensorTarget < handle
    % ITENSORTARGET  Abstract interface for domain targets optimizable by OceanStats.Optim.
    %
    %   Bridges Computer Vision, NLP, Signal Processing, Reinforcement
    %   Learning, and Continuous Optimization targets to the domain-agnostic
    %   VectorizedOptimizerEngine.
    %
    %   CONTRACT:
    %     theta = target.getParameters()          — Extract parameters (1D flat vector)
    %     target.setParameters(theta)             — Scatter updated parameters back to model
    %     grads = target.computeGradients(X, y)    — Compute loss gradients w.r.t parameters
    %
    %   See also: OceanStats.Optim.IOptimizerStrategy
    %             OceanStats.Optim.VectorizedOptimizerEngine

    methods (Abstract)
        theta = getParameters(obj)
        % GETPARAMETERS  Extract trainable parameters as a contiguous double vector.

        setParameters(obj, theta)
        % SETPARAMETERS  Scatter updated double vector back into target parameters.

        grads = computeGradients(obj, X, y)
        % COMPUTEGRADIENTS  Compute analytical or backprop loss gradients w.r.t theta.
    end
end
