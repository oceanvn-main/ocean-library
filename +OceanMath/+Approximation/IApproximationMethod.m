classdef IApproximationMethod < handle
    % IAPPROXIMATIONMETHOD Interface for local function approximation methods.
    %
    % A purely mathematical strategy that estimates the gain/value of moving
    % in a direction given the **gradient** and **curvature** at the current
    % point. This interface is agnostic to the domain (MARS, GLM, GP, etc.).
    %
    % Implementations define the specific approximation formula, e.g.:
    %   - SecondOrderApproximation  : gain = sum(grad.^2) / curv   (Score Test)
    %   - FirstOrderApproximation   : gain = sum(grad)             (gradient only)
    %
    % See also: OceanStats.Base.IObjectiveApproximation
    
    methods (Abstract)
        % COMPUTE Approximate the objective gain from gradient and curvature.
        %
        % INPUTS:
        %   gradient  - scalar or (1 x K) vector of first-order derivatives
        %   curvature - scalar second-order term (Schur complement, variance, etc.)
        %
        % OUTPUT:
        %   gain - scalar; higher means this direction is more promising
        gain = compute(obj, gradient, curvature)
    end
end
