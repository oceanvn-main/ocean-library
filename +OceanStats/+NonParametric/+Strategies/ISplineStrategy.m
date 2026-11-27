classdef (Abstract) ISplineStrategy < OceanStats.Base.IUnivariateSmoother
    % ISPLINESTRATEGY Abstract interface for all spline fitting strategies.
    %
    % Extends IUnivariateSmoother, adding getDF() — the effective degrees
    % of freedom, a spline-specific concept not needed by Backfitting itself.
    %
    % INHERITED from IUnivariateSmoother (do NOT re-declare here):
    %   model  = fit(obj, X, Y, Weights)    — fit smoother to 1-D data
    %   y_pred = predict(obj, model, X_new) — evaluate at new locations
    %
    % ADDED here (spline-specific):
    %   Name  property  — human-readable label (e.g. "P-Spline")
    %   k = getDF(obj, model) — trace of smoother matrix
    %
    % INHERITANCE CHAIN:
    %   Base.IUnivariateSmoother   (fit + predict)
    %     └─ ISplineStrategy       (+ Name, + getDF)
    %          ├─ PSplineStrategy
    %          ├─ NaturalCubicStrategy
    %          └─ RegressionSplineStrategy
    %
    % Author: Ocean Library System
    % Date:   2026-02-28

    properties (Abstract)
        Name string   % Human-readable label, e.g. "P-Spline"
    end

    methods (Abstract)
        % GETDF  Return the effective degrees of freedom of the fitted smoother.
        %   Typically trace(S) where S is the smoother (hat) matrix.
        %   model - struct returned by fit()
        %   k     - scalar effective df
        k = getDF(obj, model)
    end
end
