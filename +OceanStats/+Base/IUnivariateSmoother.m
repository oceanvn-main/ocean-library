classdef (Abstract) IUnivariateSmoother < handle
    % IUNIVARIATESMOOTHER  Base contract for all 1-D curve fitting strategies.
    %
    % PURPOSE:
    %   Any object that can fit a smooth curve to (X, Y) data and predict
    %   at new points satisfies this interface.  It is the foundational
    %   contract depended upon by:
    %
    %     OceanStats.GAM.Backfitting      — iterates over smoothers per predictor
    %     OceanStats.NonParametric.
    %       Strategies.ISplineStrategy    — extends with getDF()
    %
    % DEPENDENCY ARROWS (correct, all point downward):
    %
    %   +GAM           →  +Base
    %   +NonParametric →  +Base
    %
    % CONTRACT:
    %   model  = fit(obj, X, Y, Weights)    — fit a curve to 1-D data
    %   y_pred = predict(obj, model, X_new) — evaluate at new locations
    %
    %   INPUTS:
    %     X       - (N x 1) predictor values
    %     Y       - (N x 1) response or partial residual
    %     Weights - (N x 1) per-observation weights (pass [] for uniform)
    %     model   - opaque struct returned by fit()
    %     X_new   - (M x 1) query points
    %
    %   OUTPUTS:
    %     model  - struct carrying all parameters needed by predict()
    %     y_pred - (M x 1) fitted/predicted values
    %
    % AUTHOR: Ocean Library System
    % DATE:   2026-02-28

    methods (Abstract)
        % FIT  Fit a 1-D smoother to weighted data.
        model = fit(obj, X, Y, Weights)

        % PREDICT  Evaluate the fitted smoother at new locations.
        y_pred = predict(obj, model, X_new)
    end
end
