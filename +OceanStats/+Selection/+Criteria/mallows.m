function score = mallows(RSS, sigma_sq_hat, N, p, options)
% MALLOWS Compute Mallows' Cp statistic (Standard or Textbook version).
%
%   score = mallows(RSS, sigma_sq, N, p)
%   score = mallows(..., 'Variant', 'Textbook')
%
%   This function calculates the Mallows' Cp statistic, which estimates the
%   in-sample prediction error. It provides two common variants:
%
%   1. "Standard" (Default): Unitless. Best for Model Selection.
%      Formula: Cp = (RSS / sigma^2) - N + 2*p
%      Target:  Good models have Cp approx p.
%
%   2. "Textbook" (ESL Eq 7.26): Units of Y^2. Best for Error Reporting.
%      Formula: Err = (RSS / N) + 2 * (p / N) * sigma^2
%      Target:  The estimated Mean Squared Error (MSE) on new data.
%
%   Inputs:
%       RSS          - Residual Sum of Squares (Training Error)
%       sigma_sq_hat - Estimate of true noise variance (from Low-Bias model)
%       N            - Sample size
%       p            - Number of parameters (Degree + 1)
%       options.Variant - "Standard" (default) or "Textbook"
%
%   Example:
%       cp = OceanStats.ModelSelection.mallows(50.5, 0.6, 100, 5);
%       err = OceanStats.ModelSelection.mallows(50.5, 0.6, 100, 5, 'Variant', 'Textbook');

    arguments
        RSS (1,1) double {mustBeNonnegative}
        sigma_sq_hat (1,1) double {mustBePositive}
        N (1,1) double {mustBeInteger, mustBePositive}
        p (1,1) double {mustBeInteger, mustBeNonnegative}
        options.Variant (1,1) string {mustBeMember(options.Variant, ["Standard", "Textbook"])} = "Standard"
    end

    switch options.Variant
        case "Standard"
            % The classic 1973 definition. Used for finding the "best" model.
            % Ideally close to p (or d+1).
            score = (RSS / sigma_sq_hat) - N + 2 * p;
            
        case "Textbook"
            % The ESL definition. Used for estimating actual MSE.
            % Represents: Training MSE + Optimism Penalty
            Training_MSE = RSS / N;
            Optimism     = 2 * (p / N) * sigma_sq_hat;
            score        = Training_MSE + Optimism;
    end
end