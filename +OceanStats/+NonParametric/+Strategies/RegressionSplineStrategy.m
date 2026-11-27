classdef RegressionSplineStrategy < OceanStats.NonParametric.Strategies.ISplineStrategy
    % REGRESSIONSPLINESTRATEGY Natural cubic spline basis + OLS (no penalty).
    %
    % THEORY (ESL §5.2):
    %   Uses the truncated power basis (getNSBasis) which enforces the natural
    %   boundary conditions (zero second-derivative at boundaries) exactly.
    %   Fits OLS: beta = (N'N)^-1 * N'y, where N is the (K-1) x N basis matrix.
    %
    %   Unlike P-Splines, complexity is controlled entirely by the NUMBER of knots,
    %   not by a penalty parameter. Quantile-placed knots give good coverage.
    %
    % PARAMETERS:
    %   NumKnots - Number of interior knots (knots placed at quantiles of X)
    %   Knots    - Explicit knot locations (overrides NumKnots if specified)
    %
    % Author: Ocean Library System
    % Date:   2026-02-28

    properties
        Name     string = "Regression Spline (Natural Cubic)"
        NumKnots (1,1) double {mustBePositive, mustBeInteger} = 5
        Knots    double = []  % If non-empty, used directly (overrides NumKnots)
    end

    methods
        function obj = RegressionSplineStrategy(options)
            arguments
                options.NumKnots = 5
                options.Knots    = []
            end
            obj.NumKnots = options.NumKnots;
            obj.Knots    = options.Knots;
        end

        % -----------------------------------------------------------------
        function model = fit(obj, X, Y, ~)
            % FIT  Fit natural cubic spline regression via OLS.
            X = X(:); Y = Y(:);

            % Resolve knot locations
            knots = obj.resolveKnots(X);

            % Build the natural spline basis (N x K-1), excluding intercept
            N_basis = OceanStats.NonParametric.getNSBasis(X, knots);

            % Prepend intercept column
            H = [ones(length(X), 1), N_basis];

            % OLS: beta = (H'H + ridgeEps*I) \ H'Y
            % Small ridge for numerical safety (not a regularization penalty)
            p     = size(H, 2);
            beta  = (H' * H + 1e-10 * eye(p)) \ (H' * Y);

            y_fit = H * beta;
            df    = p;  % No shrinkage: df = number of parameters

            model = struct( ...
                'Beta',   beta,  ...
                'Knots',  knots, ...
                'df',     df,    ...
                'x_train', X,    ...
                'y_fit',   y_fit);
        end

        % -----------------------------------------------------------------
        function y_pred = predict(obj, model, X_new)
            % PREDICT  Evaluate regression spline at new points.
            X_new  = X_new(:);
            N_new  = OceanStats.NonParametric.getNSBasis(X_new, model.Knots);
            H_new  = [ones(length(X_new), 1), N_new];
            y_pred = H_new * model.Beta;
        end

        % -----------------------------------------------------------------
        function k = getDF(~, model)
            k = model.df;
        end
    end

    % =====================================================================
    methods (Access = private)
        function knots = resolveKnots(obj, X)
            % Choose knot locations: use explicit Knots if provided,
            % otherwise place NumKnots at uniform quantiles of X.
            if ~isempty(obj.Knots)
                knots = sort(obj.Knots(:));
            else
                % Include boundary knots at 0th and 100th percentile
                quantile_levels = linspace(0, 1, obj.NumKnots + 2);
                knots = quantile(X, quantile_levels);
            end
        end
    end
end
