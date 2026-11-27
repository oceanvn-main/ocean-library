classdef NaturalCubicStrategy < OceanStats.NonParametric.Strategies.ISplineStrategy
    % NATURALCUBICSTRATEGY ESL-exact natural cubic spline smoother strategy.
    %
    % THEORY (ESL §5.4, Ex 5.9):
    %   The smoothing spline minimizes:
    %       RSS(f, lambda) = sum(y_i - f(x_i))^2 + lambda * integral(f''(t)^2 dt)
    %
    %   The unique minimizer is a natural cubic spline with knots at each x_i.
    %   The smoother matrix is:
    %       S_lambda = N * (N'N + lambda*Omega_N)^-1 * N'
    %
    %   This implementation uses the efficient Q/R eigendecomposition:
    %       K = Q * (R \ Q'),   S_lambda = U * diag(1/(1 + lambda*d)) * U'
    %   where [U, D] = eig(K), d = diag(D).
    %
    %   KEY PROPERTY: S_lambda preserves any linear function exactly:
    %       S_lambda * (a + b*x) = (a + b*x)   for all a, b.
    %
    % PARAMETERS:
    %   DF   - Target effective df (trace of S_lambda). Takes priority over Lambda.
    %   Mode - 'df' (auto-tune lambda) or 'lambda' (use Lambda directly)
    %
    % Author: Ocean Library System
    % Date:   2026-02-28

    properties
        Name   string = "Natural Cubic Spline"
        DF     (1,1) double {mustBePositive} = 8
        Lambda (1,1) double {mustBeNonnegative} = 0
        Mode   string {mustBeMember(Mode, {'df','lambda'})} = 'df'
    end

    methods
        function obj = NaturalCubicStrategy(options)
            arguments
                options.DF     = 8
                options.Lambda = 0
                options.Mode   = 'df'
            end
            obj.DF     = options.DF;
            obj.Lambda = options.Lambda;
            obj.Mode   = options.Mode;
        end

        % -----------------------------------------------------------------
        function model = fit(obj, X, Y, ~)
            % FIT  Builds the smoother matrix S_lambda and returns in-sample fit.
            X = X(:); Y = Y(:);
            
            % CRITICAL FIX: S_lambda requires strictly unique knots (X values).
            % If X contains duplicates, the spacing h=diff(X) contains zeros,
            % leading to divide-by-zero (NaN) when building the Q difference matrix.
            [X_uniq, ~, ic] = unique(X);
            Y_uniq = zeros(length(X_uniq), 1);
            for k = 1:length(X_uniq)
                 Y_uniq(k) = mean(Y(ic == k));
            end

            if strcmp(obj.Mode, 'df')
                [S, lam] = OceanStats.NonParametric.getSmootherMatrix(X_uniq, obj.DF, 'df');
            else
                [S, lam] = OceanStats.NonParametric.getSmootherMatrix(X_uniq, obj.Lambda, 'lambda');
            end
            
            % Smooth only the unique values
            y_fit_uniq = S * Y_uniq;
            df         = trace(S);
            
            % Broadcast the unique fitted values back to the full N-length vector
            y_fit = y_fit_uniq(ic);

            model = struct( ...
                'S',       S,          ...
                'lambda',  lam,        ...
                'x_train', X_uniq,     ...
                'y_train', Y_uniq,     ...
                'y_fit',   y_fit_uniq, ...
                'df',      df);
        end

        function y_pred = predict(~, model, X_new)
            % PREDICT  Evaluate at new (out-of-sample) query points.
            X_new  = X_new(:);
            X_uniq = model.x_train(:); % already unique from fit()
            Y_uniq_fit = model.y_fit(:); % already uniquely aggregated from fit()
            N_new  = length(X_new);
            
            y_pred = zeros(N_new, 1);
            for i = 1:N_new
                % Linear interpolation from training fitted values
                % If a test point falls outside the training range, 'extrap'
                % can occasionally overflow/NaN if knots are tight.
                % Use 'extrap' but explicitly catch and map NaNs to Nearest.
                y_pred(i) = interp1(X_uniq, Y_uniq_fit, X_new(i), 'linear', 'extrap');
            end
            
            % Secondary safety fallback to ensure NO NaNs enter the GAM
            if any(isnan(y_pred))
                nanIdx = isnan(y_pred);
                y_pred(nanIdx) = interp1(X_uniq, Y_uniq_fit, X_new(nanIdx), 'nearest', 'extrap');
            end
        end

        % -----------------------------------------------------------------
        function k = getDF(~, model)
            k = model.df;
        end
    end
end
