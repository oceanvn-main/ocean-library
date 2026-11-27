classdef PSplineStrategy < OceanStats.NonParametric.Strategies.ISplineStrategy
    % PSPLINESTRATEGY P-Spline fitting strategy (B-Spline basis + 2nd-order difference penalty).
    %
    % THEORY:
    %   Minimize: || sqrt(W)(y - H*beta) ||^2 + lambda * beta' * D'D * beta
    %   where H is a B-Spline basis and D = diff(eye(p), DiffOrder).
    %
    %   Using a difference penalty instead of ridge (L2) makes the smoother
    %   converge to a polynomial of degree (DiffOrder-1) as lambda -> Inf,
    %   instead of collapsing to zero. This gives much more interpretable
    %   regularization.
    %
    % PARAMETERS:
    %   Knots    - Scalar (# uniform knots) or vector (knot locations)
    %   Degree   - B-Spline polynomial degree (default 3 = cubic)
    %   Lambda   - Regularization strength (overridden if TargetDF > 0)
    %   Domain   - [min, max] of predictor domain
    %   TargetDF - If > 0, auto-tune Lambda to achieve this trace(S)
    %   DiffOrder- Order of the difference penalty (default 2 = curvature)
    %
    % Author: Ocean Library System
    % Date:   2026-02-28

    properties
        Name     string = "P-Spline"

        Knots    double {mustBeNonempty, mustBeReal} = 15
        Degree   (1,1) double {mustBePositive, mustBeInteger} = 3
        Lambda   (1,1) double {mustBeNonnegative} = 0.1
        Domain   (1,2) double = [0, 1]
        TargetDF (1,1) double {mustBeNonnegative} = 0
        DiffOrder(1,1) double {mustBePositive, mustBeInteger} = 2
    end

    properties (Access = private)
        % Basis cache to avoid regenerating for repeated calls
        CachedBasis struct = struct('H', [], 'CheckSum', [])
    end

    methods
        function obj = PSplineStrategy(options)
            arguments
                options.Knots     = 15
                options.Degree    = 3
                options.Lambda    = 0.1
                options.Domain    = [0, 1]
                options.TargetDF  = 0
                options.DiffOrder = 2
            end
            obj.Knots     = options.Knots;
            obj.Degree    = options.Degree;
            obj.Lambda    = options.Lambda;
            obj.Domain    = options.Domain;
            obj.TargetDF  = options.TargetDF;
            obj.DiffOrder = options.DiffOrder;
        end

        % -----------------------------------------------------------------
        function model = fit(obj, X, Y, Weights)
            % FIT  Fits 1D P-Spline to (X, Y).
            X = X(:); Y = Y(:);
            N = length(X);

            if nargin < 4 || isempty(Weights)
                Weights = ones(N, 1);
            end
            Weights = Weights(:);

            % 1. Generate / retrieve basis H (N x p)
            H = obj.getBasis(X);
            [~, p] = size(H);

            % 2. Weighted normal equations
            W_diag = spdiags(Weights, 0, N, N);
            HTWH   = H' * W_diag * H;
            HTWy   = H' * W_diag * Y;

            % 3. Difference penalty matrix  Omega = D'D
            if p > obj.DiffOrder
                D     = diff(eye(p), obj.DiffOrder);
                Omega = D' * D;
            else
                Omega = eye(p);
            end

            % 4. Auto-tune Lambda for target df
            lam = obj.Lambda;
            if obj.TargetDF > 0
                safe_df = min(max(obj.TargetDF, 0.1), p - 0.1);
                lam = obj.findLambdaForDF(HTWH, Omega, safe_df);
            end

            % 5. Solve
            beta = (HTWH + (lam + 1e-10) * Omega) \ HTWy;

            % 6. Effective df = trace( (H'WH + lam*Omega)^-1 * H'WH )
            try
                df = trace((HTWH + (lam + 1e-10) * Omega) \ HTWH);
            catch
                df = p;
            end

            model = struct( ...
                'Beta',    beta,       ...
                'Knots',   obj.Knots,  ...
                'Degree',  obj.Degree, ...
                'Domain',  obj.Domain, ...
                'Lambda',  lam,        ...
                'df',      df);
        end

        % -----------------------------------------------------------------
        function y_pred = predict(obj, model, X_new)
            % PREDICT  Evaluate P-Spline at new points.
            X_new = X_new(:);
            import OceanStats.NonParametric.generateBSplineBasis;
            H_new  = generateBSplineBasis(X_new, model.Domain, model.Knots, model.Degree);
            y_pred = H_new * model.Beta;
        end

        % -----------------------------------------------------------------
        function k = getDF(~, model)
            k = model.df;
        end
    end

    % =====================================================================
    methods (Access = private)

        function H = getBasis(obj, X)
            % Return cached basis or regenerate if X has changed.
            checkSum = sum(X) + X(1) + X(end);
            if isempty(obj.CachedBasis.H) || obj.CachedBasis.CheckSum ~= checkSum
                import OceanStats.NonParametric.generateBSplineBasis;
                H_dense = generateBSplineBasis(X, obj.Domain, obj.Knots, obj.Degree);
                obj.CachedBasis.H        = sparse(H_dense);
                obj.CachedBasis.CheckSum = checkSum;
            end
            H = obj.CachedBasis.H;
        end

        function lambda = findLambdaForDF(obj, HTWH, Omega, target_df)
            % FINDLAMBDAFORDF  Solve trace(S_lambda) = target_df via fzero in log-space.
            p = size(HTWH, 1);
            if target_df >= p, lambda = 0;   return; end
            if target_df <= 1e-3, lambda = 1e8; return; end

            cost = @(log_lam) obj.traceS(HTWH, Omega, 10^log_lam) - target_df;

            f_lo = cost(-8);
            f_hi = cost(8);

            if f_lo * f_hi > 0
                % Bracket failed — pick closest edge
                if abs(f_lo) < abs(f_hi), lambda = 1e-8; else, lambda = 1e8; end
                return
            end

            try
                lambda = 10^fzero(cost, [-8, 8]);
            catch
                lambda = 1.0;
            end
        end

        function df = traceS(~, HTWH, Omega, lambda)
            df = trace((HTWH + (lambda + 1e-10) * Omega) \ HTWH);
        end
    end
end
