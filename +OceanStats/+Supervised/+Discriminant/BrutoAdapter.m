classdef BrutoAdapter < handle
    % BRUTOADAPTER  Additive smoothing-spline model (GAM via backfitting).
    %
    %   A faithful MATLAB reimplementation of Friedman's "bruto" algorithm
    %   used for FDA in ESL Section 12.5.  Each predictor gets its own
    %   cubic smoothing spline; the splines are coordinated by iterative
    %   backfitting (Gauss–Seidel on partial residuals).
    %
    %   Interface matches what FDA.BaseModelConstructor expects:
    %     obj.fit(X, y)           — train the additive model
    %     y_hat = obj.predict(X)  — predict new data
    %
    %   Uses MATLAB built-in csaps() for the individual smoothers.
    %
    %   Usage inside FDA:
    %     brutoBuilder = @() OceanStats.Supervised.Discriminant.BrutoAdapter();
    %     fda = OceanStats.Supervised.Discriminant.FDA('BaseModelConstructor', brutoBuilder);

    properties
        SmoothingParam  double = []   % csaps smoothing parameter p; [] = auto
        MaxIter         double = 50   % backfitting iterations
        Tol             double = 1e-5 % convergence tolerance on RSS change
        DfPerFeature    double = 4    % target effective df per smoother (used when SmoothingParam=[])
    end

    properties (SetAccess = private)
        Intercept_      double        % scalar intercept (mean of y)
        SplinePP_       cell          % 1×p cell of piecewise-polynomial structs
        CenterConst_    double        % 1×p vector of centering constants (training means)
        p_              double        % number of predictors
    end

    methods
        function obj = BrutoAdapter(varargin)
            for i = 1:2:numel(varargin)
                obj.(varargin{i}) = varargin{i+1};
            end
        end

        function fit(obj, X, y)
            % FIT  Train additive model via backfitting with csaps.
            [N, p] = size(X);
            obj.p_ = p;
            y = y(:);

            % Centre the response
            obj.Intercept_ = mean(y);
            r = y - obj.Intercept_;

            % Choose smoothing parameter:
            %   If user set SmoothingParam, use it directly.
            %   Otherwise, use csaps GCV per-feature but clamp to prevent
            %   overfitting (cap at p_max derived from target df).
            p_max = max(1e-6, min(0.999, ...
                (obj.DfPerFeature - 1) / (N - 1)));
            use_gcv = isempty(obj.SmoothingParam);

            % Initialise individual spline contributions to zero
            f = zeros(N, p);
            obj.SplinePP_    = cell(1, p);
            obj.CenterConst_ = zeros(1, p);

            prevRSS = Inf;
            for iter = 1:obj.MaxIter
                for j = 1:p
                    % Partial residual: everything except feature j
                    partial = r - sum(f, 2) + f(:, j);

                    % Sort by feature j (csaps needs sorted x)
                    [xj_sorted, si] = sort(X(:, j));
                    partial_sorted = partial(si);

                    % Handle duplicate x: average y at tied values
                    [xu, ~, ic] = unique(xj_sorted);
                    if length(xu) < length(xj_sorted)
                        yu = accumarray(ic, partial_sorted, [], @mean);
                    else
                        xu = xj_sorted;
                        yu = partial_sorted;
                    end

                    % Fit cubic smoothing spline
                    if use_gcv
                        % Let csaps choose p via GCV, then clamp
                        [pp, p_auto] = csaps(xu, yu);
                        if p_auto > p_max
                            pp = csaps(xu, yu, p_max);
                        end
                    else
                        pp = csaps(xu, yu, obj.SmoothingParam);
                    end
                    obj.SplinePP_{j} = pp;

                    % Evaluate on training data and centre
                    fj_sorted = fnval(pp, xj_sorted);
                    fj = zeros(N, 1);
                    fj(si) = fj_sorted;
                    cj = mean(fj);
                    obj.CenterConst_(j) = cj;
                    f(:, j) = fj - cj;
                end

                % Check convergence
                fitted = obj.Intercept_ + sum(f, 2);
                rss = sum((y - fitted).^2);
                if abs(prevRSS - rss) / (prevRSS + 1e-12) < obj.Tol
                    break;
                end
                prevRSS = rss;
            end
        end

        function y_hat = predict(obj, X)
            % PREDICT  Evaluate the fitted additive model on new data.
            N = size(X, 1);
            y_hat = obj.Intercept_ * ones(N, 1);
            for j = 1:obj.p_
                fj = fnval(obj.SplinePP_{j}, X(:, j));
                fj = fj(:) - obj.CenterConst_(j);   % use TRAINING centering constant
                y_hat = y_hat + fj;
            end
        end
    end
end
