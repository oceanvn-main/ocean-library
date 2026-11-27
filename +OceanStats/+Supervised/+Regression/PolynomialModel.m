classdef PolynomialModel < OceanStats.Base.ProbabilisticModel
    % POLYNOMIALMODEL A standard polynomial regression model.
    %
    % Inherits from ProbabilisticModel to work with BootstrapSampler.
    %
    % Author: Ocean Library System
    % Date: 2026-02-11

    properties
        Degree (1,1) double {mustBeNonnegative, mustBeInteger} = 1
    end

    methods
        function obj = PolynomialModel(degree)
            obj.Degree = degree;
        end

        function params = fit(obj, data, weights)
            % Fits a polynomial of obj.Degree to data, with optional weights.
            % Assumes data = [X, y]. If X contains an intercept, we strip it.
            
            X_raw = data(:, 1:end-1);
            if all(X_raw(:,1) == 1), X_raw = X_raw(:, 2:end); end
            
            x = X_raw(:, 1); % Univariate for this model
            y = data(:, end);
            
            % Construct Vandermonde Matrix: [1, x, x^2, ..., x^d]
            X_poly = ones(length(x), 1);
            for d = 1:obj.Degree
                X_poly = [X_poly, x.^d];
            end
            
            % Solve via WLS
            if nargin < 3 || isempty(weights) || all(weights == 1)
                beta = X_poly \ y;
                y_hat = X_poly * beta;
                sigma_sq = sum((y - y_hat).^2) / length(y);
            else
                W = max(weights(:), 1e-10);
                sqrtW = sqrt(W);
                Xw = X_poly .* sqrtW;
                yw = y .* sqrtW;
                beta = Xw \ yw;
                y_hat = X_poly * beta;
                sigma_sq = sum(W .* (y - y_hat).^2) / sum(W);
            end
            
            params = struct('beta', beta, 'degree', obj.Degree, 'sigma', sqrt(sigma_sq));
        end

        function y_pred = predict(obj, X, params)
            % X may or may not have intercept.
            if all(X(:,1) == 1), X = X(:, 2:end); end
            x = X(:, 1);
            
            % Reconstruct Vandermonde for prediction
            X_poly = ones(length(x), 1);
            for d = 1:params.degree
                X_poly = [X_poly, x.^d];
            end
            y_pred = X_poly * params.beta;
        end

        function data_new = boot_gen(obj, params, n, x_context)
            % Parametric Bootstrap variance generation
            if nargin < 4, error('Requires x_context'); end
            
            y_mean = obj.predict(x_context, params);
            epsilon = params.sigma * randn(size(x_context));
            data_new = [x_context, y_mean + epsilon];
        end
    end
end