classdef KNNModel < OceanStats.Base.ProbabilisticModel
    % KNNMODEL K-Nearest Neighbors Regression.
    %
    % A simple non-parametric model.
    % K=1 is highly unstable (High Variance).
    % High K is stable (High Bias).
    
    properties
        K (1,1) double {mustBePositive, mustBeInteger} = 1
    end

    methods
        function obj = KNNModel(k)
            obj.K = k;
        end

        function params = fit(obj, data)
            % For KNN, fitting is just storing the data.
            params = struct('X', data(:,1), 'Y', data(:,2));
        end

        function y_pred = predict(obj, x_new, params)
            % Find K nearest neighbors for each x_new
            X = params.X;
            Y = params.Y;
            N_new = length(x_new);
            y_pred = zeros(N_new, 1);
            
            for i = 1:N_new
                dist = abs(X - x_new(i));
                [~, idx] = sort(dist);
                neighbors_idx = idx(1:min(obj.K, length(X)));
                y_pred(i) = mean(Y(neighbors_idx));
            end
        end

        function data_new = boot_gen(obj, params, n, x_context)
            % Parametric Bootstrap? 
            % Usually KNN doesn't have a generative model (sigma).
            % But we can use the residuals to estimate sigma.
            
            if nargin < 4, error('Requires x_context'); end
            
            y_hat = obj.predict(params.X, params); % Predict on training data
            residuals = params.Y - y_hat;
            sigma = std(residuals);
            
            y_mean = obj.predict(x_context, params);
            epsilon = sigma * randn(size(x_context));
            
            % But wait! Bagging usually uses Non-Parametric Bootstrap (Resampling Data).
            % Our Experiment_2_1 uses `sampler.drawNonParametric`.
            % So `boot_gen` is NOT CALLED during bagging!
            % It is only called if we do Parametric Bootstrap.
            % But we should implement it for completeness.
            
            data_new = [x_context, y_mean + epsilon];
        end
    end
end
