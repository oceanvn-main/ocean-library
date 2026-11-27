classdef PolynomialLogisticModel < OceanStats.Base.ProbabilisticModel
    % POLYNOMIALLOGISTICMODEL Logistic Regression with Polynomial Basis expansion.
    
    properties
        Degree
        Beta
        Link
    end
    
    methods
        function obj = PolynomialLogisticModel(degree)
            if nargin < 1, degree = 2; end
            obj.Degree = degree;
            obj.Link = OceanStats.Link.SoftmaxLink();
        end
        
        function params = fit(obj, data, weights)
            X = data(:, 1:end-1);
            Y = data(:, end);
            
            % Polynomial Expansion
            X_poly = obj.augmentFeatures(X);
            
            % Convert Y to Indicator Matrix (Multi-class support)
            if size(Y, 2) == 1
                uniqueY = unique(Y);
                if length(uniqueY) == 2 && all(ismember(uniqueY, [0, 1]))
                    Y_hot = [1-Y, Y];
                else
                    % Use Indicator Encoding for K classes
                    Y_hot = zeros(size(Y,1), max(uniqueY));
                    for i = 1:size(Y,1)
                        if Y(i) > 0, Y_hot(i, Y(i)) = 1; end
                    end
                end
            else
                Y_hot = Y;
            end
            
            import OceanStats.Optimization.*
            recipe = GLMRecipe(X_poly, Y_hot, obj.Link, 'Weights', weights, 'Name', 'PolyLogistic Fit');
            optimizer = GenericOptimizer('MaxIter', 25, 'Verbose', false);
            [final_params, ~] = optimizer.run(recipe, []);
            
            obj.Beta = final_params.Theta;
            params = struct('beta', obj.Beta, 'degree', obj.Degree);
        end
        
        function y_prob = predict(obj, X, params)
            if nargin < 3 || isempty(params)
                beta = obj.Beta;
                degree = obj.Degree;
            else
                beta = params.beta;
                degree = params.degree;
            end
            
            X_poly = obj.augmentFeatures(X, degree);
            y_prob = obj.Link.apply(X_poly * beta);
            
            % For binary, return only P(Y=1) to stay consistent with regression output expectations
            if size(y_prob, 2) == 2
                y_prob = y_prob(:, 2);
            end
        end
        
        function data_new = boot_gen(~, ~, ~, ~)
            error('boot_gen not implemented.');
        end
    end
    
    methods (Access = private)
        function X_out = augmentFeatures(obj, X, degree)
            if nargin < 3, degree = obj.Degree; end
            
            % Strip intercept if present to avoid double augmentation
            has_intercept = all(X(:,1) == 1);
            if has_intercept, X = X(:, 2:end); end
            
            num_original = size(X, 2);
            
            if num_original == 1
                % Univariate expansion: [1, x, x^2, ...]
                x = X(:, end);
                X_out = ones(size(X, 1), 1);
                for d = 1:degree
                    X_out = [X_out, x.^d];
                end
            elseif num_original == 2
                % Bivariate expansion: [1, x1, x2, x1^2, x2^2, x1*x2]
                x1 = X(:, end-1);
                x2 = X(:, end);
                X_out = [ones(size(X, 1), 1), x1, x2, x1.^2, x2.^2, x1.*x2];
            else
                % Final safety: re-add intercept if we stripped it but didn't expand
                X_out = X;
                if has_intercept, X_out = [ones(size(X,1),1), X_out]; end
            end
        end
    end
end
