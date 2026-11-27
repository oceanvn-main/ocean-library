classdef LogisticModel < OceanStats.Base.ProbabilisticModel
    % LOGISTICMODEL Logistic Regression model for classification.
    %
    % Uses GLMRecipe (IRLS) for fitting.
    
    properties
        Beta
        Link
    end
    
    methods
        function obj = LogisticModel()
            obj.Link = OceanStats.Link.SoftmaxLink(); % Using Softmax for general K-class
        end
        
        function params = fit(obj, data, weights)
            X = data(:, 1:end-1);
            Y = data(:, end);
            
            % Add intercept
            if ~all(X(:, 1) == 1)
                X = [ones(size(X, 1), 1), X];
            end
            
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
            recipe = GLMRecipe(X, Y_hot, obj.Link, 'Weights', weights, 'Name', 'LogisticModel Fit');
            optimizer = GenericOptimizer('MaxIter', 25, 'Verbose', false);
            [final_params, ~] = optimizer.run(recipe, []);
            
            obj.Beta = final_params.Theta;
            params = struct('beta', obj.Beta);
        end
        
        function y_prob = predict(obj, X, params)
            if ~all(X(:, 1) == 1)
                X = [ones(size(X, 1), 1), X];
            end
            if nargin < 3 || isempty(params)
                beta = obj.Beta;
            else
                beta = params.beta;
            end
            
            y_prob_all = obj.Link.apply(X * beta);
            % Return probability of class 1 for binary
            if size(y_prob_all, 2) == 2
                y_prob = y_prob_all(:, 2);
            else
                y_prob = y_prob_all;
            end
        end
        
        function data_new = boot_gen(~, ~, ~, ~)
            error('boot_gen not implemented for LogisticModel yet.');
        end
    end
end
