classdef QDA < OceanStats.Supervised.Discriminant.IDiscriminantModel
    % QDA Quadratic Discriminant Analysis Classifier
    %
    % Description:
    %   Implements Quadratic Discriminant Analysis (QDA).
    %   Unlike LDA, QDA assumes each class has its OWN unique Covariance Matrix (Sigma_k).
    %   This allows for classes with different shapes and sizes, resulting in a quadratic 
    %   (curved) decision boundary between classes.
    %
    % Usage:
    %   model = OceanStats.Supervised.Discriminant.QDA();
    %   model.fit(X_train, Y_train);
    %   preds = model.predict(X_test);
    
    properties (SetAccess = private)
        Means           % K x p matrix of class centroids
        Sigma           % p x p x K array of class covariances
        LogDet          % K x 1 vector of log-determinants
        NumFeatures     % Integer p
    end
    
    methods
        function obj = QDA(varargin)
            % QDA Constructor
        end
        
        function fit(obj, X, y)
            % FIT Trains the QDA Model
            X = double(X);
            y = double(y);
            
            [N, p] = size(X);
            obj.NumFeatures = p;
            
            obj.Classes = unique(y);
            obj.NumClasses = length(obj.Classes);
            K = obj.NumClasses;
            
            obj.Means = zeros(K, p);         % Means (Centroids)
            obj.Priors = zeros(K, 1);        % Priors
            obj.Sigma = zeros(p, p, K);      % Covariances
            obj.LogDet = zeros(K, 1);        % Log-Determinants
            
            for k = 1:K
                c = obj.Classes(k);
                
                % Extract training samples for class 'k'
                idx = (y == c);
                X_k = X(idx, :);
                
                % 1. Estimate Priors
                obj.Priors(k) = sum(idx) / N;
                
                % 2. Estimate Means
                obj.Means(k, :) = mean(X_k, 1);
                
                % 3. Estimate Covariance
                % We add a small regularization term to the diagonal to ensure 
                % positive-definiteness (numerical stability if N_k is small)
                Sig_k = cov(X_k) + 1e-6 * eye(p);
                obj.Sigma(:, :, k) = Sig_k;
                
                % 4. Pre-compute Log Determinant
                try
                    L = chol(Sig_k); 
                    obj.LogDet(k) = 2 * sum(log(diag(L)));
                catch 
                    obj.LogDet(k) = log(det(Sig_k));
                end
            end
        end
        
        function [y_pred, posterior] = predict(obj, X)
            % PREDICT Predicts classes and posterior probabilities
            if isempty(obj.Classes)
                error('Model must be fitted before prediction.');
            end
            
            X = double(X);
            [M, p] = size(X);
            K = obj.NumClasses;
            
            if p ~= obj.NumFeatures
                error('Number of features in test data does not match training data.');
            end
            
            scores = zeros(M, K);
            
            for k = 1:K
                mu = obj.Means(k, :)';
                Sig = obj.Sigma(:, :, k);
                log_det = obj.LogDet(k);
                prior = obj.Priors(k);
                
                % Center the data
                centered = bsxfun(@minus, X, mu');
                
                % Standardize by Inverse Covariance
                standardized = centered / Sig;
                
                % Mahalanobis Distance squared
                dist = sum(standardized .* centered, 2);
                
                % Discriminant Score: -0.5*ln|Sigma| - 0.5*dist + ln(prior)
                scores(:, k) = -0.5 * log_det - 0.5 * dist + log(prior);
            end
            
            % Maximum A Posteriori Assignment
            [~, maxIdx] = max(scores, [], 2);
            y_pred = obj.Classes(maxIdx);
            
            if nargout > 1
                % Convert discriminant log-scores to probabilities
                maxScores = max(scores, [], 2);
                expScores = exp(bsxfun(@minus, scores, maxScores));
                posterior = bsxfun(@rdivide, expScores, sum(expScores, 2));
            end
        end
    end
end