classdef LocalLDA < handle
    % LOCALLDA Local Linear Discriminant Analysis Classifier
    %
    % Description:
    %   Implements Local LDA as described in ESL Exercise 6.12.
    %   It fits a weighted LDA model locally for each query point.
    %
    % Usage:
    %   model = LocalLDA('Bandwidth', 5.0);
    %   model.fit(X_train, y_train);
    %   preds = model.predict(X_test);
    %
    % Properties:
    %   Bandwidth  : The kernel width (lambda).
    %   RidgeParam : Regularization for covariance stability.
    
    properties
        Bandwidth = 1.0;      % Default lambda
        RidgeParam = 1e-4;    % Default ridge
    end
    
    properties (SetAccess = private)
        X_train
        y_train
        Classes
        NumClasses
        NumFeatures
    end
    
    methods
        % --- Constructor ---
        function obj = LocalLDA(varargin)
            % Parse optional Name-Value pairs
            p = inputParser;
            addParameter(p, 'Bandwidth', 1.0);
            addParameter(p, 'RidgeParam', 1e-4);
            parse(p, varargin{:});
            
            obj.Bandwidth = p.Results.Bandwidth;
            obj.RidgeParam = p.Results.RidgeParam;
        end
        
        % --- Fit Method (Lazy Learning) ---
        function fit(obj, X, y)
            % Stores data. No computation happens here because Local LDA
            % is instance-based (lazy).
            obj.X_train = double(X);
            obj.y_train = double(y);
            obj.Classes = unique(y);
            obj.NumClasses = length(obj.Classes);
            obj.NumFeatures = size(X, 2);
        end
        
        % --- Predict Method ---
        function preds = predict(obj, X_query)
            X_query = double(X_query);
            M = size(X_query, 1);
            preds = zeros(M, 1);
            
            % Check if fitted
            if isempty(obj.X_train)
                error('Model is not fitted. Call fit(X, y) first.');
            end
            
            %Pre-fetch properties (Crucial for parfor efficiency)
            %parfor cannot access 'obj' properties directly and
            %efficiently. We copy them to local variables first.
            TrainX = obj.X_train;
            TrainY = obj.y_train;
            ClassesList = obj.Classes;
            nClasses = obj.NumClasses;
            nFeat = obj.NumFeatures;
            h = obj.Bandwidth;
            ridge = obj.RidgeParam;

            % --- PARALLEL LOOP ---
            % 'parfor' splits these M iterations across CPU cores.
            parfor i = 1:M
                x0 = X_query(i, :);
                
                % 1. Compute Weights
                dists_sq = sum(bsxfun(@minus, obj.X_train, x0).^2, 2);
                weights = exp(-dists_sq / (2 * obj.Bandwidth^2));
                sum_w = sum(weights);
                
                if sum_w < 1e-15
                    weights = ones(size(weights)); % Fallback to global
                    sum_w = sum(weights);
                end
                
                % 2. Compute Local Statistics
                LocalMeans = zeros(obj.NumClasses, obj.NumFeatures);
                LocalPriors = zeros(obj.NumClasses, 1);
                PooledCov = zeros(obj.NumFeatures, obj.NumFeatures);
                
                for k = 1:obj.NumClasses
                    c = obj.Classes(k);
                    idx = (obj.y_train == c);
                    
                    X_k = obj.X_train(idx, :);
                    w_k = weights(idx);
                    sum_w_k = sum(w_k);
                    
                    % Prior
                    LocalPriors(k) = sum_w_k / sum_w;
                    
                    % Mean
                    if sum_w_k > 0
                        mu_k = sum(bsxfun(@times, X_k, w_k), 1) / sum_w_k;
                    else
                        mu_k = zeros(1, obj.NumFeatures);
                    end
                    LocalMeans(k, :) = mu_k;
                    
                    % Covariance Accumulation
                    X_centered = bsxfun(@minus, X_k, mu_k);
                    X_weighted = bsxfun(@times, X_centered, w_k);
                    PooledCov = PooledCov + (X_centered' * X_weighted);
                end
                
                PooledCov = PooledCov / sum_w;
                PooledCov = PooledCov + eye(obj.NumFeatures) * obj.RidgeParam;
                InvSigma = pinv(PooledCov);
                
                % 3. Discriminant Rule
                best_score = -inf;
                best_class = NaN;
                
                for k = 1:obj.NumClasses
                    mu = LocalMeans(k, :)';
                    prior = LocalPriors(k);
                    
                    term1 = x0 * InvSigma * mu;
                    term2 = 0.5 * (mu' * InvSigma * mu);
                    
                    if prior > 1e-15
                        term3 = log(prior);
                    else
                        term3 = -1e9;
                    end
                    
                    score = term1 - term2 + term3;
                    
                    if score > best_score
                        best_score = score;
                        best_class = obj.Classes(k);
                    end
                end
                preds(i) = best_class;
                
                % Progress for large queries
                if mod(i, 100) == 0 && M > 500
                    fprintf('Processed %d / %d...\n', i, M);
                end
            end
        end
    end
end