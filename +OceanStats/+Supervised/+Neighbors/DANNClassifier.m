classdef DANNClassifier < handle
    % DANNCLASSIFIER  Discriminant Adaptive Nearest Neighbors (ESL 13.4.1).
    %
    % ALGORITHM:
    %   For each query x0:
    %     1. Find Knn Euclidean nearest neighbors
    %     2. Compute local within-class (W) and between-class (B) covariances
    %     3. Build adaptive metric: Sigma = W^{-1/2} (W^{-1/2} B W^{-1/2} + eps*I) W^{-1/2}
    %     4. Re-rank neighbors by Mahalanobis distance with Sigma
    %     5. Classify using top K neighbors under the adaptive metric
    %
    %   Near decision boundaries, Sigma stretches neighborhoods orthogonally
    %   to the boundary. In pure regions, Sigma ≈ spherical.
    %
    % USAGE:
    %   dann = OceanStats.Supervised.Neighbors.DANNClassifier('K', 15, 'Knn', 50);
    %   dann.fit(X_train, y_train);
    %   y_pred = dann.predict(X_test);
    %   [Sigma, nn_idx] = dann.getLocalMetric(x0);  % for visualization
    %
    % Ref: Hastie & Tibshirani (1996), Hastie et al. (2009) ESL Section 13.4.1
    
    properties
        K       (1,1) double {mustBePositive, mustBeInteger} = 15   % final neighbors for vote
        Knn     (1,1) double {mustBePositive, mustBeInteger} = 50   % neighbors to estimate metric
        Epsilon (1,1) double {mustBeNonnegative} = 1.0              % regularization (strip→ellipse)
        
        % Stored training data (set during fit)
        X_tr    double   % (N_train x p)
        y_tr    double   % (N_train x 1)
        Classes double   % (1 x nClasses) sorted unique labels
    end
    
    methods
        function obj = DANNClassifier(opts)
            % DANNCLASSIFIER  Constructor.
            %   dann = DANNClassifier('K', 15, 'Knn', 50, 'Epsilon', 1)
            arguments
                opts.K       (1,1) double = 15
                opts.Knn     (1,1) double = 50
                opts.Epsilon (1,1) double = 1.0
            end
            obj.K       = opts.K;
            obj.Knn     = opts.Knn;
            obj.Epsilon = opts.Epsilon;
        end
        
        function fit(obj, X, y)
            % FIT  Store training data (lazy — no explicit training phase).
            obj.X_tr    = X;
            obj.y_tr    = y(:);
            obj.Classes = unique(y(:))';
        end
        
        function y_pred = predict(obj, X_test)
            % PREDICT  Classify via DANN adaptive metric + majority vote.
            probs = obj.predictProba(X_test);
            [~, ci] = max(probs, [], 2);
            y_pred = obj.Classes(ci)';
        end
        
        function probs = predictProba(obj, X_test)
            % PREDICTPROBA  Class fractions from adaptive K neighbors.
            N_te = size(X_test, 1);
            nC   = numel(obj.Classes);
            probs = zeros(N_te, nC);
            
            for i = 1:N_te
                [~, nn_idx] = obj.getLocalMetric(X_test(i,:));
                nn_y = obj.y_tr(nn_idx);
                for ci = 1:nC
                    probs(i, ci) = mean(nn_y == obj.Classes(ci));
                end
            end
        end
        
        function [Sigma, nn_idx] = getLocalMetric(obj, x0)
            % GETLOCALMETRIC  Compute the DANN metric at a single query point.
            %
            %   [Sigma, nn_idx] = dann.getLocalMetric(x0)
            %
            %   Sigma  : (p x p) local metric matrix
            %   nn_idx : (K x 1) indices of adaptive K nearest neighbors
            %
            % The metric Sigma defines a Mahalanobis distance:
            %   d(x, x0) = (x - x0)' * Sigma * (x - x0)
            
            x0 = x0(:)';
            [N_tr, p] = size(obj.X_tr);
            
            % 1. Euclidean nearest neighbors
            diffs = obj.X_tr - x0;
            d2_euc = sum(diffs.^2, 2);
            [~, sortIdx] = sort(d2_euc);
            Knn_eff = min(obj.Knn, N_tr);
            nn_euc = sortIdx(1:Knn_eff);
            
            X_nn = obj.X_tr(nn_euc, :);
            y_nn = obj.y_tr(nn_euc);
            
            % 2. Compute within-class covariance W
            W = zeros(p, p);
            classes_local = unique(y_nn)';
            for c = classes_local
                X_c = X_nn(y_nn == c, :);
                n_c = size(X_c, 1);
                if n_c > 1
                    W = W + (n_c - 1) * cov(X_c);
                end
            end
            W = W / Knn_eff;
            
            % Regularize W for numerical stability
            W = W + 1e-6 * eye(p);
            
            % 3. Compute between-class covariance B
            mu_all = mean(X_nn, 1);
            B = zeros(p, p);
            for c = classes_local
                X_c = X_nn(y_nn == c, :);
                n_c = size(X_c, 1);
                mu_c = mean(X_c, 1);
                d = (mu_c - mu_all)';
                B = B + n_c * (d * d');
            end
            B = B / Knn_eff;
            
            % 4. Build local metric: Sigma = W^{-1/2} (W^{-1/2} B W^{-1/2} + eps*I) W^{-1/2}
            [V_w, D_w] = eig(W);
            d_w = diag(D_w);
            d_w = max(d_w, 1e-10);
            W_inv_half = V_w * diag(1 ./ sqrt(d_w)) * V_w';
            
            M = W_inv_half * B * W_inv_half + obj.Epsilon * eye(p);
            Sigma = W_inv_half * M * W_inv_half;
            
            % Symmetrize for numerical safety
            Sigma = (Sigma + Sigma') / 2;
            
            % 5. Re-rank ALL training points using the adaptive metric
            diffs_all = obj.X_tr - x0;
            d2_adapt = sum((diffs_all * Sigma) .* diffs_all, 2);
            [~, sortIdx2] = sort(d2_adapt);
            K_eff = min(obj.K, N_tr);
            nn_idx = sortIdx2(1:K_eff);
        end
    end
end
