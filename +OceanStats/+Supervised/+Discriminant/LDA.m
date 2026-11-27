classdef LDA < OceanStats.Supervised.Discriminant.IDiscriminantModel
    % LDA Linear Discriminant Analysis Classifier
    %
    % Description:
    %   Implements standard global Linear Discriminant Analysis.
    %   Assumes all classes share a common pooled covariance matrix.
    %   Provides exact Bayes classification under this assumption as well as
    %   dimensionality reduction via canonical variates.
    %
    % Usage:
    %   model = OceanStats.Supervised.Discriminant.LDA();
    %   model.fit(X_train, Y_train);
    %   preds = model.predict(X_test);
    %   coords = model.project(X_test, 2); % Project to 2D canonical space
    
    properties (SetAccess = private)
        Means           % K x p matrix of class centroids
        Sigma           % p x p pooled within-class covariance
        InvSigma        % p x p inverse covariance
        NumFeatures     % Integer p
        
        Scaling         % p x (K-1) matrix for transforming to canonical space
    end
    
    methods
        function obj = LDA(varargin)
            % LDA Constructor
        end
        
        function fit(obj, X, y)
            % FIT Trains the LDA model
            X = double(X);
            y = double(y);
            
            [N, p] = size(X);
            obj.NumFeatures = p;
            obj.Classes = unique(y);
            obj.NumClasses = length(obj.Classes);
            
            K = obj.NumClasses;
            
            obj.Priors = zeros(K, 1);
            obj.Means = zeros(K, p);
            obj.Sigma = zeros(p, p);
            
            % Compute Priors, Means, and Pooled Covariance
            for k = 1:K
                c = obj.Classes(k);
                idx = (y == c);
                N_k = sum(idx);
                
                if N_k == 0
                    continue;
                end
                
                obj.Priors(k) = N_k / N;
                X_k = X(idx, :);
                mu_k = mean(X_k, 1);
                obj.Means(k, :) = mu_k;
                
                % Accumulate centered scatter
                X_centered = bsxfun(@minus, X_k, mu_k);
                obj.Sigma = obj.Sigma + (X_centered' * X_centered);
            end
            
            % Unbiased estimate of pooled covariance
            % Denominator is N - K
            if N > K
                obj.Sigma = obj.Sigma / (N - K);
            else
                obj.Sigma = obj.Sigma / N;
            end
            
            % Robust covariance inversion via SVD (avoids numerical instability
            % from pinv(Sigma) * BetweenScatter when Sigma is rank-deficient,
            % e.g. when Sigma is computed from highly collinear Z_hat from FDA).
            [U_s, S_s, ~] = svd(obj.Sigma, 'econ');
            sv = diag(S_s);
            % Dynamic rank threshold: keep singular values above max * eps^0.5
            tol_sv = max(sv) * max(size(obj.Sigma)) * sqrt(eps);
            r = sum(sv > tol_sv);
            
            if r == 0
                % Degenerate: all singular values below threshold (empty node)
                obj.InvSigma = eye(p) / (tol_sv + eps);
                obj.Scaling  = eye(p, min(p, K - 1));
                return;
            end
            
            % Truncated pseudo-inverse — more stable than adding ridge then inverting
            U_r  = U_s(:, 1:r);
            S_r  = diag(sv(1:r));
            obj.InvSigma = U_r * diag(1 ./ sv(1:r)) * U_r';
            
            % ----------------------------------------------------------------
            % Canonical Variates via SVD-based whitening
            % Algorithm (avoids ill-conditioned W^-1 B product):
            %   1. Whiten Sigma:  W_half = S_r^{-1/2} U_r'   (r x p)
            %   2. Project BetweenScatter into whitened space: B_w = W_half B W_half'
            %   3. Eigen-decompose small r x r symmetric B_w  (stable)
            %   4. Unwhiten back:  Scaling = W_half' V_w
            % ----------------------------------------------------------------
            globalMean = sum(bsxfun(@times, obj.Means, obj.Priors), 1);
            BetweenScatter = zeros(p, p);
            for k = 1:K
                c = obj.Classes(k);
                N_k = sum(y == c);
                mu_diff = (obj.Means(k, :) - globalMean)';
                BetweenScatter = BetweenScatter + N_k * (mu_diff * mu_diff');
            end
            BetweenScatter = BetweenScatter / N;
            
            % Whitening transform  (r x p)
            % Rotate to principal components
            %  Then Stretches/Compresses to unit variance
            W_half = diag(1 ./ sqrt(sv(1:r))) * U_r';
            
            % Reduce to r x r problem — perfectly conditioned by construction
            B_w = W_half * BetweenScatter * W_half';
            B_w = (B_w + B_w') / 2;    % enforce symmetry (eliminate rounding artefacts)
            
            [V_w, D_w] = eig(B_w);     % safe: B_w is small, symmetric, positive semi-definite
            evals = real(diag(D_w));
            [~, sortIdx] = sort(evals, 'descend');
            V_w = real(V_w(:, sortIdx));
            
            % Unwhiten back to original p-dimensional space
            V_full = W_half' * V_w;
            
            % Keep up to K-1 meaningful discriminant directions
            maxDims = min(r, K - 1);
            obj.Scaling = V_full(:, 1:maxDims);
        end
        
        function [y_pred, posterior] = predict(obj, X)
            % PREDICT Assigns classes to new observations
            if isempty(obj.Classes)
                error('Model must be fitted before prediction.');
            end
            
            X = double(X);
            M = size(X, 1);
            K = obj.NumClasses;
            scores = zeros(M, K);
            
            % Discriminant function: d_k(x) = x^T Sigma^-1 mu_k - 0.5 mu_k^T Sigma^-1 mu_k + log(pi_k)
            for k = 1:K
                mu = obj.Means(k, :)';
                prior = obj.Priors(k);
                
                term1 = X * obj.InvSigma * mu;
                term2 = 0.5 * (mu' * obj.InvSigma * mu);
                term3 = log(prior + eps);
                
                scores(:, k) = term1 - term2 + term3;
            end
            
            [~, maxIdx] = max(scores, [], 2);
            y_pred = obj.Classes(maxIdx);
            
            if nargout > 1
                % Convert discriminant log-scores to probabilities via softmax
                maxScores = max(scores, [], 2);
                expScores = exp(bsxfun(@minus, scores, maxScores));
                posterior = bsxfun(@rdivide, expScores, sum(expScores, 2));
            end
        end
        
        function coords = project(obj, X, dims)
            % PROJECT Transforms data into the canonical discriminant space
            % Optional dims argument limits the number of output dimensions
            if isempty(obj.Scaling)
                error('Model must be fitted before projection.');
            end
            
            X = double(X);
            % Center data using global mean of class centroids weighted by prior
            % Note: In canonical space, centering doesn't strictly matter for relative spread,
            % but traditionally we center it.
            globalMu = sum(bsxfun(@times, obj.Means, obj.Priors), 1);
            X_c = bsxfun(@minus, X, globalMu);
            
            coords = X_c * obj.Scaling;
            
            if nargin > 2 && dims < size(coords, 2)
                coords = coords(:, 1:dims);
            end
        end
    end
end
