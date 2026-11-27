classdef MDA2 < OceanStats.Supervised.Discriminant.IDiscriminantModel
    % MDA2  Shared-Center Mixture Discriminant Analysis (ESL Ex 12.11).
    %
    % THEORY:
    %   Generalization of MDA where R Gaussian centers are SHARED across
    %   all K classes.  Each center r has a class profile P_r(G=k) that
    %   describes its association with each class.
    %
    %   P(G,X) = Sum_r pi_r * P_r(G) * phi(x; mu_r, Sigma)   (eq 12.67)
    %
    %   When initialized with exclusive K-means assignments (each center
    %   belongs to exactly one class), MDA2 reduces to standard MDA (part c).
    %
    % ARCHITECTURE:
    %   This is a thin wrapper.  Training is delegated to:
    %     MDA2Model       (pure state)
    %     Recipe_MDA2_EM  (E/M steps, IMMRecipe)
    %     GenericOptimizer (iteration loop)
    %
    % PROPERTIES:
    %   R        - Number of shared centers (default: 10)
    %   MaxIter  - EM iterations (default: 100)
    %   Tol      - Convergence tolerance (default: 1e-4)
    %   Verbose  - Print progress (default: false)
    %
    % USAGE:
    %   model = OceanStats.Supervised.Discriminant.MDA2('R', 10);
    %   model.fit(X_train, y_train);
    %   y_pred = model.predict(X_test);
    %   coords = model.project(X_test, 2);
    %
    % See also: MDA, MDA2Model, Recipe_MDA2_EM
    %
    % Author: Ocean Library System
    % Date:   2026-03-24

    properties
        R        (1,1) double  = 10
        MaxIter  (1,1) double  = 100
        Tol      (1,1) double  = 1e-4
        Verbose  (1,1) logical = false

        % Internal state (set during fit)
        Model     % MDA2Model instance
    end

    methods
        function obj = MDA2(varargin)
            p = inputParser;
            addParameter(p, 'R', 10);
            addParameter(p, 'MaxIter', 100);
            addParameter(p, 'Tol', 1e-4);
            addParameter(p, 'Verbose', false);
            parse(p, varargin{:});

            obj.R       = p.Results.R;
            obj.MaxIter = p.Results.MaxIter;
            obj.Tol     = p.Results.Tol;
            obj.Verbose = p.Results.Verbose;
        end

        function fit(obj, X, y)
            % FIT  Train MDA2 via EM on X (N x p) with labels y (N x 1).

            X = double(X);
            y = double(y(:));
            [N, D] = size(X);

            obj.Classes    = unique(y);
            obj.NumClasses = numel(obj.Classes);
            K = obj.NumClasses;
            R = obj.R;

            % Class priors
            obj.Priors = zeros(K, 1);
            for k = 1:K
                obj.Priors(k) = sum(y == obj.Classes(k)) / N;
            end

            % --- Initialize MDA2Model ---
            mdl = OceanStats.Inference.Models.MDA2Model(R, K, D);
            mdl.Classes     = obj.Classes;
            mdl.ClassPriors = obj.Priors;

            % K-means on full dataset for initial centers
            [idx, C] = kmeans(X, R, 'Replicates', 3, 'EmptyAction', 'singleton');
            mdl.Mu = C;

            % Initial mixing weights
            mdl.Pi = zeros(R, 1);
            for r = 1:R
                mdl.Pi(r) = sum(idx == r) / N;
            end

            % Initial class profiles from observed class distribution in each cluster
            mdl.ClassProfile = zeros(R, K);
            for r = 1:R
                members = y(idx == r);
                for k = 1:K
                    mdl.ClassProfile(r, k) = sum(members == obj.Classes(k)) / (numel(members) + eps);
                end
            end

            % Initial pooled covariance
            Sigma_init = zeros(D, D);
            for r = 1:R
                diff = X(idx == r, :) - C(r, :);
                Sigma_init = Sigma_init + (diff' * diff);
            end
            mdl.Sigma = Sigma_init / N + eye(D) * mdl.Regularization;

            obj.Model = mdl;

            % --- Assemble Recipe + Optimizer ---
            recipe    = OceanStats.Inference.Recipes.Recipe_MDA2_EM(mdl, obj.Tol);
            optimizer = OceanStats.Optimization.GenericOptimizer( ...
                'MaxIter', obj.MaxIter, 'Verbose', obj.Verbose);

            % Data format for MM framework: [X, y]
            data = [X, y];
            optimizer.run(recipe, data);

            if obj.Verbose
                fprintf('MDA2: Fitted %d shared centers across %d classes.\n', R, K);
            end
        end

        function [y_pred, posterior] = predict(obj, X)
            % PREDICT  Classify via posterior P(G=k|X) (eq 12.69).
            %
            % P(G=k|x) = Sum_r pi_r * P_r(G=k) * phi(x; mu_r, Sigma)
            %            / Sum_r pi_r * phi(x; mu_r, Sigma)  [marginalized over G]

            X = double(X);
            [M, ~] = size(X);
            K = obj.NumClasses;
            R = obj.R;
            mdl = obj.Model;

            if isempty(mdl)
                error('MDA2:NotFitted', 'Model must be fitted before predict.');
            end

            % Log Gaussian densities: (M x R)
            logPhi = mdl.computeLogPhi(X);

            logPost = zeros(M, K);
            for k = 1:K
                % log[ Sum_r pi_r * P_r(G=k) * phi(x; mu_r, Sigma) ] + log(P(G=k))
                log_terms = log(mdl.Pi + eps) ...
                          + log(mdl.ClassProfile(:, k) + eps) ...
                          + logPhi';  % (R x M) — need transpose

                % Actually logPhi is (M x R), so:
                log_terms_mat = logPhi + log(mdl.Pi + eps)' + log(mdl.ClassProfile(:, k) + eps)';  % (M x R)

                % Log-sum-exp per sample
                maxVal = max(log_terms_mat, [], 2);
                logPost(:, k) = maxVal + log(sum(exp(log_terms_mat - maxVal), 2));
                logPost(:, k) = logPost(:, k) + log(obj.Priors(k));
            end

            % Classify by maximum log-posterior
            [~, maxIdx] = max(logPost, [], 2);
            y_pred = obj.Classes(maxIdx);

            % Normalized posterior probabilities
            if nargout > 1
                maxLogPost = max(logPost, [], 2);
                expPost = exp(logPost - maxLogPost);
                posterior = expPost ./ sum(expPost, 2);
            end
        end

        function X_proj = project(obj, X, dims)
            % PROJECT  Canonical variate projection via W^(-1)B eigendecomposition.
            %
            % W = pooled within-center covariance (= Model.Sigma)
            % B = between-center covariance (weighted by effective counts)

            if nargin < 3, dims = 2; end

            mdl = obj.Model;
            if isempty(mdl)
                error('MDA2:NotFitted', 'Model must be fitted before project.');
            end

            % Between-center covariance
            globalMean = mdl.Pi' * mdl.Mu;  % (1 x p) weighted mean
            B = zeros(mdl.p, mdl.p);
            for r = 1:mdl.R
                d = mdl.Mu(r, :) - globalMean;
                B = B + mdl.Pi(r) * (d' * d);
            end

            W = mdl.Sigma;

            % Generalized eigenvalue: B v = lambda W v
            [V, D_eig] = eig(B, W);
            eigenvalues = real(diag(D_eig));
            [~, sortIdx] = sort(eigenvalues, 'descend');
            V = real(V(:, sortIdx));
            V_top = V(:, 1:min(dims, size(V, 2)));

            % Normalize: V' W V = I
            for j = 1:size(V_top, 2)
                v = V_top(:, j);
                V_top(:, j) = v / sqrt(v' * W * v + eps);
            end

            X_proj = double(X) * V_top;
        end
    end
end
