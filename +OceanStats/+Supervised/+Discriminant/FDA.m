classdef FDA < OceanStats.Supervised.Discriminant.IDiscriminantModel
    % FDA Flexible Discriminant Analysis Wrapper
    %  Assumptions: Sufficient Base Learner Capacity
    %   Homoscedasticity in the Transformed space
    %   Mutually Exclusive Classes
    %   Motivation:
    %   Standard Linear Discriminant Analysis (LDA) is computationally efficient and 
    %   mathematically stable, but it is fundamentally restricted to drawing rigid, fl  at decision boundaries. 
    %   Real-world machine learning data is highly non-linear. Rather than abandoning the elegant probability framework of LDA to build an entirely new non-linear classifier from scratch, we need a mechanism to leverage existing, powerful non-linear regression algorithms (like MARS or GAMs) to handle the complex geometry, while retaining LDA for the final classification.
    % Description:
    %   Implements Flexible Discriminant Analysis (FDA) via the Optimal
    %   Scoring algorithm (Hastie, Tibshirani, Buja 1994).
    %   It acts as a wrapper that elevates ANY multi-response regression 
    %   model into a multi-class discriminant classifier.
    %   Architecture: Wrapper/Delegation Architecture
    %
    % Usage:
    %   % Wrap MARS
    %   marsBuilder = @() OceanStats.Supervised.MARS.MARS('Degree', 2);
    %   model = OceanStats.Supervised.Discriminant.FDA('BaseModelConstructor', marsBuilder);
    %   
    %   model.fit(X_train, y_train);
    %   preds = model.predict(X_test);
    %   coords = model.project(X_test, 2); % Project to 2D canonical space
    %
    % FitMethod Options:
    %   'LDA'            (default) — After multi-response regression, fits a
    %                    standard LDA on the fitted values Z_hat. Simpler, reuses
    %                    existing LDA code. Classification results are equivalent.
    %
    %   'OptimalScoring' (textbook) — Implements ESL p.445 exactly. After
    %                    multi-response regression, computes the eigen-decomposition
    %                    of Y'*Y_hat = Y'*S_lambda*Y, normalized so that
    %                    Theta'*D_pi*Theta = I (D_pi = class priors).
    %                    Produces the textbook-exact canonical score matrix Theta.
    %                    Better handles class imbalance via D_pi normalization.
    %
    % Extended Usage (FitMethod):
    %   fda = OceanStats.Supervised.Discriminant.FDA(...
    %       'BaseModelConstructor', marsBuilder, 'FitMethod', 'OptimalScoring');
    %   fda.fit(X_train, y_train);
    %   preds  = fda.predict(X_test);
    %   coords = fda.project(X_test, 2); % Canonical 2D projection
    
    properties
        % Function handle returning a fresh IModelStrategy regression instance.
        % E.g. @() OceanStats.Supervised.MARS.MARS('MaxTerms', 15)
        BaseModelConstructor 
        
        % Fitting strategy: 'LDA' (default) or 'OptimalScoring' (textbook)
        FitMethod (1,1) string = "LDA"
    end
    
    properties (SetAccess = private)
        K_Models     % 1 x K cell of fitted base regressors (one per class)
        InternalLDA  % LDA instance (only used when FitMethod == 'LDA')
        Theta        % K x (K-1) canonical score matrix (OptimalScoring only)
        Z_bar        % 1 x K class prior probabilities for centering (OptimalScoring)
    end
    
    methods
        function obj = FDA(varargin)
            p = inputParser;
            addParameter(p, 'BaseModelConstructor', []);
            addParameter(p, 'FitMethod', 'LDA');
            parse(p, varargin{:});
            
            obj.BaseModelConstructor = p.Results.BaseModelConstructor;
            obj.FitMethod = string(p.Results.FitMethod);
            
            if isempty(obj.BaseModelConstructor)
                error('FDA requires a BaseModelConstructor, e.g. @() MARS().');
            end
            if ~ismember(obj.FitMethod, ["LDA", "OptimalScoring"])
                error('FDA: FitMethod must be ''LDA'' or ''OptimalScoring''.');
            end
        end
        
        function fit(obj, X, y)
            X = double(X);
            y = double(y);
            
            [N, ~]      = size(X);
            obj.Classes = unique(y);
            obj.NumClasses = length(obj.Classes);
            K = obj.NumClasses;
            
            % ── Step 1: Build indicator matrix Z (N x K) ──────────────────
            Z = zeros(N, K);
            for k = 1:K
                Z(y == obj.Classes(k), k) = 1;
            end
            
            % ── Step 2: Multi-Response Regression ─────────────────────────
            % Fit K independent regressors, one per class indicator column.
            obj.K_Models = cell(K, 1);
            Z_hat = zeros(N, K);
            for k = 1:K
                model_k = obj.BaseModelConstructor();
                model_k.fit(X, Z(:, k));
                obj.K_Models{k} = model_k;
                Z_hat(:, k) = model_k.predict(X);
            end
            
            % ── Step 3: Discriminant Direction Extraction ──────────────────
            switch obj.FitMethod
                
                case "LDA"
                    % ── Option A: delegate to LDA on Z_hat ────────────────
                    % ESL p.431: "apply standard LDA to the fitted values Y_hat"
                    % Simple, reuses existing LDA class. Classification accuracy
                    % is equivalent to OptimalScoring for balanced classes.
                    obj.InternalLDA = OceanStats.Supervised.Discriminant.LDA();
                    obj.InternalLDA.fit(Z_hat, y);
                    obj.Priors = obj.InternalLDA.Priors;
                    
                case "OptimalScoring"
                    %
                    % ── Option B: Textbook Optimal Scoring (ESL p.445) ────
                    % Eigen-decomposition of W = Y'*Y_hat normalized by D_pi.
                    %
                    % W  = Z' * Z_hat                    (K x K cross-product)
                    % Dp = diag(Z' * Z / N)              (K x K diagonal priors)
                    % Solve: W * Theta = Dp * Theta * Lambda  (generalized eig)
                    % Normalize: Theta' * Dp * Theta = I
                    %
                    % Result: Theta is the K x (K-1) canonical score matrix.
                    
                    % Class prior probabilities (diagonal of D_pi)
                    pi_k = zeros(K, 1);
                    for k = 1:K
                        pi_k(k) = sum(y == obj.Classes(k)) / N;
                    end
                    Dp = diag(pi_k); % K x K diagonal prior matrix
                    
                    % Cross-product W = Z' * Z_hat  (K x K)
                    %  Z is N x K indicator matrix
                    %  Z_hat is N x K fitted values matrix (Soft continuous score form regressors)
                    %   W(j, k) : is sample i truly in class j? and how strongly does the regressor predict class k for sample i?.
                    W = Z' * Z_hat / N;
                    
                    % Generalized eigenvalue problem:  W * Theta = Dp * Theta * Lambda
                    % Transform to standard form: Dp^{-1/2} W Dp^{-1/2} v = lambda v
                    % then unscale: Theta = Dp^{-1/2} * v
                    D_half_inv = diag(1 ./ (sqrt(pi_k) + eps));  % K x K
                    W_sym = D_half_inv * W * D_half_inv;
                    W_sym = (W_sym + W_sym') / 2;  % enforce exact symmetry
                    
                    [V, Lambda] = eig(W_sym);
                    evals = real(diag(Lambda));
                    [~, sortIdx] = sort(evals, 'descend');
                    V = real(V(:, sortIdx));
                    
                    % Unscale to get Theta in original K-space
                    Theta_raw = D_half_inv * V;
                    
                    % Keep only K-1 directions (rank of B is at most K-1)
                    maxDims = K - 1;
                    obj.Theta = Theta_raw(:, 1:maxDims);  % K x (K-1)
                    obj.Z_bar = pi_k;            % store priors for centering at predict
                    obj.Priors = pi_k;
            end
        end
        
        function [y_pred, posterior] = predict(obj, X)
            if isempty(obj.Classes)
                error('FDA model is not fitted.');
            end
            
            X = double(X);
            M = size(X, 1);
            K = obj.NumClasses;
            
            % Compute fitted values on test data via the K base regressors
            Z_hat_test = zeros(M, K);
            for k = 1:K
                Z_hat_test(:, k) = obj.K_Models{k}.predict(X);
            end
            
            switch obj.FitMethod
                
                case "LDA"
                    if nargout > 1
                        [y_pred, posterior] = obj.InternalLDA.predict(Z_hat_test);
                    else
                        y_pred = obj.InternalLDA.predict(Z_hat_test);
                    end
                    
                case "OptimalScoring"
                    % eta(x) = Theta' * eta*(x)
                    % Scores live in (K-1)-dimensional canonical space.
                    % Classify by nearest class centroid in that space.
                    eta = Z_hat_test * obj.Theta;  % M x (K-1)
                    
                    % Compute canonical class centroids from training priors
                    % (one-hot centroids projected through Theta)
                    % centroid_k = e_k' * Theta  where e_k is the k-th unit vector
                    centroids = obj.Theta;  % K x (K-1): each row is a class centroid
                    
                    % Assign to nearest centroid (Euclidean distance)
                    dists = zeros(M, K);
                    for k = 1:K
                        diff = bsxfun(@minus, eta, centroids(k, :));
                        dists(:, k) = sum(diff.^2, 2);
                    end
                    [~, minIdx] = min(dists, [], 2);
                    y_pred = obj.Classes(minIdx);
                    
                    if nargout > 1
                        % Soft-min distances as approximate posterior
                        neg_dists = -dists;
                        maxD = max(neg_dists, [], 2);
                        expD = exp(bsxfun(@minus, neg_dists, maxD));
                        posterior = bsxfun(@rdivide, expD, sum(expD, 2));
                    end
            end
        end
        
        function coords = project(obj, X, dims)
            if isempty(obj.K_Models)
                error('FDA model is not fitted.');
            end
            
            X = double(X);
            M = size(X, 1);
            K = obj.NumClasses;
            
            Z_hat_test = zeros(M, K);
            for k = 1:K
                Z_hat_test(:, k) = obj.K_Models{k}.predict(X);
            end
            
            switch obj.FitMethod
                case "LDA"
                    coords = obj.InternalLDA.project(Z_hat_test, dims);
                    
                case "OptimalScoring"
                    % eta(x) = Theta' * eta*(x) projected via Theta
                    coords = Z_hat_test * obj.Theta;
                    if nargin > 2 && dims < size(coords, 2)
                        coords = coords(:, 1:dims);
                    end
            end
        end

        function S = getScaling(obj)
            % GETSCALING Returns the discriminant scaling matrix in original X space (p x K-1).
            %
            % Output:
            %   S - (p x K-1) matrix of canonical discriminant directions in the
            %       original feature space (e.g., 256 x 9 for digit recognition).
            %
            % How it works:
            %   FDA fits K linear base regressors, each producing a (p x 1) coefficient
            %   vector Beta_k. Stacking gives B_mat (p x K). InternalLDA.Scaling is
            %   (K x K-1) — the canonical directions in Z_hat space. Composing:
            %
            %     S = B_mat * InternalLDA.Scaling    (p x K-1)
            %
            %   gives the discriminant directions mapped back into the original p-space.
            %   These are what appear as the coefficient images in ESL Figure 12.11.
            %
            % Requirements:
            %   Base models must expose a 'Beta' property (e.g., PenalizedRegression,
            %   LinearModel). For nonlinear models (MARS), no closed-form X-space
            %   composition exists — use project() for projections instead.
            %
            % Notes:
            %   For FitMethod = 'OptimalScoring', canonical directions live in Z_hat
            %   space via obj.Theta (K x K-1). No X-space scaling matrix exists.
            
            if isempty(obj.K_Models)
                error('FDA:NotFitted', 'FDA model must be fitted before calling getScaling().');
            end
            
            switch obj.FitMethod
                case "LDA"
                    % Check whether base models expose a linear Beta coefficient
                    K = obj.NumClasses;
                    m1 = obj.K_Models{1};
                    
                    if isprop(m1, 'Beta') && ~isempty(m1.Beta)
                        % Linear base model (PenalizedRegression, LinearModel, etc.)
                        % Beta is (p+1 x 1): [intercept; slope coefficients]
                        p = length(m1.Beta) - 1;
                        B_mat = zeros(p, K);
                        for k = 1:K
                            B_mat(:, k) = obj.K_Models{k}.Beta(2:end);
                        end
                        % Compose: p x K  times  K x (K-1)  =  p x (K-1)
                        S = B_mat * obj.InternalLDA.Scaling;
                        
                    else
                        % Nonlinear base model — no closed-form X-space composition
                        error('FDA:NoXSpaceScaling', ...
                            ['getScaling() requires base models with a linear ''Beta'' property.\n' ...
                             'Nonlinear base models (e.g., MARS) do not have a closed-form\n' ...
                             'X-space scaling matrix. Use pda.project(X, dims) for projections.']);
                    end
                    
                case "OptimalScoring"
                    error('FDA:NoScaling', ...
                        ['getScaling() is not defined for FitMethod=''OptimalScoring''.\n' ...
                         'Canonical directions live in Z_hat space (obj.Theta, K x K-1),\n' ...
                         'not in the original p-dimensional feature space.\n' ...
                         'Use obj.Theta for canonical scores in the transformed space.']);
            end
        end
    end
end
