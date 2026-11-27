classdef Backfitting
    % BACKFITTING Gauss-Seidel algorithm for Generalized Additive Models.
    %
    % THEORY OVERVIEW:
    %   The goal is to fit the additive model: E[Y] = alpha + f1(x1) + ... + fp(xp).
    %   Since we cannot solve for all functions fj simultaneously, we use an 
    %   iterative "Block Coordinate Descent" approach (Gauss-Seidel).
    %
    % ALGORITHM (Hastie & Tibshirani, Alg 9.1):
    %   1. Initialize: Set alpha = mean(Y) and all curves fj = 0.
    %   2. Cycle: For each feature j = 1 to P:
    %      a. ISOLATE: Calculate the "Partial Residual" (Rj). This represents
    %         the part of Y that is NOT explained by the other predictors.
    %         Rj = Y - alpha - Sum(fk for k!=j)
    %      b. FIT: Smooth this residual against Xj to find the best curve fj.
    %         fj = Smooth(Xj, Rj)
    %      c. CENTER: Enforce the identifiability constraint Sum(fj) = 0.
    %         fj = fj - mean(fj)
    %   3. Repeat until the functions stop changing (Convergence).
    
    properties
        MaxIter (1,1) double = 20      % Safety limit to prevent infinite loops
        Tol     (1,1) double = 1e-4    % Convergence threshold (STOP if change < Tol)
        Verbose (1,1) logical = true   % Print progress to console

        % OPTIMIZATION (Ex 9.1): Skip inner-loop centering.
        %
        % Mathematical justification:
        %   Any mean drift (bias) accumulated by smoother j is algebraically
        %   inverted in the partial residual target fed to the next smoother k.
        %   Because linear smoothers preserve constants, smoother k absorbs
        %   that inverted bias.  When the final additive model is summed, the
        %   biases cancel exactly.  Therefore the per-step centering inside
        %   the loop is mathematically redundant.
        %
        % When true:  inner centering is removed; a single O(N) centering
        %             pass runs once after convergence and saves offsets.
        % When false: standard Algorithm 9.1 behaviour (default, safe).
        OptimizeCentering (1,1) logical = false
    end
    
    methods
        function obj = Backfitting(maxIter, tol, optimizeCentering)
            if nargin > 0, obj.MaxIter = maxIter; end
            if nargin > 1, obj.Tol = tol; end
            if nargin > 2, obj.OptimizeCentering = optimizeCentering; end
        end
        
        function model = fit(obj, model, X, Y, Weights)
            % FIT Fits the AdditiveModel using Backfitting.
            % Inputs:
            %   model:   An initialized AdditiveModel object (contains strategies)
            %   X, Y:    Data matrices
            %   Weights: Observation weights (Crucial for IRLS/Logistic Regression)
            
            [N, P] = size(X);
            J = length(model.Strategies);
            
            if nargin < 5 || isempty(Weights)
                Weights = ones(N, 1);
            end
            
            % ============================================================
            % 1. INITIALIZATION
            % ============================================================
            % Theory: The intercept alpha is determined solely by the global mean.
            % In an additive model, we fix alpha = mean(Y) and force all 
            % other functions fj to have a mean of zero.
            mean_Y = sum(Weights .* Y) / sum(Weights);
            model.Intercept = mean_Y;
            
            % Cache Matrix (N x J): 
            % Stores the current prediction of each function f_j(x_ij).
            % We initialize this to ZERO.
            F_matrix = zeros(N, J);
            
            % ============================================================
            % 2. THE GAUSS-SEIDEL LOOP
            % ============================================================
            prev_rss = inf;
            
            for iter = 1:obj.MaxIter
                
                max_change = 0; % Track the biggest shift in any curve this round
                
                % Cycle through each predictor j one by one
                for j = 1:J
                    
                    % -----------------------------------------------------
                    % Step A: Compute Partial Residual (The "Add-Back" Trick)
                    % -----------------------------------------------------
                    % Theory: We want to fit f_j to the signal left over by everyone else.
                    % Formula: R_j = Y - alpha - Sum_{k != j} f_k
                    %
                    % Computational Shortcut: 
                    % Instead of summing (k!=j), we take the Total Sum and 
                    % subtract only the current f_j.
                    % R_j = (Y - alpha - F_Total) + f_j_old
                    
                    F_total = sum(F_matrix, 2);
                    R_j = Y - model.Intercept - F_total + F_matrix(:, j);
                    
                    % -----------------------------------------------------
                    % Step B: Fit the Smoother
                    % -----------------------------------------------------
                    % We ask the specific strategy (e.g., Spline, Loess) to 
                    % find the curve that best explains R_j.
                    
                    strat_j = model.Strategies{j};
                    x_j = X(:, j);
                    
                    fitted_mj = strat_j.fit(x_j, R_j, Weights);
                    f_j_new = strat_j.predict(fitted_mj, x_j);
                    
                    % -----------------------------------------------------
                    % Step C: Centering (Identifiability Constraint)
                    % -----------------------------------------------------
                    % Theory: Additive models are not unique! 
                    % f1(x) + f2(x) is the same as (f1(x)+c) + (f2(x)-c).
                    % To fix this, we enforce E[f_j] = 0.
                    %
                    % OPTIMIZATION: When OptimizeCentering is true this step
                    % is deferred to after convergence (see below).  This is
                    % mathematically safe by the proof in Ex 9.1.
                    if ~obj.OptimizeCentering
                        mean_f = sum(Weights .* f_j_new) / sum(Weights);
                        f_j_new = f_j_new - mean_f;
                    end
                    
                    % -----------------------------------------------------
                    % Step D: Update State & Check Convergence
                    % -----------------------------------------------------
                    % Measure how much the curve changed from the last iteration.
                    change = mean((F_matrix(:, j) - f_j_new).^2);
                    max_change = max(max_change, change);
                    
                    % Gauss-Seidel Update:
                    % Update the cache IMMEDIATELY so the next variable (j+1)
                    % sees the new version of f_j.
                    F_matrix(:, j) = f_j_new;
                    model.Terms{j} = fitted_mj;
                    
                end % End of Feature Cycle
                
                % =========================================================
                % 3. GLOBAL CONVERGENCE CHECK
                % =========================================================
                % Calculate Residual Sum of Squares (RSS) for the whole model
                residual = Y - model.Intercept - sum(F_matrix, 2);
                rss = sum(Weights .* residual.^2);
                
                if obj.Verbose
                    fprintf('  [Backfit] Iter %d: RSS = %.4f, MaxChange = %.6f\n', iter, rss, max_change);
                end
                
                % Stop if the curves stopped moving
                if max_change < obj.Tol
                    if obj.Verbose, fprintf('  Converged.\n'); end
                    break;
                end
            end

            % =============================================================
            % 4. DEFERRED CENTERING PASS (OptimizeCentering only)
            % =============================================================
            % After convergence the functions may carry non-zero means.
            % We apply a single O(N) centering pass here and store the
            % offsets on the model so AdditiveModel.predict can apply them.
            if obj.OptimizeCentering
                J = length(model.Strategies);
                term_offsets = zeros(1, J);
                for j = 1:J
                    mean_f = sum(Weights .* F_matrix(:, j)) / sum(Weights);
                    term_offsets(j) = mean_f;
                end
                model.TermOffsets = term_offsets;
                % Absorb the sum of all offsets into the intercept so the
                % overall prediction level is unchanged.
                model.Intercept = model.Intercept + sum(term_offsets);
            else
                model.TermOffsets = zeros(1, length(model.Strategies));
            end
        end
    end
end