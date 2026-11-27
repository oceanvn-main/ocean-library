classdef LocalScoring
    % LOCALSCORING Iteratively Reweighted Least Squares (IRLS) for GAMs.
    %
    % THEORY OVERVIEW:
    %   Generalized Linear Models (GLMs) connect the outcome Y to the predictors
    %   via a "Link Function" g(mu) = eta.
    %     - For Gaussian: eta = mu (Identity Link).
    %     - For Binomial: eta = log(p / 1-p) (Logit Link).
    %
    %   Since the Logit link is non-linear, we cannot solve for 'eta' directly.
    %   We use Newton-Raphson (Iteratively Reweighted Least Squares) to 
    %   linearize the problem locally.
    %
    % ALGORITHM (ESL 9.2):
    %   1. Initialize the linear predictor 'eta'.
    %   2. OUTER LOOP (Newton-Raphson):
    %      a. Calculate Probability 'p' from current 'eta'.
    %      b. Calculate 'Working Response' (z): A synthetic target that points 
    %         where the model should go next.
    %      c. Calculate Weights (w): How much we trust each data point 
    %         (Inverse Variance).
    %      d. INNER LOOP (Backfitting): Fit a weighted additive model to 'z'.
    %      e. Update 'eta' with the new additive sum.
    %
    % FAMILIES SUPPORTED:
    %   - 'Gaussian': Standard Regression (One pass).
    %   - 'Binomial': Logistic Regression (Iterative).
    
    properties
        % The distribution of the target variable.
        Family string {mustBeMember(Family, ["Gaussian", "Binomial"])} = "Gaussian"
        
        % Newton-Raphson Settings
        MaxIter (1,1) double = 10      % Max outer loops
        Tol     (1,1) double = 1e-4    % Convergence threshold
        
        % The engine that solves the weighted regression problem.
        % This is Dependency Injection: LocalScoring delegates the "fitting" 
        % work to this object.
        Backfitter OceanStats.GAM.Backfitting
        
        Verbose logical = true
    end
    
    methods
        function obj = LocalScoring(family, backfitter)
            % CONSTRUCTOR
            % Sets up the algorithm mode and the backfitting engine.
            
            obj.Family = family;
            
            % Default Fallback:
            % If the user doesn't provide a specific backfitter, create a 
            % standard one automatically.
            if nargin < 2
                obj.Backfitter = OceanStats.GAM.Backfitting();
            else
                obj.Backfitter = backfitter;
            end
        end
        
        function model = fit(obj, model, X, Y)
            % FIT Fits the GAM using the Local Scoring Algorithm.
            % Inputs:
            %   model: An empty AdditiveModel structure.
            %   X, Y:  Data.
            
            [N, ~] = size(X);
            
            % Ensure Numeric
            if ~isa(Y, 'double'), Y = double(Y); end
            if ~isa(X, 'double'), X = double(X); end
            
            % ============================================================
            % 1. INITIALIZATION (Getting a safe starting point)
            % ============================================================
            if strcmp(obj.Family, "Binomial")
                % LOGISTIC CASE:
                % We need a starting 'eta' (log-odds). 
                % We cannot use log(Y/(1-Y)) directly because Y is 0 or 1,
                % which leads to log(0) = -Inf or log(Inf) = +Inf.
                
                % Solution: "Soft Starts". 
                % Treat Y=1 as 0.75 and Y=0 as 0.25 to get a finite start.
                mu = (Y + 0.5) / 2; 
                eta = log(mu ./ (1 - mu));
                
                % Set the model intercept to the average log-odds.
                model.Intercept = mean(eta);
                
            else
                % GAUSSIAN CASE:
                % The link is Identity. The best starting guess is simply 
                % the global average of Y.
                eta = mean(Y);
                model.Intercept = eta;
            end
            
            % ============================================================
            % 2. THE NEWTON-RAPHSON OUTER LOOP
            % ============================================================
            for iter = 1:obj.MaxIter
                
                if strcmp(obj.Family, "Binomial")
                    % -----------------------------------------------------
                    % Step A: Compute Probabilities (p)
                    % -----------------------------------------------------
                    % Convert current log-odds 'eta' back to probability.
                    p = 1 ./ (1 + exp(-eta));
                    
                    % -----------------------------------------------------
                    % Step B: Compute Weights (w)
                    % -----------------------------------------------------
                    % Theory: In Newton-Raphson, weight = 2nd Derivative (Hessian).
                    % For Logistic Regression, Hessian = p(1-p).
                    % Intuitively: We trust uncertain points (p=0.5) the most
                    % because they define the decision boundary.
                    w = p .* (1 - p);
                    w = max(w, 1e-5); % Numerical stability (avoid division by zero)
                    
                    % -----------------------------------------------------
                    % Step C: Compute Working Response (z)
                    % -----------------------------------------------------
                    % Theory: This is the Linearized Target.
                    % z = Current_Pos + (Error / Curvature)
                    % z = eta + (Y - p) / (p(1-p))
                    % This tells the inner loop: "If the world were linear,
                    % move the prediction to 'z' to fix the error."
                    z = eta + (Y - p) ./ w;
                    
                else
                    % GAUSSIAN CASE:
                    % The world is already linear.
                    % Target is just the raw data Y. Weights are equal.
                    z = Y;
                    w = ones(N, 1);
                end
                
                % ---------------------------------------------------------
                % Step D: The Inner Loop (Delegation)
                % ---------------------------------------------------------
                % We hand the simplified problem (fit 'z' using weights 'w')
                % to the Backfitting engine.
                if obj.Verbose
                      fprintf('--- [Local Scoring] Outer Iter %d ---\n', iter);
                end
                
                model = obj.Backfitter.fit(model, X, z, w);
                
                % ---------------------------------------------------------
                % Step E: Synchronization
                % ---------------------------------------------------------
                % The Backfitter has updated the curves f_j.
                % We must now recalculate 'eta' (the sum of curves) so that
                % the next iteration uses the correct probabilities.
                eta = model.predict(X);
                
                % ---------------------------------------------------------
                % Step F: Convergence Check
                % ---------------------------------------------------------
                % For Gaussian models, the solution is exact in one step.
                % For Binomial, we loop until 'eta' stabilizes (omitted for brevity).
                if strcmp(obj.Family, "Gaussian")
                    break;
                end
            end
        end
    end
end