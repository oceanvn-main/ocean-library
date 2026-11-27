classdef RegularizedLinearModel < OceanStats.Base.IModelStrategy
    % REGULARIZEDLINEARMODEL Unified engine for regularized linear models.
    %
    %   OVERVIEW:
    %     This is a Context class that unifies Ridge, Lasso, Elastic Net,
    %     and Forward Stagewise regression using the Strategy Pattern.
    %     It handles all data standardization and routes execution to the
    %     injected IRegularizationSolver, completely removing the need for 
    %     if/else string checking.
    %
    %   USAGE:
    %     solver = OceanStats.Regularization.Solvers.CoordinateDescentSolver();
    %     opts.Alpha = 1.0;
    %     model = OceanStats.Supervised.Regression.RegularizedLinearModel(...
    %                solver, opts);
    %     fit = model.train(X, Y);
    
    properties
        Name          string = "Regularized Linear Model"
        SupportsAIC   logical = true
        
        Solver        OceanStats.Regularization.Solvers.IRegularizationSolver = OceanStats.Regularization.Solvers.CoordinateDescentSolver()
        SolverOptions struct  = struct()
        Standardize   logical = true
    end
    
    methods
        function obj = RegularizedLinearModel(solver, options, standardize)
            arguments
                solver        OceanStats.Regularization.Solvers.IRegularizationSolver = OceanStats.Regularization.Solvers.CoordinateDescentSolver()
                options       struct  = struct('Alpha', 1.0, 'Lambdas', 'auto') %Lasso by default
                standardize   logical = true
            end
            obj.Solver        = solver;
            obj.SolverOptions = options;
            obj.Standardize   = standardize;
        end
        
        function model = train(obj, X, Y, ~)
            [N, p] = size(X);
            
            % 1. Standardization (Crucial for correlation checks and penalties)
            if obj.Standardize
                mu_x = mean(X, 1);
                sigma_x = std(X, 0, 1); %Std for every feature, 0 for using N-1 Bessel's correction, 1 for across columns
                sigma_x(sigma_x == 0) = 1; % Prevent div by zero
                X_proc = (X - mu_x) ./ sigma_x;
                
                mu_y = mean(Y); %Mean-center Y at 0.
                Y_proc = Y - mu_y;
            else
                X_proc = X;
                Y_proc = Y;
                mu_x = zeros(1, p);
                sigma_x = ones(1, p);
                mu_y = 0;
            end
            
            % 2. Execute Injected Strategy (No if/else needed!)
            [beta_path, extra_info] = obj.Solver.solve(X_proc, Y_proc, obj.SolverOptions);
            
            % 3. Select Final Coefficients (least regularized at end of path)
            beta_final = beta_path(:, end);
            
            % 4. Package standard model struct
            model.BetaPath = beta_path;
            model.BetaFinal = beta_final;
            model.ExtraInfo = extra_info;
            
            % Store standardization params for prediction
            model.MuX = mu_x;
            model.SigmaX = sigma_x;
            model.MuY = mu_y;
        end
        
        function Y_pred = predict(obj, model, X)
            X_proc = (X - model.MuX) ./ model.SigmaX; %Do the same to X_test
            Y_pred = X_proc * model.BetaFinal + model.MuY; % Add mean Y back
        end
        
        function err = evaluate(~, Y_true, Y_pred)
            err = mean((Y_true - Y_pred).^2);
        end
        
        function k = getComplexity(~, model)
            % GETCOMPLEXITY Returns the Effective Degrees of Freedom (DF).
            % For L1 regularized models, the unbiased estimate of DF is 
            % the number of non-zero coefficients (Zou et al. 2007).
            % For L2 (Ridge), it is the trace of the hat matrix.
            
            if isfield(model.ExtraInfo, 'DF') && ~isempty(model.ExtraInfo.DF)
                % If solver natively computed DF for the path (e.g. CoordinateDescent)
                df_path = model.ExtraInfo.DF;
                if isvector(df_path) && length(df_path) > 1
                    k = df_path(end); % DF of the final selected model
                else
                    k = df_path;
                end
            else
                % Fallback: Count non-zero coefficients (Standard L1/Stagewise assumption)
                % We use a small epsilon to avoid floating point precision issues
                k = sum(abs(model.BetaFinal) > 1e-9);
            end
            
            % Add 1 to account for the Intercept (MuY)
            k = k + 1;
        end
    end
end
