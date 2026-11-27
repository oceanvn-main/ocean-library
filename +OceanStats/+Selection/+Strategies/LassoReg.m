classdef LassoReg < OceanStats.Base.IModelStrategy
    % LASSOREG Lasso Regression Strategy (L1 Penalty).
    %
    %   OVERVIEW:
    %     Legacy wrapper for single-Lambda Lasso. This now delegates 
    %     entirely to the unified RegularizedLinearModel engine.
    %   This is the What. Definition of Problem
    %
    %   USAGE:
    %     strat = OceanStats.Selection.Strategies.LassoReg(0.01);
    
    properties
        Name string = "Lasso Regression"
        SupportsAIC logical = true 
        Lambda (1, 1) double = 0.01
    end
    
    properties (Access = private)
        Engine
    end
    
    methods
        function obj = LassoReg(lambda, solver)
            arguments
                lambda double = 0.01
                solver OceanStats.Regularization.Solvers.IRegularizationSolver = OceanStats.Regularization.Solvers.CoordinateDescentSolver() %Default
            end
            
            obj.Lambda = lambda;
            
            % 2. Setup options (Alpha = 1.0 means pure Lasso)
            opts.Alpha = 1.0;
            opts.Lambdas = obj.Lambda;
            
            % 3. Inject into Context
            obj.Engine = OceanStats.Supervised.Regression.RegularizedLinearModel(...
                solver, opts, true);
        end
        
        function model = train(obj, X, Y, W)
            % If X is empty (Null Model)
            if size(X, 2) == 0
                model = struct('Beta', [], 'Intercept', mean(Y), 'df', 1);
                return;
            end
            
            if nargin < 4, W = []; end
            
            % The unified engine returns model.BetaFinal, model.MuY, etc.
            % We repackage it into the struct expected by older Selection code.
            fit = obj.Engine.train(X, Y, W);
            
            model.Beta = fit.BetaFinal;
            model.Intercept = fit.MuY - (fit.MuX ./ fit.SigmaX) * fit.BetaFinal; % Map to unstandardized
            model.df = sum(fit.BetaFinal ~= 0) + 1; % Non-zero coeff + intercept
            model.Lambda = obj.Lambda;
            
            % Keep the underlying fit for internal use if needed
            model.EngineFit = fit;
        end
        
        function y_pred = predict(obj, model, X)
             if isempty(model.Beta)
                y_pred = repmat(model.Intercept, size(X, 1), 1);
            else
                % Use the Engine's prediction logic for consistency
                y_pred = obj.Engine.predict(model.EngineFit, X);
            end
        end
        
        function err = evaluate(obj, Y_true, Y_pred)
            err = obj.Engine.evaluate(Y_true, Y_pred);
        end
        
        function k = getComplexity(obj, model)
            k = model.df;
        end
    end
end
