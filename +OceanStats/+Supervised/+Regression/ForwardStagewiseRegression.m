classdef ForwardStagewiseRegression < OceanStats.Base.IModelStrategy
    % FORWARDSTAGEWISEREGRESSION Infinitesimal Forward Stagewise Linear Regression.
    %
    %   OVERVIEW:
    %     Legacy wrapper. This now delegates entirely to the unified
    %     RegularizedLinearModel engine.
    %
    %   See also: OceanStats.Supervised.Regression.RegularizedLinearModel
    
    properties
        Name          string = "Forward Stagewise Regression"
        SupportsAIC   logical = false
        
        NumIterations double = 5000
        LearningRate  double = 0.01
        Standardize   logical = true
    end
    
    properties (Access = private)
        Engine
    end
    
    methods
        function obj = ForwardStagewiseRegression(options)
            arguments
                options.NumIterations double = 5000
                options.LearningRate  double = 0.01
                options.Standardize   logical = true
            end
            obj.NumIterations = options.NumIterations;
            obj.LearningRate  = options.LearningRate;
            obj.Standardize   = options.Standardize;
            
            % 1. Create the Strategy
            solver = OceanStats.Regularization.Solvers.ForwardStagewiseSolver();
            
            % 2. Setup options
            opts.NumIterations = obj.NumIterations;
            opts.LearningRate  = obj.LearningRate;
            
            % 3. Inject into Context
            obj.Engine = OceanStats.Supervised.Regression.RegularizedLinearModel(...
                solver, opts, obj.Standardize);
            obj.Engine.Name = obj.Name;
        end
        
        function model = train(obj, X, Y, W)
            if nargin < 4, W = []; end
            model = obj.Engine.train(X, Y, W);
        end
        
        function Y_pred = predict(obj, model, X)
            Y_pred = obj.Engine.predict(model, X);
        end
        
        function err = evaluate(obj, Y_true, Y_pred)
            err = obj.Engine.evaluate(Y_true, Y_pred);
        end
        
        function k = getComplexity(obj, model)
            k = obj.Engine.getComplexity(model);
        end
    end
end
