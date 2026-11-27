classdef EMTrainer < handle
    % EMTRAINER Orchestrates the HME training process with Warmup and Annealing.
    
    properties
        MaxIter = 100
        Tol = 1e-4
        Initializer % OceanStats.Ensemble.HME.IInitializer worker
    end
    
    methods
        function obj = EMTrainer(options)
            arguments
                options.MaxIter = 100
                options.Tol = 1e-4
                options.Initializer = OceanStats.Ensemble.HME.Initialization.KMeansInitializer()
            end
            obj.MaxIter = options.MaxIter;
            obj.Tol = options.Tol;
            obj.Initializer = options.Initializer;
        end
        
        function [final_ll, params] = train(obj, model, data)
            % Alias for fit that handles data matrix decomposition
            X = data(:, 1:end-1);
            Y = data(:, end);
            [final_ll, params] = obj.fit(model, X, Y);
        end
        
        function [final_ll, params] = fit(obj, model, X, Y)
            % Use the configured worker for structural warmup
            obj.Initializer.initialize(model, X, Y);
            [final_ll, params] = obj.runEM(model, X, Y);
        end
    end
    
    methods (Access = private)
        function [final_ll, params] = runEM(obj, model, X, Y)
            last_ll = -inf;
            for iter = 1:obj.MaxIter
                % E-Step
                L_root = model.computeLikelihoodPass(X, Y, iter);
                total_ll = sum(log(max(L_root, 1e-10)));
                
                if mod(iter, 10) == 0
                    fprintf('    Iteration %d: Log-Lik = %.4f\n', iter, total_ll);
                end
                
                % Check convergence
                if abs(total_ll - last_ll) < obj.Tol, break; end
                last_ll = total_ll;
                
                model.distributeResponsibilityPass(X, Y, L_root);
                
                % M-Step
                model.updateParameters(X, Y, iter);
            end
            final_ll = last_ll;
            params = model.getParams();
        end
    end
end