classdef (Abstract) WeightedEnsemble < OceanStats.ModelSelection.IModelStrategy
    % WEIGHTEDENSEMBLE Base class for ensembles that average predictions.
    %
    % OVERVIEW:
    %   Abstracts the logic of fitting multiple base models and combining 
    %   their predictions via a weighted sum: y_hat = Sum(w_i * y_hat_i).
    %
    % SUBCLASSES MUST IMPLEMENT:
    %   computeWeights(obj, metrics) -> [1 x M] weight vector.
    
    properties
        BaseModels cell % Cell array of IModelStrategy
        Weights double  % [1 x M] vector of model weights
        Name string
        SupportsAIC logical = false
    end
    
    methods
        function obj = WeightedEnsemble(base_models)
            obj.BaseModels = base_models;
        end
        
        function model = train(obj, X, Y)
            % TRAIN Fits base models and calculates weights.
            
            M = length(obj.BaseModels);
            base_fits = cell(1, M);
            metrics = zeros(1, M);
            
            % 1. Fit Base Models
            for i = 1:M
                bm = obj.BaseModels{i};
                fit_artifact = bm.train(X, Y);
                base_fits{i} = fit_artifact;
                
                % 2. Collect Metrics for Weighting
                % We assume the subclass needs some metric (RSS, LogLik, etc.)
                % Default: RSS
                y_pred = bm.predict(fit_artifact, X);
                metrics(i) = sum((Y - y_pred).^2); 
            end
            
            % 3. Compute Weights (Delegated to Subclass)
            % Subclasses might need more info (e.g., complexity k, N), 
            % so we might need to pass more context or store it.
            % For now, pass metrics and N.
            N = size(X, 1);
            obj.Weights = obj.computeWeights(metrics, N, base_fits);
            
            model.BaseFits = base_fits;
            model.Weights = obj.Weights;
            model.Metrics = metrics;
        end
        
        function preds = predict(obj, model, X)
            % PREDICT Weighted average.
            M = length(obj.BaseModels);
            N = size(X, 1);
            preds = zeros(N, 1);
            
            for i = 1:M
                w = model.Weights(i);
                if w < 1e-6, continue; end
                
                bm = obj.BaseModels{i};
                y_hat = bm.predict(model.BaseFits{i}, X);
                preds = preds + w * y_hat;
            end
        end
        
        function err = evaluate(obj, Y_true, Y_pred)
            err = mean((Y_true - Y_pred).^2);
        end
        
        function k = getComplexity(obj, ~)
            k = length(obj.BaseModels); % Proxy
        end
    end
    
    methods (Abstract)
        weights = computeWeights(obj, metrics, N, base_fits)
    end
end
