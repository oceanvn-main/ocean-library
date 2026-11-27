classdef BMA < OceanStats.Ensemble.WeightedEnsemble
    % BMA Bayesian Model Averaging via WeightedEnsemble.
    %
    % Implements computeWeights using BIC/AIC.
    
    properties
        Name string = "Bayesian Model Averaging"
        SupportsAIC logical = true
        
        % WeightingFunction: f(rss, k, 'N', N, 'Type', 'RSS') -> score
        WeightingFunction function_handle = @OceanStats.ModelSelection.bic 
    end
    
    methods
        function obj = BMA(base_models, weight_fcn)
            % BMA Constructor
            obj@OceanStats.Ensemble.WeightedEnsemble(base_models);
            
            if nargin > 1
                obj.WeightingFunction = weight_fcn;
            end
        end
        
        function weights = computeWeights(obj, metrics, N, base_fits)
            % COMPUTEWEIGHTS Calculates posterior weights via BIC/AIC.
            % metrics: Vector of RSS values (from WeightedEnsemble.train).
            
            M = length(obj.BaseModels);
            scores = zeros(1, M);
            
            for i = 1:M
                rss = metrics(i);
                k = obj.BaseModels{i}.getComplexity(base_fits{i});
                
                % Calculate Score
                scores(i) = obj.WeightingFunction(rss, k, 'N', N, 'Type', 'RSS');
            end
            
            % Stable Softmax
            min_score = min(scores);
            delta = scores - min_score;
            raw = exp(-0.5 * delta);
            weights = raw / sum(raw);
        end
    end
end
