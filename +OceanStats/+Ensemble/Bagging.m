classdef Bagging < OceanStats.Ensemble.WeightedEnsemble
    % BAGGING Bootstrap Aggregation.
    %
    % OVERVIEW:
    %   Reduces variance by averaging models trained on bootstrap samples.
    %   - Training: Sample N items with replacement B times. Train B models.
    %   - Prediction: Uniform average of B independent predictions.
    
    properties
        PrototypeModel % The template model to clone
        NumEstimators double = 50
    end
    
    methods
        function obj = Bagging(prototype, n_estimators)
            % Constructor: Pass emtpy list to superclass init
            obj = obj@OceanStats.Ensemble.WeightedEnsemble({});
            obj.PrototypeModel = prototype;
            obj.NumEstimators = n_estimators;
            obj.Name = "Bagging (" + prototype.Name + ")";
        end
        
        function model = train(obj, X, Y)
            % TRAIN Bagging via BootstrapSampler.
            % We use the reusable generator to handle resampling and parallelization.
            import OceanStats.Data.BootstrapSampler
            
            % Concatenate X and Y for sampling (preserving row alignment)
            data_combined = [X, Y];
            N_features = size(X, 2);
            
            sampler = BootstrapSampler(data_combined);
            
            % Helper function to unpack data and train
            % Input 'd' is [X_boot, Y_boot]
            train_on_sample = @(d) obj.PrototypeModel.train(d(:, 1:N_features), d(:, N_features+1:end));
            
            % Execute Parallel Bootstrap Training
            base_fits = sampler.drawNonParametric(train_on_sample, ...
                'Replicates', obj.NumEstimators, ...
                'OutputFormat', 'cell');
            
            % Convert Column vector to Row vector (if needed) to match WeightedEnsemble expects
            base_fits = base_fits'; 
            
            obj.BaseModels = cell(1, obj.NumEstimators);
            for i = 1:obj.NumEstimators
                obj.BaseModels{i} = obj.PrototypeModel;
            end
            
            model.BaseFits = base_fits;
            
            % Uniform Weights
            B = obj.NumEstimators;
            obj.Weights = ones(1, B) / B;
            model.Weights = obj.Weights;
        end
        
        function weights = computeWeights(obj, ~, ~, ~)
            % COMPUTEWEIGHTS Uniform (1/B).
            B = length(obj.BaseModels);
            weights = ones(1, B) / B;
        end
    end
end
