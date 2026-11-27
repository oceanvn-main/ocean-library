classdef StackedModel < OceanStats.ModelSelection.IModelStrategy
    % STACKEDMODEL Universal Stacking Ensemble.
    %
    %   ARCHITECTURAL DESIGN:
    %   1. Base Layer (Level 0): Diverse experts trained on raw data X.
    %   2. Meta Layer (Level 1): A judge (NNLS/Linear) trained on expert predictions Z.
    %   3. Leakage Prevention: Z is generated via Cross-Validation/Bootstrap to 
    %      ensure the Meta-Learner sees honest out-of-sample errors.
    %   Stacking relies on a single, massive assumption: The relative performance of models stays the same as you add more data.
    properties
        Name string = "Universal Stacked Ensemble"
        SupportsAIC logical = false % Stacking complexity is hard to estimate for AIC
        ParallelThreshold (1,1) double = 1e5 % Heuristic for parfor efficiency
        UseParallel (1,1) logical = true
        
        BaseModels cell       % The Level 0 experts
        MetaLearner           % The Level 1 judge (IModelStrategy)
        CrossValidator        % Data splitting strategy (ICrossValidator)
    end
    
    methods
        function obj = StackedModel(base_models, meta_learner, cv_strategy)
            obj.BaseModels = base_models;
            obj.MetaLearner = meta_learner;
            obj.CrossValidator = cv_strategy;
        end
        
        function model = train(obj, X, Y)
            % TRAIN Executes the two-tier training pipeline.
            [N, ~] = size(X);
            M = length(obj.BaseModels);
            K = obj.CrossValidator.NumFolds;
            
            % Heuristic: Decide if parfor overhead is worth the throughput gain
            work_load = N * M * K;
            is_efficient_to_parallel = obj.UseParallel && ...
                                      (work_load > obj.ParallelThreshold) && ...
                                      ~isempty(gcp('nocreate'));

            % Accumulators for Level 1 Matrix (Z)
            % Z_sum tracks the total predictions, Z_count tracks fold-overlaps
            Z_sum = zeros(N, M);
            Z_count = zeros(N, 1); 
            
            fprintf('Stacking: Generating Level 1 data (K=%d, M=%d)...\n', K, M);
            
            % --- PHASE 1: CROSS-VALIDATION (Generating Honest Predictions) ---
            for k = 1:K
                [train_idx, test_idx] = obj.CrossValidator.getIndices(N, k);
                X_train = X(train_idx, :); Y_train = Y(train_idx, :);
                X_test  = X(test_idx, :);
                Z_count(test_idx) = Z_count(test_idx) + 1;
                
                base_preds = cell(1, M);
                if is_efficient_to_parallel
                    parfor m = 1:M
                        % Train temporary "fold-expert"
                        tmp_fit = obj.BaseModels{m}.train(X_train, Y_train);
                        % Predict on unseen fold data
                        base_preds{m} = obj.BaseModels{m}.predict(tmp_fit, X_test);
                    end
                else
                    for m = 1:M
                        tmp_fit = obj.BaseModels{m}.train(X_train, Y_train);
                        base_preds{m} = obj.BaseModels{m}.predict(tmp_fit, X_test);
                    end
                end
                
                % Aggregate into sum (Safe serial operation)
                for m = 1:M
                    Z_sum(test_idx, m) = Z_sum(test_idx, m) + base_preds{m};
                end
            end
            
            % --- PHASE 2: HOLE IMPUTATION ---
            % Fixes data points that were never in a Test Fold (Bootstrap artifacts)
            holes = (Z_count == 0);
            if any(holes)
                warning('OceanStats:Coverage', 'Holes in Z detected. Imputing via full-model fallback.');
                for m = 1:M
                    % Slightly biased, but prevents the Meta-Learner from seeing 0s
                    full_mdl = obj.BaseModels{m}.train(X, Y);
                    Z_sum(holes, m) = obj.BaseModels{m}.predict(full_mdl, X(holes, :));
                end
                Z_count(holes) = 1;
            end
            
            % Final Level 1 Matrix Z (The Average OOB Predictions)
            Z = Z_sum ./ repmat(Z_count, 1, M);
            
            % --- PHASE 3: META-LEARNING ---
            % Judge learns how to weight the Experts
            meta_fit = obj.MetaLearner.train(Z, Y); %Get Meta Learner(Learning the relationship between the model errors and the truth). NOTICE. THE BRAIN ALREADY DECIDED HERE.
            % Using Z as feature to predict Y
            
            % --- PHASE 4: FINAL RETRAINING ---
            % Now that we have the meta-weights, we train the base models on 100% 
            % of data to maximize their intelligence for production deployment.
            final_base_fits = cell(1, M);
            for m = 1:M
                final_base_fits{m} = obj.BaseModels{m}.train(X, Y);
            end
            
            model.MetaFit = meta_fit;
            model.BaseFits = final_base_fits;
            model.Z_Used = Z; % Store for diagnostics
        end
        
        function preds = predict(obj, model, X)
            % PREDICT Hierarchical Prediction Flow:
            % 1. Experts (BaseModels) predict X -> Z_new
            % 2. Judge (MetaLearner) predicts Z_new -> Final Output
            N = size(X, 1);
            M = length(obj.BaseModels);
            Z_new = zeros(N, M);
            for m = 1:M
                Z_new(:, m) = obj.BaseModels{m}.predict(model.BaseFits{m}, X);
            end
            preds = obj.MetaLearner.predict(model.MetaFit, Z_new);
        end

        function importance = getModelImportance(obj, model)
            % GETMODELIMPORTANCE Analyzes the Meta-Learner's preference.
            % 
            % If Meta-Learner is Linear/NNLS: Returns normalized coefficients.
            % Higher values = High trust in that specific base model.
            
            if isfield(model.MetaFit, 'Coeffs')
                w = model.MetaFit.Coeffs;
                % Normalize to percentage of total trust
                importance = (w / sum(w)) * 100;
            else
                % Fallback for non-linear meta-learners (e.g. Trees)
                importance = ones(length(obj.BaseModels), 1) * NaN;
                warning('Meta-Learner does not support coefficient extraction.');
            end
        end
        
        function err = evaluate(~, Y_true, Y_pred)
            err = mean((Y_true - Y_pred).^2);
        end
        
        function k = getComplexity(~, ~)
            k = NaN; % Not supported
        end
    end
end