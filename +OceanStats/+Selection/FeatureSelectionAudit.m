classdef FeatureSelectionAudit < handle
    % FEATURESELECTIONAUDIT (Moves to +OceanStats/+ModelSelection/+Audits)
    % FEATURESELECTIONAUDIT Universal Engine for Feature Selection & Validation.
    %
    %   OVERVIEW:
    %   A rigorous, model-agnostic engine that finds the optimal feature subset
    %   for ANY machine learning algorithm (Linear, Tree, etc.).
    %
    %   ARCHITECTURAL DESIGN:
    %   1. Strategy Pattern: Delegates all math (Training/Prediction) to an 
    %      injected 'IModelStrategy'.
    %   2. Nested Validation: Performs feature selection INSIDE Cross-Validation
    %      and Bootstrap loops to prevent data leakage.
    %   3. Pareto Optimization: Identifies the best model for every subset size k.
    %
    %   USAGE:
    %       1. Define Strategy
    %       strat = OceanStats.Selection.Strategies.LinearReg();
    %    
    %       2. Initialize Audit
    %       audit = OceanStats.Selection.FeatureSelectionAudit(X, Y, strat);
    %
    %       % 3. Run & Plot
    %       audit.run();
    %       audit.plot(viz);
    
    % --- 1. Configuration ---
    properties 
        K_Folds (1, :) double {mustBeInteger} = [5, 10] %CV configuration
        B_Bootstrap (1, 1) double {mustBeInteger} = 50 %Bootstrap samples
        UseParallel (1, 1) logical = true %Enable 'parfor'
    end
    
    % --- 2. State & Result ---
    properties (SetAccess = private)
        Dataset struct          % Immutable Data (X, Y, N, P)
        Strategy
        Results table           % Final Report (Error Metrics)
        BestModels struct       % The Pareto Frontier (Best model per size k)
        IsAnalyzed logical = false
    end
    
    % --- Public Methods ---
    methods
        function obj = FeatureSelectionAudit(X, Y, strategy, options)
            % CONSTRUCTOR
            % Inputs:
            %   X, Y     : Data
            %   strategy : An instance of IModelStrategy (e.g., LinearReg)
            %   names    : (Optional) Feature Names
            
            arguments
                X (:, :) double {mustBeReal, mustBeFinite, mustBeNonempty}
                Y (:, 1) double {mustBeReal, mustBeFinite, mustBeNonempty}
                strategy (1, 1) OceanStats.Base.IModelStrategy
                options.Names (1, :) string = []                
            end
            %Integriry check
            if size(X, 1) ~= size(Y, 1)
                error('OceanStats:DimMismatch', 'Rows in X (%d) != Rows in Y (%d).', size(X,1), size(Y,1));
            end
            [N, P] = size(X);
            %Default Names
            if isempty(options.Names), options.Names = "Feat" + (1:P); end
            
            %Store State
            obj.Dataset = struct('X', X, 'Y', Y, 'Names', options.Names, 'N', N, 'P', P);
            obj.Strategy = strategy;      
        end
        
        function result = run(obj)
            %RUN Execute the universal analysis pipeline.
            P = obj.Dataset.P;
            fprintf('Audit [%s]: Analyzing %d features...\n', obj.Strategy.Name, obj.Dataset.P)
            
            
            % --- Phase 1: Global Search ---
            %Find best subsets using the Full Dataset.
            obj.BestModels = obj.exhaustiveSearch(obj.Dataset.X, obj.Dataset.Y);
            
            % --- Phase 2: Analytical Metrics (Conditional) ---
            % Only calculate AIC/BIC if the strategy supports it (e.g. Linear).
            % Trees or KNN usually do not support standard AIC.
            obj.Results = obj.compileBasicResults();
            if obj.Strategy.SupportsAIC
                obj.Results = obj.appendInformationCriteria(obj.Results);     
            end
            
            % --- Phase 3: Nested Cross-Validation ---
            for k = obj.K_Folds
                fprintf(' > Running %d-Fold Nested CV...\n', k);
                % MODIFIED: Now retrieves Standard Error (se) as well
                [err, se] = obj.runNestedCV(k);
                
                % Store Mean Error
                obj.Results.(sprintf('CV_%d', k)) = err;
                % Store Standard Error (Confidence Interval bound)
                obj.Results.(sprintf('CV_%d_SE', k)) = se; 
            end
            
            % --- Phase 4: Nested Bootstrap .632+ ---
            if obj.B_Bootstrap > 0
                fprintf(' > Running .632+ Bootstrap (%d reps)...\n', obj.B_Bootstrap);
                err = obj.runNestedBootstrap(obj.B_Bootstrap);
                obj.Results.Boot632 = err;
            end
            
            obj.IsAnalyzed = true;
            fprintf(' > Analysis complete.\n');
            %Return Structure for Visualization
            result = struct(...
                'Table', obj.Results, ...
                'StrategyName', obj.Strategy.Name, ...
                'N', obj.Dataset.N, ...
                'P', P);
        end
        
    end
    
    % --- PRIVATE ENGINE (GENERIC LOGIC) ---
    methods (Access = private)
        function best_models = exhaustiveSearch(obj, X, Y)
            % EXHAUSTIVE SEARCH (Generic)
            % Iterates all combinations, delegates Training/Scoring to Strategy.
            
            [N, P] = size(X);
            
            % 1. Null Model (k=0)
            % We pass empty X to strategy? No, most need at least Intercept.
            % We handle k=0 as a special case: Train on column of ones?
            % Easier: Let Strategy handle empty X if possible, or manual Intercept.
            % For simplicity here, we assume Strategy.train([], Y) handles Null.
            
            % Pre-allocate
            best_models(P+1) = struct('Size', [], 'Error', [], 'Mask', [], 'Model', []);
            
            % Handle Null Model (k=0)
            best_models(1).Size = 0;
            best_models(1).Mask = false(1, P);
            best_models(1).Model = obj.Strategy.train(zeros(N,0), Y);
            % Use empty array for predict if X is empty/null, strategy must handle
            y_null = obj.Strategy.predict(best_models(1).Model, zeros(N,0));
            best_models(1).Error = obj.Strategy.evaluate(Y, y_null);
            % 2. Combinatorial Loop
            %Progress bar setup
            if obj.UseParallel && P > 5
                D = parallel.pool.DataQueue;
                %Update Progress every 10 completions to reduce overhead
                afterEach(D, @(~) fprintf('.'));
                fprintf(' Progress: ');
            end
            %Strategy Broadcast
            strat = obj.Strategy;
            for k = 1:P
                combs = nchoosek(1:P, k);
                num_cands = size(combs, 1);
                err_cands = inf(num_cands, 1);
                
                % We can't store objects in parfor easily. 
                % We just need the Index of the winner.
                
                % Setup for Parallel   
                if obj.UseParallel && num_cands > 500
                    parfor i = 1:num_cands
                        
                        cols = combs(i, :);
                        X_sub = X(:, cols);
                        
                        % --- POLYMORPHIC CALLS ---
                        mdl = strat.train(X_sub, Y);       % Train
                        y_hat = strat.predict(mdl, X_sub); % Predict
                        err_cands(i) = strat.evaluate(Y, y_hat);    % Score
                        %Send Progress
                        if mod(i, 500) == 0, send(D, 1); end
                        
                    end
                else
                    for i = 1:num_cands
                        cols = combs(i, :);
                        X_sub = X(:, cols);
                        mdl = strat.train(X_sub, Y);
                        y_hat = strat.predict(mdl, X_sub);
                        err_cands(i) = strat.evaluate(Y, y_hat);
                    end
                end
                
                % Find Winner for size k
                [min_err, idx] = min(err_cands);
                mask = false(1, P);
                mask(combs(idx, :)) = true;
                
                % Re-train the winner to store the model object
                % (We didn't store models in loop to save RAM)
                best_models(k+1).Size = k;
                best_models(k+1).Error = min_err;
                best_models(k+1).Mask = mask;
                best_models(k+1).Model = obj.Strategy.train(X(:, mask), Y);
            end
            
            % Handle Null Model (k=0)
            best_models(1).Size = 0;
            best_models(1).Mask = false(1, P);
            % Train on empty matrix (Strategy must handle this, e.g., fit Intercept only)
            best_models(1).Model = obj.Strategy.train(zeros(N,0), Y); 
            y_null = obj.Strategy.predict(best_models(1).Model, zeros(N,0));
            best_models(1).Error = obj.Strategy.evaluate(Y, y_null);
        end
    
        function [cv_mean, cv_se] = runNestedCV(obj, K)
            % RUNNESTEDCV Performs K-Fold Cross-Validation with Internal Feature Selection.
            %
            %   MODIFIED FOR CONFIDENCE INTERVALS:
            %   Returns both the Mean CV Error and the Standard Error (SE) across folds.
            %   SE = StdDev(FoldErrors) / sqrt(K).
            %
            %   OVERVIEW:
            %   Estimates the generalization error for every subset size k (0 to P)
            %   by simulating the entire selection process within each fold.
            %
            %   CRITICAL LOGIC ("The Nested Loop"):
            %   Unlike standard CV, we do not test a fixed set of features.
            %   Instead, we re-run 'exhaustiveSearch' inside every fold.
            %   This ensures that the test data is truly "unseen," preventing
            %   selection bias (data leakage).
            %
            %   INPUTS:
            %       obj : FeatureSelectionAudit instance.
            %       K   : Number of folds (e.g., 10).
            %
            %   OUTPUTS:
            %       cv_mean : Vector [P+1 x 1] Mean Error across folds.
            %       cv_se   : Vector [P+1 x 1] Standard Error of the mean.
        
            % 1. Partition Data
            % Create random fold indices (1 to K) for all N samples.
            % 'indices' is a vector of length N.
            indices = crossvalind('Kfold', obj.Dataset.N, K);
            
            P = obj.Dataset.P;
            
            % Accumulator for Error Per Fold.
            % Matrix [ModelSize x FoldIndex]. Used to calculate variance later.
            fold_errors = zeros(P+1, K);
            
            % 2. The Cross-Validation Loop
            for i = 1:K
                % A. Define Split
                % 'test_mask' isolates the ~10% hold-out data.
                test_mask = (indices == i);
                train_mask = ~test_mask;
                
                % B. RE-RUN SEARCH (The "Training" Phase)
                % CRITICAL: We find the best models using ONLY the training data
                % for this fold. The test data is completely hidden from this search.
                % This returns a struct array of the best models for k=0..P specific to this fold.
                models_fold = obj.exhaustiveSearch(obj.Dataset.X(train_mask,:), obj.Dataset.Y(train_mask));
                
                % C. EVALUATE (The "Testing" Phase)
                % Prepare the hold-out data for evaluation.
                X_test = obj.Dataset.X(test_mask, :);
                Y_test = obj.Dataset.Y(test_mask);
                
                % Loop through every possible subset size k (0 to P).
                % COMPLEXITY LOOP
                for k_idx = 1:(P+1)
                    % Retrieve the winner for size k found in *this* fold
                    m = models_fold(k_idx);
                    
                    % D. Prepare Predictors
                    % Extract only the columns selected by the mask.
                    if m.Size == 0
                        % Special Case: Null Model (Size 0). Strategy handles empty input.
                        X_sub = zeros(sum(test_mask), 0);
                    else
                        % Standard Case: Subset the test matrix to the selected features.
                        X_sub = X_test(:, m.Mask);
                    end
                    
                    % E. Predict & Score
                    % Ask the Strategy to predict using the model trained in this fold.
                    y_pred = obj.Strategy.predict(m.Model, X_sub);
                    
                    % Calculate Total Error (RSS) for this fold
                    fold_rss = obj.Strategy.evaluate(Y_test, y_pred);
                    
                    % Normalize to MSE (Mean Squared Error) for this specific fold
                    % This puts errors on a comparable scale regardless of fold size variations
                    fold_errors(k_idx, i) = fold_rss / length(Y_test);
                end
            end
                
            % 3. Finalize
            % Compute Mean Error across all K folds
            cv_mean = mean(fold_errors, 2);
            
            % Compute Standard Error (Confidence Interval Bound)
            % SE = StandardDeviation / sqrt(N_Samples)
            % Here, our "Samples" are the folds (K).
            cv_se = std(fold_errors, 0, 2) / sqrt(K);
        end
        function err632 = runNestedBootstrap(obj, B)
            % RUNNESTEDBOOTSTRAP Computes the .632 Estimator via Nested Resampling.
            %
            %   OVERVIEW:
            %   This method estimates the generalization error using the ".632 Rule",
            %   which is designed to handle the "Pessimism" of standard Cross-Validation
            %   and the "Optimism" of Apparent (Training) Error.
            %
            %   THEORY (.632 Estimator):
            %   Standard Bootstrap (Leave-One-Out Bootstrap, Err(1)) is biased upwards 
            %   because each bootstrap sample only contains ~63.2% of unique data points,
            %   making the models weaker than a model trained on the full N.
            %   The .632 Estimator corrects this by averaging the Training Error (biased down)
            %   and the OOB Error (biased up).
            %
            %   Formula: Err(.632) = 0.368 * Err_train + 0.632 * Err_1
            %   (Where 0.368 = 1/e, the probability of being OOB).
            %
            %   NESTED LOGIC:
            %   Just like CV, we must re-run 'exhaustiveSearch' inside every bootstrap
            %   sample. If we didn't, we would be testing features selected on the
            %   OOB data, leading to severe data leakage.
            %
            %   INPUTS:
            %       obj : FeatureSelectionAudit instance.
            %       B   : Number of bootstrap replications (e.g., 50 or 100).
            %
            %   OUTPUTS:
            %       err632 : Vector [P+1 x 1] of the estimated error for each model size k.
            
            N = obj.Dataset.N;
            P = obj.Dataset.P;
            
            
            %Step 1: Calculate Apparent Error (Err_train)
            % This is the error of the best models found on the FULL dataset.
            % These models are already stored in 'obj.BestModels' from the Global Search phase.
            % We normalize by N to get the Mean Error (e.g., MSE).
            apparent_err = [obj.BestModels.Error]' / N;
            
            %Step 2: Setup OOB Accumulators
            % To compute Err(1) correctly, we must track errors for each *observation*
            % strictly when it is Out-Of-Bag (OOB).
            
            % oob_err_sum(i, k): Sum of errors for observation i, for model size k.
            oob_err_sum = zeros(N, P+1);
            
            % oob_counts(i): Number of times observation i appeared in the OOB set.
            oob_counts = zeros(N, 1);
            
            %Step 3: The Bootstrap Loop
            for b = 1:B
                %Resample (Bootstrap Trainning Set)
                %Create a dataset of size N by sampling WITH replacement
                boot_idx = randsample(N, N, true); 
                
                %Identify OOB(Test Set)
                % The observations NOT selected in the bootstrap sample.
                % On average, this is ~36.8% of the data.
                oob_idx = setdiff(1:N, boot_idx); 
                
                if isempty(oob_idx), continue; end 
                
                %Track that these specific observations were OOB in this round
                % IMPORTANT: Done ONCE per bootstrap round, outside the K loop.
                oob_counts(oob_idx) = oob_counts(oob_idx) + 1;
                
                %Nested Search
                % Find the best features using ONLY the bootstrap sample.
                % This simulates the model building process on a perturbed dataset.
                models_boot = obj.exhaustiveSearch(obj.Dataset.X(boot_idx,:), obj.Dataset.Y(boot_idx));
              
                %Predicted OOB
                %Prepare the "Test Data" (OOB observations)
                X_oob = obj.Dataset.X(oob_idx, :);
                Y_oob = obj.Dataset.Y(oob_idx);
                
                %Loop through all model sizes k (0 to P). COMPLEXITY LOOP
                for k = 1:(P+1)
                    m = models_boot(k);
                    
                    % Handle Null Model vs Active Model
                    if m.Size == 0
                        % Null: No features, usually predicts Mean(Y)
                        % We construct a zero-column matrix so Predict handles it.
                        X_sub = zeros(numel(oob_idx), 0);
                    else
                        % Active: Select only the features chosen in *this* bootstrap rep
                        X_sub = X_oob(:, m.Mask);
                    end
                    
                    %Generate Prediction
                    y_pred = obj.Strategy.predict(m.Model, X_sub);
                    
                    % Calculate Vectorized Error. Raw Squared Error for
                    % Every item in the OOB set (Test set)
                    vec_err = (Y_oob - y_pred).^2;
                    
                    %Accumulate error for each specific observation i
                    for j = 1:numel(oob_idx) %Local Scope
                        oid = oob_idx(j); %Get Global ID from the Mapping
                        oob_err_sum(oid, k) = oob_err_sum(oid, k) + vec_err(j); 
                    end
                end
            end
            
            %Compute Leave-One-Out Bootstrap Error (Err_1)
            % Err(1) is defined as the average error of observation i, calculated 
            % only over the bootstrap repetitions where i was OOB.
            
            % Avoid division by zero for observations that were never OOB (unlikely for large B)
            valid = oob_counts > 0;
            
            %Calculate average error per observation
            err_1_obs = zeros(N, P+1);
            err_1_obs(valid,:) = oob_err_sum(valid,:) ./ oob_counts(valid);
            
            %Average across all N observations to get the global Err(1)
            err_1_global = mean(err_1_obs, 1)';
            
            %Compute Final .632 Estimator
            % Weighted average of the Apparent Error (Training) and the Leave-One-Out Error.
            err632 = 0.368 * apparent_err + 0.632 * err_1_global;
        end
        function tbl = compileBasicResults(obj)
            % Initializes results table with K, Features, and TrainError
            P = obj.Dataset.P;
            tbl = table('Size', [P+1, 3], 'VariableTypes', {'double', 'string', 'double'}, ...
                'VariableNames', {'K', 'Features', 'TrainErr'});
            
            for i = 1:(P+1)
                bm = obj.BestModels(i);
                tbl.K(i) = bm.Size;
                tbl.TrainErr(i) = bm.Error / obj.Dataset.N; % MSE
                if bm.Size == 0
                    tbl.Features(i) = "(Null)";
                else
                    tbl.Features(i) = strjoin(obj.Dataset.Names(bm.Mask), ', ');
                end
            end
        end
        function tbl = appendInformationCriteria(obj, tbl)
            % Appends AIC/BIC if supported
            N = obj.Dataset.N;
            P = obj.Dataset.P;
            
            aic_col = zeros(P+1, 1);
            bic_col = zeros(P+1, 1);
            
            for i = 1:(P+1)
                bm = obj.BestModels(i);
                k_param = obj.Strategy.getComplexity(bm.Model);
                rss = bm.Error;
                %We Pass RSS and N, letting the function handle the
                %Gaussian logL conversion. We use 'AICc' if sample size is
                %small relative to features (Rule of thumb: N/k <40)
                if (N/ (k_param+1)) < 40
                    variant = "AICc";
                else
                    variant = "Standard";
                end
                aic_col(i) = OceanStats.ModelSelection.Criteria.aic(rss, k_param, 'N', N, 'Variant', variant);
                bic_col(i) = OceanStats.ModelSelection.Criteria.bic(rss, k_param, 'N', N);
                %BIC           
            end
            
            tbl.AIC = aic_col;
            tbl.BIC = bic_col;
        end
        
    end
end