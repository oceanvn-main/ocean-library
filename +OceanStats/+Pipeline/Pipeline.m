classdef Pipeline < OceanStats.ModelSelection.IModelStrategy
    % PIPELINE Sequentially applies a list of transforms and a final estimator.
    %
    % CONCEPT:
    %   Machine learning is rarely just "fitting a model". It usually involves
    %   a sequence of steps: Scaling -> Feature Expansion -> Fitting.
    %   A Pipeline bundles these steps into a single object that looks and
    %   behaves like a standard model.
    %
    % CRITICAL ROLE - PREVENTING DATA LEAKAGE:
    %   By bundling the steps, the Pipeline ensures that transformations
    %   (like scaling) on Test Data use the statistics (Mean/Std) learned
    %   strictly from the Training Data.
    %
    % USAGE:
    %   pipe = Pipeline({StandardScaler(), PolyExpander(2), LinearReg()});
    %   model = pipe.train(X_train, Y_train);  % Learns means, fits poly, fits regression
    %   y_hat = pipe.predict(model, X_test);   % Re-uses means, expands poly, predicts
    %
    % Author: Ocean Library System
    % Date: 2026-02-12

    properties
        Name string = "Pipeline"
        SupportsAIC logical
        Steps cell % Cell array of IModelStrategy objects (The ordered assembly line)
    end
    
    methods
        function obj = Pipeline(steps)
            % PIPELINE Constructor
            % Input: steps (cell array) 
            %   - The first N-1 elements are Transformers (modify X).
            %   - The LAST element must be an Estimator (predicts Y).
            arguments
                steps cell
            end
            obj.Steps = steps;
            
            % IDENTITY MASQUERADE:
            % The pipeline "pretends" to be the final estimator.
            % If the final step is LinearReg, the pipeline reports itself as such.
            final_step = steps{end};
            obj.Name = "Pipeline -> " + final_step.Name;
            obj.SupportsAIC = final_step.SupportsAIC;
        end
        
        function model = train(obj, X, Y)
            % TRAIN Sequentially transform X, then train the final estimator.
            % This is the "Blueprint Creation" phase.
            
            X_current = X; % X_current flows through the pipe, changing shape at each step.
            num_steps = length(obj.Steps);
            
            % We must store the "fitted state" of every step (e.g., the Mean of the Scaler)
            % so we can reproduce it exactly during testing.
            step_fits = cell(1, num_steps);
            
            % -------------------------------------------------------------
            % PHASE 1: Run Transformers (Steps 1 to N-1)
            % -------------------------------------------------------------
            for i = 1:(num_steps - 1)
                transformer = obj.Steps{i};
                
                % 1. Learn the Transformation (e.g., Calculate Mean/Std of X)
                step_fit = transformer.train(X_current, Y);
                step_fits{i} = step_fit;
                
                % 2. Apply the Transformation (e.g., X_new = (X - Mean) / Std)
                % Note: In IModelStrategy, 'predict' on a transformer acts as 'transform'.
                X_current = transformer.predict(step_fit, X_current);
            end
            
            % -------------------------------------------------------------
            % PHASE 2: Train Final Estimator (Step N)
            % -------------------------------------------------------------
            % Now that X_current is fully processed (scaled, expanded, etc.),
            % we feed it to the actual model (e.g., Linear Regression).
            estimator = obj.Steps{end};
            final_fit = estimator.train(X_current, Y);
            step_fits{num_steps} = final_fit;
            
            % Encapsulate the entire state of the assembly line into one struct
            model.StepFits = step_fits;
        end
        
        function preds = predict(obj, model, X)
            % PREDICT Sequentially transform X, then predict with final estimator.
            % This is the "Assembly Line Execution" phase.
            %
            % CRITICAL: We do NOT call 'train' here. We use the saved 'step_fits'.
            
            X_current = X;
            num_steps = length(obj.Steps);
            
            % -------------------------------------------------------------
            % PHASE 1: Run Transformers (Using OLD learned logic)
            % -------------------------------------------------------------
            for i = 1:(num_steps - 1)
                transformer = obj.Steps{i};
                
                % RETRIEVE state (e.g., the Mean from the Training Set)
                step_fit = model.StepFits{i}; 
                
                % TRANSFORM test data using Training logic (Prevent Data Leakage)
                X_current = transformer.predict(step_fit, X_current);
            end
            
            % -------------------------------------------------------------
            % PHASE 2: Predict with Final Estimator
            % -------------------------------------------------------------
            estimator = obj.Steps{end};
            final_fit = model.StepFits{end};
            preds = estimator.predict(final_fit, X_current);
        end
        
        function err = evaluate(obj, Y_true, Y_pred)
            % EVALUATE Standard MSE calculation.
            err = mean((Y_true - Y_pred).^2);
        end
        
        function k = getComplexity(obj, model)
            % GETCOMPLEXITY Delegates complexity check to the final estimator.
            %
            % Rationale: While transformers increase data dimensionality (e.g., Poly^2),
            % the "Statistical Complexity" (k) for AIC/BIC is usually defined by the 
            % number of coefficients in the final model that consumes that data.
            estimator = obj.Steps{end};
            final_fit = model.StepFits{end};
            k = estimator.getComplexity(final_fit);
        end
    end
end