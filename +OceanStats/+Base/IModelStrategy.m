classdef (Abstract) IModelStrategy < handle
    % IMODELSTRATEGY Interface for any Machine Learning algorithm.
    %
    %   PROFESSIONAL CONTRACT:
    %   Any model used in the 'FeatureSelectionAudit' MUST implement these 
    %   methods. This allows the audit engine to drive Linear Regression, 
    %   Decision Trees, or SVMs interchangeably.
    
    properties (Abstract)
        Name string           % Human-readable name (e.g., "Decision Tree")
        SupportsAIC logical   % Can this model calculate AIC/BIC? (True/False)
    end
    
    methods (Abstract)
        % TRAIN: Fits the model to data.
        % Inputs: X (Features), Y (Targets), Weights (Optional, N x 1)
        % Output: A model object/struct (Lightweight preferred)
        model = train(obj, X, Y, Weights)
        
        % PREDICT: Generates predictions from a trained model.
        % Inputs: model (from train), X (New Features)
        % Output: Predictions (y_hat for regression, probabilities for classification)
        preds = predict(obj, model, X)
        
        % EVALUATE: Calculates the error metric to be minimized.
        % Inputs: Y_true, Y_pred
        % Output: Scalar error (MSE, LogLoss, Misclassification Rate)
        err = evaluate(obj, Y_true, Y_pred)
        
        % GETCOMPLEXITY: Returns the "Degrees of Freedom" or complexity proxy.
        % Output: Scalar integer (k)
        k = getComplexity(obj, model)
    end
end