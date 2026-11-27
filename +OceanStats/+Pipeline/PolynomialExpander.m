classdef PolynomialExpander < OceanStats.ModelSelection.IModelStrategy
    % POLYNOMIALEXPANDER Transformer that expands features to polynomial space.
    %
    % OVERVIEW:
    %   Takes an input matrix X (N x 1) and expands it to a Design Matrix 
    %   Phi = [X^0, X^1, ..., X^d].
    %   Currently supports 1D input features.
    %
    % IMPORTANT ARCHITECTURAL DISTINCTION vs PolynomialMapper:
    %   - PolynomialExpander (this class) extends IModelStrategy for use in 
    %     statistical pipelines (GLMs, Bayesian Model Averaging). It is designed 
    %     to compute full multivariate basis expansions (including interactions like X1*X2).
    %   - PolynomialMapper (in +Preprocessing) extends IPreprocessor for use in 
    %     Neural Networks. It computes only simple univariate stacks ([X1; X1^2]) 
    %     without interactions, leaving interaction learning to the NN hidden layers.
    %
    % COMPLIANCE:
    %   Implements IModelStrategy, but 'train' is a no-op (stateless transformer).
    %   Primary method is 'transform'.
    
    properties
        Name string
        Degree double
        SupportsAIC logical = false % Transformers don't have AIC
    end
    
    methods
        function obj = PolynomialExpander(degree)
            obj.Degree = degree;
            obj.Name = "PolyExpander (d=" + degree + ")";
        end
        
        function model = train(obj, ~, ~)
            % Stateless transformer - nothing to learn
            model = struct(); 
        end
        
        function X_new = predict(obj, ~, X)
            % Alias for transform to satisfy IModelStrategy
            X_new = obj.transform(X);
        end
        
        function X_new = transform(obj, X)
            % TRANSFORM Maps X to polynomial features using generator utility.
            X_new = OceanStats.Data.polyFeatures(X, obj.Degree);
        end
        
        function err = evaluate(~, ~, ~)
             err = NaN; % Not applicable
        end
        
        function k = getComplexity(~, ~)
            k = 0; % Transformers don't add parameters to the model count directly
        end
    end
end
