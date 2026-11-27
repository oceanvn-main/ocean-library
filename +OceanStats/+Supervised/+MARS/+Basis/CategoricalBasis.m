classdef CategoricalBasis < OceanStats.Base.IBasisFunction
    % CATEGORICALBASIS An indicator function for categorical predictors in MARS.
    % Representing I(X_j in S) where S is a subset of categories
    % As described in ESL 9.4.3 "Mixed Inputs"
    
    properties
        FeatureIndex (1,1) double
        Subset       (1,:) double
    end
    
    methods
        function obj = CategoricalBasis(featIdx, subset)
            obj.FeatureIndex = featIdx;
            obj.Subset = subset;
        end
        
        function val = evaluate(obj, X)
            x_j = X(:, obj.FeatureIndex);
            % Indicator function: 1 if x_j is in the Subset, 0 otherwise
            val = double(ismember(x_j, obj.Subset));
        end
        
        function str = toString(obj, featNames)
            if nargin > 1 && ~isempty(featNames)
                name = string(featNames{obj.FeatureIndex});
            else
                name = sprintf('X%d', obj.FeatureIndex);
            end
            
            subsetStr = "{" + join(string(obj.Subset), ",") + "}";
            str = "I(" + name + " \in " + subsetStr + ")";
        end
        
        function deg = getDegree(~)
            deg = 1;
        end
    end
end
