classdef (Abstract) IValidator < handle
    % IVALIDATOR Interface for Model Validation Strategies.
    %
    % METHODS:
    %   validate(obj, selector, strategy, X, Y) -> result
    %
    %   - selector: IFeatureSelector (e.g. BestSubset)
    %   - strategy: IModelStrategy (e.g. LinearReg)
    
    properties (Abstract)
        Name string
    end
    
    methods (Abstract)
        % VALIDATE Runs the validation process (e.g., Nested CV).
        % Returns a struct containing metrics (Error, SE, History).
        result = validate(obj, selector, strategy, X, Y)
    end
end
