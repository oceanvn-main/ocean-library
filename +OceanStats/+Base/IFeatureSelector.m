classdef (Abstract) IFeatureSelector < handle
    % IFEATURESELECTOR Interface for feature selection algorithms.
    % 
    % METHODS:
    %   select(X, Y, strategy) -> [mask, history]
    
    methods (Abstract)
        % SELECT Returns a logical mask of selected features.
        % Inputs:
        %   X        : [N x P] Feature Matrix
        %   Y        : [N x 1] Target
        %   strategy : IModelStrategy (The evaluator)
        % Outputs:
        %   mask     : [1 x P] Logical mask
        %   history  : Struct containing search path/metrics (optional)
        [mask, history] = select(obj, X, Y, strategy)
    end
end
