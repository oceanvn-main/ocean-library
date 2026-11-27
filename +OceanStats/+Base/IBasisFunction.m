classdef IBasisFunction < handle
    % IBASISFUNCTION Interface for generatable basis functions (Splines, Hinges, etc).
    %
    % Allows uniform evaluation and string representation of basis functions
    % across additive models (MARS, GAM, etc).
    
    methods (Abstract)
        % Evaluate the basis function on a set of features X (N x P matrix).
        % Returns an N x 1 column vector of evaluated basis values.
        val = evaluate(obj, X);
        
        % String representation of the basis function for display.
        % featNames is an optional cell array of feature names.
        str = toString(obj, featNames);
        
        % Returns the interaction degree (depth) of this basis function.
        deg = getDegree(obj);
    end
end
