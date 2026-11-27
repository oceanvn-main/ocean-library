classdef (Abstract) INormScheme < handle
    % INORMSCHEME  Abstract interface for row-wise normalization strategies.
    
    methods (Abstract)
        % NORMALIZE  Apply the normalization to the input matrix W
        %   W = normalize(obj, W)
        W = normalize(obj, W)
        
        % NAME  Return a short string identifier for the scheme (e.g. "l2", "l1", "none")
        n = name(obj)
        
        % SYMBOLICLATEX  Return the closed-form formula of the normalization
        str = symbolicLatex(obj)
    end
end
