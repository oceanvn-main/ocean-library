classdef IPruningCriterion < handle
    % IPRUNINGCRITERION Interface for model selection criteria used in pruning.
    %
    % Evaluates the quality of a model with a given RSS and complexity,
    % returning a score that the backward pass minimises to select the
    % optimal model size.
    %
    % Implementations:
    %   GCVCriterion — Generalized Cross-Validation (default for MARS)
    
    methods (Abstract)
        % EVALUATE Compute the selection criterion score.
        %
        % INPUTS:
        %   rss      - scalar Residual Sum of Squares
        %   N        - number of observations
        %   numTerms - number of terms in the current model
        %   options  - struct (may contain PenaltyCost or other params)
        %
        % OUTPUT:
        %   score - scalar; lower is better
        score = evaluate(obj, rss, N, numTerms, options)
    end
end
