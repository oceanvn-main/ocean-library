classdef GCVCriterion < OceanStats.Base.IPruningCriterion
    % GCVCRITERION Generalized Cross-Validation criterion for MARS pruning.
    %
    % Implements the GCV penalty from Friedman (1991):
    %
    %   GCV(M) = RSS / ( N * (1 - C(M)/N)^2 )
    %
    % where C(M) = numTerms + penaltyCost * (numTerms - 1) is the effective
    % number of parameters (intercept is free, each other term has cost c).
    %
    % PenaltyCost is read from options.PenaltyCost (default 3.0 per ESL).
    %
    % Reference: Friedman (1991) "Multivariate Adaptive Regression Splines"
    %   Ann. Stat. 19(1), pp.1-141.
    
    methods
        function score = evaluate(~, rss, N, numTerms, options)
            % EVALUATE Compute GCV score (lower = better model).
            
            penalty = 3.0;
            if isfield(options, 'PenaltyCost')
                penalty = options.PenaltyCost;
            end
            
            non_constant  = numTerms - 1;
            effective_par = numTerms + penalty * non_constant;
            
            if effective_par >= N
                score = inf;
            else
                denom = N * (1 - (effective_par / N))^2;
                score = rss / denom;
            end
        end
    end
end
