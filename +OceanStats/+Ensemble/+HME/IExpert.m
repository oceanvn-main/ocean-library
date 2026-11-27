classdef (Abstract) IExpert < handle
    % IEXPERT Abstract interface for HME experts.
    
    methods (Abstract)
        % PREDICT Computes the expert's local prediction (e.g., \mu = \theta' * X).
        Y_hat = predict(obj, X)
        
        % CALCULATELIKELIHOOD Computes the probability density P(y|x, \theta).
        likelihood = calculateLikelihood(obj, X, Y)
        
        % UPDATE Performs the M-step update using local responsibilities as weights.
        update(obj, X, Y, responsibility)
        
        % State Persistence
        params = getParams(obj)
        setParams(obj, params)
        reset(obj)
        
        % Advanced Warmup
        warmup(obj, X, Y)
        
        % Domain Identity (Used for pool validation)
        type = getMeasureType(obj) % Returns 'Density' or 'Mass'
    end
end
