classdef (Abstract) IInitializer < handle
    % IINITIALIZER Abstract interface for HME structural initialization strategies.
    
    methods (Abstract)
        % INITIALIZE Performs the structural warmup of the HME tree.
        % This typically involves partitioning the data space and 
        % "pre-fitting" experts and gating networks to their territories.
        initialize(obj, model, X, Y)
    end
end
