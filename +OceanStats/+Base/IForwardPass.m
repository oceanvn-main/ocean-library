classdef IForwardPass < handle
    % IFORWARDPASS Interface for stepwise forward model builders.
    % 
    % The forward pass is responsible for aggressively growing an additive
    % model by adding genenerated basis functions.
    
    methods (Abstract)
        % BUILD
        % Builds the initial large set of MARS basis functions.
        % 
        % INPUTS:
        %   X        - (N x P) feature matrix
        %   Y        - (N x 1) target vector
        %   options  - struct of parameters (MaxTerms, MaxInteractionDegree, etc)
        %
        % OUTPUTS:
        %   basisList - cell array of OceanStats.Base.IBasisFunction
        basisList = build(obj, X, Y, options);
    end
end
