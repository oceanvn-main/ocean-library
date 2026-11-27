classdef ILinkFunction < handle
    % ILINKFUNCTION Interface for GLM link/inverse-link functions.
    %
    % Maps linear scores Z = B*beta to predicted probabilities/responses P,
    % and provides the corresponding Fisher information weights for WLS.
    %
    % Implementations:
    %   SoftmaxLink  — Multinomial logistic  (PolyMARS)
    %   SigmoidLink  — Binary logistic
    %   IdentityLink — Gaussian / OLS (P = Z, w = 1)
    
    methods (Abstract)
        % APPLY Inverse link: map linear scores to response space.
        %   Z - (N x K) linear scores
        %   P - (N x K) predicted values/probabilities
        P = apply(obj, Z)
        %Returns the diagonal of the Fisher information matrix — how much curvature the log-likelihood has at observation i
        % WEIGHTS Fisher information diagonal weights from current predictions.
        %   P - (N x K) current predictions (output of apply)
        %   w - (N x 1) scalar weight per observation
        w = weights(obj, P)
    end
end
