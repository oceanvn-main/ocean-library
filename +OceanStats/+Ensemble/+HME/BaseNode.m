classdef (Abstract) BaseNode < handle
    % BASENODE Abstract base class for Hierarchical Mixture of Experts (HME) tree nodes.
    % Inherits from 'handle' to use reference semantics (pointers), allowing in-place 
    % weight updates during the M-step without creating memory copies.
    % 
    % Theory: The Expectation-Maximization (EM) algorithm is decoupled into a 
    % message-passing architecture: Bottom-Up (Likelihood) -> Top-Down (Responsibility) -> Update.
    
    properties (Access = protected)
        % LocalResponsibility: N x 1 vector storing the posterior probabilities E[\Delta]
        % calculated during the E-step. This is frozen and used as observation weights in the M-step.
        LocalResponsibility 
        
        % LastLikelihood: N x 1 vector storing the likelihood P(y|x) calculated
        % during the most recent Bottom-Up pass (computeLikelihood).
        LastLikelihood
    end
    
    methods (Abstract)
        % PREDICT (The Forward Pass / Inference)
        % Computes the final predicted value Y_hat.
        % - Experts: Compute standard linear regression (Y_hat = theta' * X).
        % - Gates: Compute softmax routing probabilities (g_i) and return the weighted sum of children.
        Y_hat = predict(obj, X)
        
        % COMPUTELIKELIHOOD (E-Step Phase 1: Bottom-Up)
        % Calculates the raw probability of the data given the current frozen weights.
        % - Experts: Evaluate the statistical PDF (e.g., Gaussian P(y|x, theta)).
        % - Gates: Sum the likelihoods of children weighted by local routing probabilities (g_i).
        % Theory: By the time this reaches the root node, it forms the total network marginal 
        % likelihood (the massive double-summation denominator needed for Bayes' Theorem).
        likelihood = computeLikelihood(obj, X, Y, iter)
        
        % COMPUTERESPONSIBILITY (E-Step Phase 2: Top-Down)
        % Applies Bayes' Theorem to calculate the posterior probability (the "Responsibility" h).
        % - Formula: h = (prior_from_parent * local_likelihood) / totalLikelihood_from_root
        % Theory: This answers: "Given the final prediction, what exact percentage of this data 
        % point belongs to this specific node?" The result is saved to obj.LocalResponsibility.
        computeResponsibility(obj, X, Y, prior, totalLikelihood)
        
        % UPDATEPARAMETERS (The M-Step: Maximization)
        % Physically alters the network's internal weights using the frozen responsibility matrix.
        % - Experts: Solve Weighted Least Squares (WLS) using 'h' as observation volume knobs.
        % - Gates: Solve a K-class logistic regression (IRLS) using 'h' as the target response.
        % Theory: Because 'h' was pre-calculated and frozen, nodes update completely blindly 
        % and independently of each other, avoiding mathematical gridlock.
        updateParameters(obj, X, Y, iter)
        
        % GETRESPONSIBILITY
        % Retrieves the frozen N x 1 responsibility vector for M-step matrix algebra.
        h = getResponsibility(obj)
        
        % State Persistence & Reliability
        params = getParams(obj)
        setParams(obj, params)
        reset(obj)
        
        % Advanced Reliability
        warmup(obj, X, Y, labels)
        leaves = getLeaves(obj)
    end
end