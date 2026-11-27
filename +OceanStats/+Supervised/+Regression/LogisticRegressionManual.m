%Questions: We want to predict prob that an obs X belongs to a specific
%class G with constraints: 1. Prob stay [0, 1]. 2. Sum or prob = 1.
%Solution: Log-odd
%Model: Log[P(G = k | X= x)/ P(G = relative class | X = x)] = Beta * X
%Fitting by Maximum Likelihood using the conditional likelihood of G given
%X (P(G|X).


function [beta, deviance, covb] = Logistic_Regression_Newton_Manual(X, y)
%Fits a binary Logistic Regression Model using the Newton-Raphson algorithm
%Iteratively Reweighted Least Squares - IRLS

%Theory & Assumptions
%1. Model Assumption (Bernoulli):
%The target y_i follows a Bernoulli distribution: y_i ~ Bernoulli(p_i).
%Probability mass function: P(Y = y_i) = p_i^(y_i) * (1 - p_i)^(1-y_i)
%2. Linearity Assumption (Logit Link):
%We assume the Log-Odds (Logit) is a linear combination of inputs X.
%eta_i = log(pi / (1 - pi)) = beta_0 + beta_1 * x_x1 + ....
%      = X_i * beta
%3. Optimization Objective (Maximum Likelihood):
%We want to find beta that maximize the Log-Likelihood function l(beta).
% l(beta) = Sum [ yi  * log(p_i) + ( 1- y_i) * log(1 - p_i)]
%Simply further: 
% l(beta) = Sum [ y_i * Beta * x_i - log(1 + exp (Beta *x_i))] (E.q 4.20)
% Solve: Max eq. 4.20 using Newton-Raphson rule for update beta_new
%First derivative of Logistic function (p(z)= e^z/ (1 + e^z) ) -> 
% p' (z) = p(z) * ( 1 - p(z)). 
%Second derivative. 
% beta_new = beta_old - (Hessian)^-1 * Gradient
%Questions: How to construct Hessian Matrix effectively?
%Solution: IRLS

    [N, p] = size(X);
    
    %Initialize Beta with zeros
    %Starting guess: No relationship between X and y.
    beta = zeros(p, 1);
    
    %Convergence criteria
    max_iter = 25;
    tol = 1e-6;
    
    %The Optimization Loop (Newton- Raphson)
    %Newton's method seeks the root of the gradient where slope = 0.
    %Formula: beta_new = beta_old - (Hessian)^-1 * Gradient
    for iter = 1:max_iter
        %Step A: Linear Predictor (Log-Odds)
        %We project the input data X onto the parameter vector beta.
        %eta ranges from -inf to +inf.
        eta = X * beta;
        
        %Step B: Predicted Probabilities (Sigmoid/ Inverse Logit)
        %We map the linear score eta back to the probability space [0,1]
        %ln (mu / (1-mu)) = eta 
        %Formula: p = exp(eta) / ( 1 + exp(eta)) = 1 / (1 + exp(-eta))
        mu  = 1 ./ (1  + exp(-eta));
        
        %Step C: Calculate Weights (Variance)
        weights = mu .* (1 - mu);
        
        %Numerical Stability
        %if p is exactly 0 or 1, weight become 0, making Hessian Singular.
        weights(weights < 1e-8) = 1e-8;
        
        %Step D: Calculate Gradient(Score vector)
        %dl/dbeta = Sum [xi* (yi - pi)]
        %Matrix Form: Gradient = X' * ( y - p)
        grad = X' * (y - mu);
        
        %Step E: Calculate Hessian ( Information Matrix)
        %d^2l/dbeta^2 = - Sum(xi * xi' * pi ( 1 - pi)]
        %Matrix Form: Hesian = - X' * W * W
        %Efficient Computation
        %Instead of making a huge diagonal matrix W, we compute X weighted.
        % (X * sqrt(W))' * (X * sqrt(W)) is mathematically equivalent to X'WX.
        X_weighted = X .* sqrt(weights);
        
        %Fisher Information Matrix (I) = - Hesian
        Fisher_Info = X_weighted' * X_weighted;
        
        %The Actual Hessian is negative definite
        %Hessian = -Fisher_Info;
        
        %Step F: The Newton Update Step
        %General Newton Rule: x_new = x_old - f'(x)/ f''(x)
        %Vector Form: beta_new = beta_old - (Hessian)^-1 * Gradient
        %Substituting Hesian = -Fisher_Info:
        %beta_new = beta_old - (-Fisher_Info)^-1 * Gradient
        %beta_new = beta_old + (Fisher_Info)^-1 * Gradient
        %We solve the linear system (Fisher_Info *delta = Gradient)
        %Using backslash is numerically more stable than inv.
        %Fisher_Infor is X'WX
        %grad is X'(y-p)
        %beta_new = b_old + (X'WX)^-1 X'(y-p) = b_old + Fisher^-1 * Gradient
        lambda = 1e-5; %STABILIZER: add A tiny number to the diagonal (Ridge Regularization)
        %This prevent division by zero when features are highly correlated.
        delta = (Fisher_Info + lambda * eye(p)) \ grad;
        beta = beta + delta;
    
        % %Step G: Check Convergence
        % if norm(delta) < tol
        %     fprintf('Converged at iteration %d. \n', iter);
        %     break; % FIX: Exit loop upon convergence
        % end
    end
    
    %%3. Post-Estimation Statistics
    % (Moved outside the loop to calculate only once using final beta)
    
    %Re-calculate final probabilities
    eta = X * beta;
    mu = 1 ./ (1 + exp(-eta));
    
    %Deviance (Goodness of Fit)
    %Formula D = -2 * LogLikeliHood
    %LogLikelihood = Sum [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
    mu_safe = max(min(mu, 1-1e-10), 1e-10); % Avoid log(0)
    ll = sum(y .* log(mu_safe) + (1-y) .* log(1-mu_safe));
    deviance = -2 * ll;
    
    %Covariance Matrix of Coefficients
    %Asymptonic theory states that the variance of the MLE estimates is
    %the inverse of the Fisher Information Matrix
    %Cov(beta) approx (X'WX)^-1
    weights = mu .* (1- mu);
    weights(weights < 1e-8) = 1e-8;
    
    % FIX: Variable name consistency (X_weighted vs X_weights)
    X_weighted = X .* sqrt(weights); 
    Fisher_Info = X_weighted' * X_weighted;

    covb = inv(Fisher_Info + lambda * eye(p));

end