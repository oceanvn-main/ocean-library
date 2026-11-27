function score = bic(inputs, k, options)
% BIC Compute Bayesian Information Criterion (Schwarz Criterion).
%
%   score = bic(logL, k, 'N', N)
%   score = bic(RSS, k, 'N', N)
%
%   OVERVIEW:
%       Calculates the BIC score, a model selection metric that penalizes 
%       complexity more heavily than AIC. While AIC tries to minimize predictive 
%       error (asymptotically equivalent to Cross-Validation), BIC tries to 
%       identify the "True Model" in the candidate set.
%
%       As N increases, BIC's penalty (k * ln(N)) grows larger than AIC's 
%       penalty (2k), leading to sparser, more parsimonious models.
%
%   FORMULAS:
%       1. General Form:    BIC = -2 * ln(L) + k * ln(N)
%       2. Gaussian (RSS):  BIC = N * ln(RSS/N) + k * ln(N) + C
%          (Where C is a constant involving 2*pi, usually ignored in textbooks 
%           but calculated here for exact likelihood consistency).
%
%   INPUTS:
%       inputs (double) : Primary metric.
%                         - IF 'N' is NaN: Interpreted as Log-Likelihood (logL).
%                         - IF 'N' is set: Interpreted as Residual Sum of Squares (RSS).
%
%       k      (double) : Number of estimated parameters (degrees of freedom).
%                         For Regression: k = num_features + 1 (intercept).
%                         For Trees:      k = num_branch_nodes + 1.
%
%   NAME-VALUE OPTIONS:
%       'N'    (double) : Sample size. 
%                         - REQUIRED for BIC (unlike Standard AIC).
%                         - The penalty term depends strictly on log(N).
%
%   EXAMPLES:
%       % 1. Logistic Regression (LogL = -540, k=5, N=1000)
%       score = bic(-540, 5, 'N', 1000);
%
%       % 2. Linear Regression (RSS = 120.5, k=4, N=100)
%       score = bic(120.5, 4, 'N', 100);
%
%   See also OCEANSTATS.MODELSELECTION.AIC, OCEANSTATS.MODELSELECTION.MALLOWS.
    arguments
        inputs (1, 1) double
        k (1, 1) double {mustBeNonnegative}
        options.N (1, 1) double = NaN   
        options.Type (1, 1) string {mustBeMember(options.Type, ["LogLik", "RSS"])} = "LogLik"
    end

    %Validation
    %BIC strictly requires N because the penalty term is k*ln(N)
    if isnan(options.N)
        error('OceanStats:MissingN', ...
            'BIC requires the sample size ''N'' to calculate the complexity penalty k*ln(N).');
    end
    
    if options.N <=0
        error('OceanStats:InvalidN', 'Sample size N must be positive.');
    end
    N = options.N;

    % --- 2. Standardize Inputs (RSS -> logL) ---
    switch lower(options.Type)
        case 'loglik'
            logL = inputs;
        case 'rss'
            if inputs <= 0
                error('OceanStats:BIC:InvalidRSS', ...
                      'RSS must be strictly positive.');
            end
            % Gaussian log-likelihood (with constant term)
            logL = -0.5 * N * (log(2*pi) + log(inputs / N) + 1);
    end
    % --- 3. Compute BIC ---
    % Formula: -2*ln(L) + k*ln(N)
    score = -2 * logL + k * log(N);

end

    

