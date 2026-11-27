function score = aic(inputs, k, options)
% AIC Compute Akaike Information Criterion (Universal & Corrected).
%
%   score = aic(logL, k)
%   score = aic(RSS, k, 'N', N)
%   score = aic(..., 'Variant', 'AICc')
%
%   Overview:
%       Calculates the AIC model selection metric. This function is "Universal"
%       because it accepts either raw Log-Likelihood (for GLMs like Logistic/Poisson)
%       or Residual Sum of Squares (for Gaussian/OLS models), automatically 
%       handling the conversion.
%
%   Formulas:
%       1. Standard AIC:  -2*ln(L) + 2k
%       2. AICc (Small N): AIC + (2k(k+1)) / (N - k - 1)
%       3. Gaussian L:    -0.5*N * (ln(2*pi) + ln(RSS/N) + 1)
%
%   Inputs:
%       inputs (double) : Primary metric. 
%                         - IF 'N' is NaN: Interpreted as Log-Likelihood (logL).
%                         - IF 'N' is set: Interpreted as Residual Sum of Squares (RSS).
%
%       k      (double) : Number of estimated parameters (degrees of freedom).
%                         For Regression: k = num_features + 1 (intercept).
%
%   Name-Value Options:
%       'N'       (double) : Sample size. 
%                            - REQUIRED if input is RSS.
%                            - REQUIRED if Variant is 'AICc' or 'ESL'.
%                            - Default: NaN.
%
%       'Variant' (string) : selection metric to compute.
%                            - "Standard" (Default): The classic asymptotic formula.
%                            - "AICc": Finite-sample corrected version. Strongly 
%                                      recommended when N/k < 40.
%                            - "ESL": Elements of Statistical Learning definition
%                                     (AIC / N), representing average loss.
%
%   Examples:
%       % 1. Logistic Regression (Using Log-Likelihood)
%       score = aic(-540.2, 5); 
%
%       % 2. Linear Regression (Using RSS, N=100, k=4)
%       score = aic(120.5, 4, 'N', 100);
%
%       % 3. Small Sample Correction (N=50, k=45) -> Prevents overfitting!
%       score = aic(120.5, 45, 'N', 50, 'Variant', 'AICc');
%
%   See also OCEANSTATS.MODELSELECTION.MALLOWS, GLMFIT.

    arguments 
        inputs (1, 1) double
        k (1, 1) double {mustBeNonnegative}
        options.N (1, 1) double = NaN
        options.Variant (1, 1) string {mustBeMember(options.Variant, ["Standard", "AICc", "ESL"])} = "Standard"
    end

    % 1. Determine if input is LogLikelihood or RSS
    if isnan(options.N)
        % Case A: Input is Log-Likelihood (Universal)
        logL = inputs;
    else
        % Case B: Input is RSS (Gaussian Helper)
        % If N is provided, we assume the user passed RSS.
        % We convert RSS to the exact Gaussian Log-Likelihood.
        
        if options.N <= 0
            error('OceanStats:InvalidN', 'Sample size N must be positive.');
        end
        
        RSS = inputs;
        N = options.N;
        
        % Exact Log-Likelihood formula for Gaussian distribution
        logL = -0.5 * N * (log(2*pi) + log(RSS/N) + 1);
    end

    % 2. Compute Standard AIC (The Base Score)
    % The fundamental tradeoff: Fit vs. Complexity
    score_standard = -2 * logL + 2 * k;
    
    % 3. Apply Variant (Correction or Scaling)
    switch options.Variant
        case "Standard"
            score = score_standard;
            
        case "AICc"
            % Corrected AIC (Hurvich & Tsai, 1989)
            % Adds an extra penalty term that explodes as k approaches N.
            % Essential for high-dimensional data or small sample sizes.
            
            if isnan(options.N)
                error("OceanStats:MissingN", "AICc requires sample size 'N' to compute the correction.");
            end
            
            % Avoid division by zero if saturation occurs (k >= N - 1)
            denominator = options.N - k - 1;
            if denominator <= 0
                score = Inf; % Model is too complex to be valid
            else
                correction = (2 * k * (k + 1)) / denominator;
                score = score_standard + correction;
            end
            
        case "ESL"
            % "Elements of Statistical Learning" definition
            % Scales AIC by 1/N to represent average prediction error.
            if isnan(options.N)
                error("OceanStats:MissingN", "ESL variant requires 'N' to average the scores.");
            end
            score = score_standard / options.N;
    end
end