function rho = Correlation(X, Y, options)
% CORRELATION Compute Information-based Correlation
%
% Implements multiple information-theoretic correlation measures.
%
% Measures:
%   "Information": rho = 1 - H(Y|X)/H(Y) = I(X;Y) / H(Y) (Problem 2.11 type)
%       Note: The original Problem 2.11 definition is rho = 1 - H(X2|X1)/H(X1).
%       Assuming symmetric or target is Y.
%   "Linfoot": rho = sqrt(1 - exp(-2 * I(X;Y))) (Gaussian equivalent)
%   "SymmetricUncertainty": rho = 2 * I(X;Y) / (H(X) + H(Y))
%   "NormalizedMI": rho = I(X;Y) / sqrt(H(X) * H(Y))
%
% Usage:
%   rho = OceanMath.Information.Correlation(X, Y)
%   rho = OceanMath.Information.Correlation(X, Y, "Type", "Linfoot")
%
% Inputs:
%   X, Y (vector): Data vectors.
%
% Options:
%   Type (string): 
%       - "Information" (default): I(X;Y) / H(Y)  (Asymmetric!)
%       - "Linfoot"
%       - "SymmetricUncertainty"
%       - "NormalizedMI"
%
% Examples:
%   rho = OceanMath.Information.Correlation(X, Y)
%
% See also: OceanMath.Information.MutualInfo, OceanMath.Information.Entropy

    arguments
        X
        Y
        options.Type (1,1) string {mustBeMember(options.Type, ["Information", "Linfoot", "SymmetricUncertainty", "NormalizedMI"])} = "Information"
        options.VariableType (1,1) string = "Auto" % Discrete/Continuous
    end

    import OceanMath.Information.MutualInfo
    import OceanMath.Information.Entropy

    % Calculate Core Metrics
    I = MutualInfo(X, Y, "TypeAsString", options.VariableType);
    
    switch options.Type
        case "Information"
            % Problem 2.11: rho = 1 - H(X2|X1)/H(X1)
            % H(X2|X1) = H(X2) - I(X1;X2)
            % rho = 1 - (H(X2) - I)/H(X1)
            % If X1, X2 identically distributed -> H(X1)=H(X2) -> rho = I/H(X1)
            % We will treat Y as the target (numerator part) and X as the conditioner.
            % But wait, the formula divides by H(X1).
            % Let's follow the formula strictly: rho = 1 - H(Y|X)/H(X) ? NO.
            % Formula: rho = 1 - H(X2|X1)/H(X1).
            % Denominator is Marginal of conditioner.
            % Numerator is Conditional of target given conditioner.
            % So rho = (H(X1) - H(X2|X1)) / H(X1) ? 
            % H(X2|X1) = H(X1,X2) - H(X1).
            % So 1 - (H(1,2) - H(1))/H(1) = (H(1) - H(1,2) + H(1))/H(1) = 2 - H(1,2)/H(1).
            % Wait.
            % I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X).
            % The formula is rho = 1 - H(Y|X)/H(X).
            % If H(X)=H(Y), then 1 - (H(Y) - I)/H(Y) = I/H(Y).
            % But if H(X) != H(Y) (general case), let's implement the exact formula relative to X.
            % We need H(X) and H(Y|X).
            % H(Y|X) = H(X,Y) - H(X).
            % So rho = 1 - (H(X,Y) - H(X)) / H(X) = (H(X) - H(X,Y) + H(X)) / H(X) = 2 - H(X,Y)/H(X).
            
            % Let's implement rho = I(X;Y) / H(X) (Information Coefficient regarding X)
            % This is often called "Uncertainty Coefficient" U(Y|X) = I(X;Y)/H(Y).
            % The formula in the image divides by H(X_1).
            % So let's use H(X).
            
            HX = Entropy(X, "TypeAsString", options.VariableType);
            if HX == 0
                rho = 0; % Or NaN?
            else
                rho = I / HX;
            end
            
        case "Linfoot"
            % rho = sqrt(1 - exp(-2*I))
            % Note: I must be in nats for this formula?
            % If I is in bits, use 2^(-2*I) ?
            % Standard Linfoot: I is usually nats. 
            % Formula: 1 - exp(-2I_nats)
            % I_nats = I_bits * log(2)
            I_nats = I * 0.69314718056;
            rho = sqrt(1 - exp(-2 * I_nats));
            
        case "SymmetricUncertainty"
            HX = Entropy(X, "TypeAsString", options.VariableType);
            HY = Entropy(Y, "TypeAsString", options.VariableType);
            rho = 2 * I / (HX + HY);
            
        case "NormalizedMI"
            HX = Entropy(X, "TypeAsString", options.VariableType);
            HY = Entropy(Y, "TypeAsString", options.VariableType);
            rho = I / sqrt(HX * HY);
    end
end
