function I = MutualInfo(X, Y, options)
% MUTUALINFO Compute Mutual Information I(X; Y)
%
% I(X; Y) = H(X) + H(Y) - H(X, Y)
%
% Usage:
%   I = OceanMath.Information.MutualInfo(X, Y)
%   I = OceanMath.Information.MutualInfo(X, Y, Name, Value)
%
% Inputs:
%   X, Y (vector): Data vectors of same length.
%
% Options:
%   Type (string): "Discrete" (default) or "Continuous"
%   Method (string): "Auto", "KDE", "KSG"
%   Base (numeric): Logarithm base (default: 2)
%
% Examples:
%   I = OceanMath.Information.MutualInfo(X, Y)
%
% See also: OceanMath.Information.Entropy

    arguments
        X
        Y
        options.TypeAsString (1,1) string {mustBeMember(options.TypeAsString, ["Auto", "Discrete", "Continuous"])} = "Auto"
        options.Method (1,1) string = "Auto"
        options.Base (1,1) double {mustBePositive} = 2
    end

    if numel(X) ~= numel(Y)
        error('OceanMath:Information:MutualInfo:SizeMismatch', 'X and Y must have same number of elements.');
    end
    
    % Auto-detect type
    if options.TypeAsString == "Auto"
        is_discrete_X = iscategorical(X) || isstring(X) || islogical(X) || (isnumeric(X) && (all(mod(X,1)==0) || numel(unique(X)) < sqrt(numel(X))));
        is_discrete_Y = iscategorical(Y) || isstring(Y) || islogical(Y) || (isnumeric(Y) && (all(mod(Y,1)==0) || numel(unique(Y)) < sqrt(numel(Y))));
        
        if is_discrete_X && is_discrete_Y
            type = "Discrete";
        else
            type = "Continuous";
        end
    else
        type = options.TypeAsString;
    end

    if type == "Discrete"
        % For discrete, I(X;Y) = sum p(x,y) log(p(x,y)/(p(x)p(y)))
        % Efficient calculation using unique and accumarray
        
        % Map to integers 1..K
        [~, ~, idxX] = unique(X);
        [~, ~, idxY] = unique(Y);
        
        % Joint distribution
        % Trick: combine indices to get unique pairs
        n = numel(X);
        max_X = max(idxX);
        joint_idx = idxX + (idxY - 1) * max_X;
        
        [~, ~, u_joint] = unique(joint_idx);
        counts_joint = accumarray(u_joint, 1);
        p_joint = counts_joint / n;
        
        counts_X = accumarray(idxX, 1);
        p_X = counts_X / n;
        
        counts_Y = accumarray(idxY, 1);
        p_Y = counts_Y / n;
        
        % Calculate MI
        % Need to match p_joint(k) with p_X(i) and p_Y(j)
        % Reconstruct i and j from u_joint is hard if we didn't save mapping
        
        % Better way: H(X) + H(Y) - H(X,Y)
        HX = -sum(p_X .* log2(p_X));
        HY = -sum(p_Y .* log2(p_Y));
        HXY = -sum(p_joint .* log2(p_joint));
        
        I = HX + HY - HXY;
        
        if options.Base ~= 2
            I = I * log(2) / log(options.Base);
        end
        
        % Ensure non-negative due to float errors
        I = max(0, I);
        
    else
        % Continuous
        % If KSG (Kraskov) is requested (TODO)
        % For now, use Histogram based or Gaussian approximation
        
        if options.Method == "Gaussian"
            % I = -0.5 * log(1 - rho^2)
            c = corr(X(:), Y(:));
            I = -0.5 * log(1 - c^2);
             if options.Base ~= 2
                I = I * log(exp(1)) / log(options.Base);
            end
        else
             % Fallback to Entropy difference via Histogram (simple binning)
             % Auto binning for X, Y, and Joint
             % This is tricky for Joint. 
             % Simplest MVP: Discretize both and call Discrete MI
             
             % Equal frequency binning (copula transform) is robust
             n_bins = round(sqrt(numel(X)/5)); % Rule of thumb
             if n_bins < 2, n_bins = 2; end
             
             % Quantize X
             edgesX = quantile(X, linspace(0, 1, n_bins+1));
             edgesX(1) = -inf; edgesX(end) = inf;
             discX = discretize(X, edgesX);
             
             % Quantize Y
             edgesY = quantile(Y, linspace(0, 1, n_bins+1));
             edgesY(1) = -inf; edgesY(end) = inf;
             discY = discretize(Y, edgesY);
             
             % Call discrete MI
             I = OceanMath.Information.MutualInfo(discX, discY, "TypeAsString", "Discrete", "Base", options.Base);
             
             % Note: This is an estimator of I(X;Y), not differential entropy difference exactly,
             % but for MI, discretization is a valid lower bound estimator.
        end
    end
end
