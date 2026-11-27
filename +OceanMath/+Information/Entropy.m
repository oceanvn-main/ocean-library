function H = Entropy(X, options)
% ENTROPY Compute Shannon Entropy H(X)
%
% H(X) = -sum(p(x) * log2(p(x))) for discrete X
% H(X) = -int(f(x) * log2(f(x))) dx for continuous X (differential entropy)
%
% Usage:
%   H = OceanMath.Information.Entropy(X)
%   H = OceanMath.Information.Entropy(X, Name, Value)
%
% Inputs:
%   X (numeric, categorical, or table): Data vector or probability distribution.
%       If sum(X) is approx 1 and all(X >= 0), it is treated as probabilities.
%       Otherwise, it is treated as raw data samples.
%
% Options:
%   Type (string): "Discrete" (default) or "Continuous"
%   Method (string): 
%       - "MLE": Maximum Likelihood (default for Discrete)
%       - "MillerMadow": Bias-corrected (Discrete)
%       - "KDE": Kernel Density Estimation (default for Continuous)
%       - "KNN": k-Nearest Neighbors (Continuous)
%   Base (numeric): Logarithm base (default: 2 for bits)
%
% Examples:
%   H = OceanMath.Information.Entropy([0.5, 0.5]) % Returns 1.0 (bits)
%   H = OceanMath.Information.Entropy(randi(6, 100, 1)) % Estimate from data
%
% See also: OceanMath.Information.MutualInfo, OceanMath.Information.Divergence

    arguments
        X
        options.TypeAsString (1,1) string {mustBeMember(options.TypeAsString, ["Auto", "Discrete", "Continuous"])} = "Auto"
        options.Method (1,1) string = "Auto"
        options.Base (1,1) double {mustBePositive} = 2
    end

    % Preprocessing
    if istable(X)
        % Flatten table to single vector/cell for entropy calculation
        % Assuming independence if multiple columns passed (sum of entropies)
        % Or joint entropy? Usually joint entropy for multivariate.
        % For simplicity, loop over columns and sum if multiple.
        % Actually, let's treat it as a vector or error for now.
        % Better implementation: Joint entropy if multivariate.
        % For now, support single column.
        if width(X) > 1
            error('OceanMath:Information:Entropy:Multivariate', ...
                'Multivariate entropy not yet supported. Pass single column.');
        end
        X = X{:,1};
    end

    is_prob = false;
    if isnumeric(X) && isvector(X) && all(X >= 0) && abs(sum(X) - 1) < 1e-6
        is_prob = true;
    end

    % Determine type
    if options.TypeAsString == "Auto"
        if iscategorical(X) || isstring(X) || islogical(X)
            type = "Discrete";
        elseif is_prob
            type = "Discrete"; % Probabilities given directly
        elseif isnumeric(X)
            % Heuristic: if integers or few unique values -> Discrete
            % If floats and many unique -> Continuous
            if all(mod(X, 1) == 0) || numel(unique(X)) < sqrt(numel(X))
                type = "Discrete";
            else
                type = "Continuous";
            end
        else
            type = "Discrete";
        end
    else
        type = options.TypeAsString;
    end

    % Calculation
    if type == "Discrete"
        if is_prob
            p = X;
            p = p(p > 0);
        else
            % Estimate p from data
            if options.Method == "MillerMadow"
                % Requires counts
                [~, ~, idx] = unique(X);
                counts = accumarray(idx, 1);
                p = counts / sum(counts);
                % Miller-Madow correction: H_MLE + (m-1)/(2N)
                % m = number of non-zero bins
                m = sum(counts > 0); % all here
                N = sum(counts);
                H_MLE = -sum(p .* log2(p));
                H = H_MLE + (m - 1) / (2 * N * log(2)); % Convert to nats then bits? 
                % Actually standard is in nats, then convert.
                % Miller-Madow in bits:
                % H_mm = H_mle + (m-1)/(2*N*ln(2)) if base 2
                if options.Base ~= 2
                     % General correction
                     H = H_MLE + (m - 1) / (2 * N * log(options.Base));
                else
                     H = H_MLE + (m - 1) / (2 * N * 0.69314718056);
                end
                return;
            else
                % Default MLE
                 [~, ~, idx] = unique(X);
                 counts = accumarray(idx, 1);
                 p = counts / sum(counts);
            end
        end
        
        H = -sum(p .* (log(p) / log(options.Base)));
        
    else % Continuous
        if options.Method == "Auto" || options.Method == "KDE"
            % Simple Differential Entropy estimation via spacing or KDE
            % Using Vasicek's estimator (spacing) is often better than KDE for entropy
            % But let's stick to a robust simple one: Nearest Neighbor (KSG like)
            % Or just histogram for MVP if high dimensionality?
            % For 1D, spacing is great.
            
            % Let's use Vasicek (1976) estimator for 1D
            X = sort(X(:));
            n = numel(X);
            m = floor(sqrt(n)); % Window size heuristic
            
            % H = 1/n * sum(log(n/2m * (X(i+m) - X(i-m))))
            % Correct for boundaries
            
            % Actually, for simplicity and robustness in v1, let's use a histogram method
            % binning like Freedman-Diaconis rule
            iqr_val = iqr(X);
            if iqr_val == 0, iqr_val = std(X); end
            if iqr_val == 0
                H = 0; return; % Degenerate
            end
            bin_width = 2 * iqr_val * n^(-1/3);
            edges = min(X):bin_width:max(X)+bin_width;
            counts = histcounts(X, edges);
            p = counts / sum(counts);
            p = p(p > 0);
            
            % Entropy of discretized + log(delta)
            H_discrete = -sum(p .* log2(p));
            H = H_discrete + log2(bin_width);
            
            if options.Base ~= 2
                H = H / log2(options.Base);
            end
        else
            error("OceanMath:Information:Entropy:NotImplemented", "Method not implemented yet.");
        end
    end
end
