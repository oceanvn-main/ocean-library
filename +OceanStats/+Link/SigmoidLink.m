classdef SigmoidLink < OceanStats.Base.ILinkFunction
    % SIGMOIDLINK Logistic (sigmoid) inverse-link for binary classification.
    %
    % apply(Z):   P = 1 / (1 + exp(-Z))   [column vector or N x 1]
    % weights(P): w = P .* (1 - P)         [Bernoulli Fisher information]
    
    methods
        function P = apply(~, Z)
            P = 1 ./ (1 + exp(-Z));
        end
        
        function w = weights(~, P)
            w = max(P .* (1 - P), 1e-8);
            % Collapse to N×1 if multi-column (shouldn't happen for binary)
            if size(w, 2) > 1
                w = mean(w, 2);
            end
        end
    end
end
