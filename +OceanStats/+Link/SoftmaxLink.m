classdef SoftmaxLink < OceanStats.Base.ILinkFunction
    % SOFTMAXLINK Multinomial logistic inverse-link (row-wise softmax).
    %
    % apply(Z):   P_ik = exp(Z_ik) / sum_j exp(Z_ij)   [numerically stable]
    % weights(P): w_i  = mean_k P_ik*(1-P_ik)           [averaged Fisher info]
    
    methods
        function P = apply(~, Z)
            Z  = Z - max(Z, [], 2);   % subtract row max for stability
            eZ = exp(Z);
            P  = eZ ./ sum(eZ, 2);
        end
        
        function w = weights(~, P)
            K = size(P, 2);
            w = max(sum(P .* (1 - P), 2) / K, 1e-8);
        end
    end
end
