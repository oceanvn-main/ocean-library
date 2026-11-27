classdef IdentityLink < OceanStats.Base.ILinkFunction
    % IDENTITYLINK Identity inverse-link for Gaussian / OLS models.
    %
    % apply(Z):   P = Z          (no transformation)
    % weights(P): w = ones(N,1)  (unit weights = ordinary least squares)
    
    methods
        function P = apply(~, Z)
            P = Z;
        end
        
        function w = weights(~, P)
            w = ones(size(P, 1), 1);
        end
    end
end
