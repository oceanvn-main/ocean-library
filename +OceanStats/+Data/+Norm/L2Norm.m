classdef L2Norm < OceanStats.Data.Norm.INormScheme
    % L2NORM  L2 row-wise normalization strategy.
    
    methods
        function W = normalize(~, W)
            nrm = sqrt(sum(W.^2, 2));
            nrm(nrm == 0) = 1;
            W = W ./ nrm;
        end
        
        function n = name(~)
            n = "l2";
        end
        
        function str = symbolicLatex(~)
            str = "\hat{w}_{d,t} = w_{d,t} / \lVert w_{d,\cdot} \rVert_2";
        end
    end
end
