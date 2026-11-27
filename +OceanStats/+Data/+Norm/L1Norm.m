classdef L1Norm < OceanStats.Data.Norm.INormScheme
    % L1NORM  L1 row-wise normalization strategy.
    
    methods
        function W = normalize(~, W)
            nrm = sum(abs(W), 2);
            nrm(nrm == 0) = 1;
            W = W ./ nrm;
        end
        
        function n = name(~)
            n = "l1";
        end
        
        function str = symbolicLatex(~)
            str = "\hat{w}_{d,t} = w_{d,t} / \lVert w_{d,\cdot} \rVert_1";
        end
    end
end
