classdef NoneNorm < OceanStats.Data.Norm.INormScheme
    % NONENORM  No-op normalization strategy.
    
    methods
        function W = normalize(~, W)
            % no-op
        end
        
        function n = name(~)
            n = "none";
        end
        
        function str = symbolicLatex(~)
            str = "\hat{w}_{d,t} = w_{d,t}";
        end
    end
end
