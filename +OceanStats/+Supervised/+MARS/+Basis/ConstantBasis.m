classdef ConstantBasis < OceanStats.Base.IBasisFunction
    % CONSTANTBASIS Constant basis function for MARS (h_0(X) = 1).
    
    methods
        function val = evaluate(~, X)
            % Evaluate simply returns 1 for all instances.
            N = size(X, 1);
            val = ones(N, 1);
        end
        
        function str = toString(~, ~)
            % String representation
            str = "1";
        end
        
        function deg = getDegree(~)
            deg = 0;
        end
    end
end
