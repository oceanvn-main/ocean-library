classdef ProductBasis < OceanStats.Base.IBasisFunction
    % PRODUCTBASIS Represents the product of two MARS basis functions.
    % Usually forms an interaction term between a Parent basis (h_l(X))
    % and a new HingeBasis ((X_j - t)_+).
    
    properties
        ParentBasis
        Hinge
    end
    
    methods
        function obj = ProductBasis(parent, hinge)
            obj.ParentBasis = parent;
            obj.Hinge = hinge;
        end
        
        function val = evaluate(obj, X)
            parentVal = obj.ParentBasis.evaluate(X);
            hingeVal = obj.Hinge.evaluate(X);
            val = parentVal .* hingeVal;
        end
        
        function str = toString(obj, featNames)
            if nargin > 1
                parentStr = obj.ParentBasis.toString(featNames);
                hingeStr = obj.Hinge.toString(featNames);
            else
                parentStr = obj.ParentBasis.toString();
                hingeStr = obj.Hinge.toString();
            end
            % Avoid putting "1 * " if the parent is the constant basis
            if parentStr == "1"
                str = hingeStr;
            else
                str = parentStr + " * " + hingeStr;
            end
        end
        
        function deg = getDegree(obj)
            deg = obj.ParentBasis.getDegree() + 1;
        end
    end
end
