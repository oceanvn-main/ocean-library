classdef HingeBasis < OceanStats.Base.IBasisFunction
    % HINGEBASIS A linear spline function for MARS.
    % Representing (X_j - t)_+ if Direction=1
    % Representing (t - X_j)_+ if Direction=-1
    
    properties
        FeatureIndex (1,1) double
        Knot         (1,1) double
        Direction    (1,1) double {mustBeMember(Direction, [1, -1])} = 1
    end
    
    methods
        function obj = HingeBasis(featIdx, knot, dir)
            obj.FeatureIndex = featIdx;
            obj.Knot = knot;
            obj.Direction = dir;
        end
        
        function val = evaluate(obj, X)
            x_j = X(:, obj.FeatureIndex);
            if obj.Direction == 1
                val = max(x_j - obj.Knot, 0);
            else
                val = max(obj.Knot - x_j, 0);
            end
        end
        
        function str = toString(obj, featNames)
            if nargin > 1 && ~isempty(featNames)
                name = string(featNames{obj.FeatureIndex});
            else
                name = sprintf('X%d', obj.FeatureIndex);
            end
            
            knotStr = sprintf('%.4g', obj.Knot);
            if obj.Direction == 1
                str = "(" + name + " - " + knotStr + ")_+";
            else
                str = "(" + knotStr + " - " + name + ")_+";
            end
        end
        
        function deg = getDegree(~)
            deg = 1;
        end
    end
end
