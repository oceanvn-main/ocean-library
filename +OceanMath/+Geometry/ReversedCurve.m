classdef ReversedCurve < OceanMath.Geometry.IParametricCurve
    % REVERSEDCURVE  An adapter that flips the parameterization of a curve.
    %
    % Wraps an existing IParametricCurve so that t=0 evaluates at the original t=1,
    % and t=1 evaluates at the original t=0. This solves routing directed 
    % agent traversal over generically drawn undirected splines.
    
    properties
        BaseCurve_
    end
    
    methods
        function obj = ReversedCurve(baseCurve)
            obj.BaseCurve_ = baseCurve;
            obj.Dimension_ = baseCurve.Dimension_;
            
            % If ControlPoints_ are publicly visible or protected, we can mirror them
            % Assuming IParametricCurve defines them
            if isprop(baseCurve, 'ControlPoints_')
                obj.ControlPoints_ = flipud(baseCurve.ControlPoints_);
            end
        end
        
        function P = evaluate(obj, t)
            P = obj.BaseCurve_.evaluate(1 - t);
        end
        
        function T = tangent(obj, t)
            % Chain rule: d/dt P(1 - t) = P'(1 - t) * (-1)
            T = -obj.BaseCurve_.tangent(1 - t);
        end
        
        function D2 = secondDerivative(obj, t)
            % Chain rule twice: d2/dt2 P(1 - t) = P''(1 - t) * (-1)^2
            D2 = obj.BaseCurve_.secondDerivative(1 - t);
        end
        
        function t = getParameterAtArcLength(obj, s, ~)
            % Distance from the current "start" (t=0) is s.
            % In the base curve, this is distance (L - s) from the base start.
            L = obj.BaseCurve_.totalLength();
            tBase = obj.BaseCurve_.getParameterAtArcLength(L - s);
            t = 1 - tBase;
        end
        
        function L = arcLength(obj, t0, t1)
            % Range [t0, t1] in the reversed curve corresponds to [1-t1, 1-t0] in the base curve
            L = obj.BaseCurve_.arcLength(1 - t1, 1 - t0);
        end
        
        function L = totalLength(obj)
            L = obj.BaseCurve_.totalLength();
        end
    end
end
