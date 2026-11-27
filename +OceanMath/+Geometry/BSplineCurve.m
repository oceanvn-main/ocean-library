classdef BSplineCurve < OceanMath.Geometry.IParametricCurve
    % BSPLINECURVE  B-spline curve of arbitrary degree via BSplineBasis.
    %
    % A B-spline curve of degree p with n control points and knot vector k:
    %
    %   C(t) = sum_{i=1}^{n} N_{i,p}(t) * P_i
    %
    % where N_{i,p} are the B-spline basis functions evaluated by BSplineBasis.
    %
    % Properties:
    %   - Local control: moving one control point affects only nearby segments
    %   - Degree is independent of number of control points
    %   - Clamped knots: curve starts at P_1 and ends at P_n
    %   - Approximating: curve does NOT pass through interior control points
    %
    % Usage:
    %   pts = [0 0; 1 3; 3 4; 5 2; 7 0; 9 1];
    %   curve = OceanMath.Geometry.BSplineCurve(pts, 'Degree', 3);
    %   P = curve.evaluate(linspace(0,1,200)');
    %   T = curve.tangent(0.5);
    %
    % See also: OceanMath.Geometry.IParametricCurve,
    %           OceanMath.Geometry.BSplineBasis,
    %           OceanMath.Geometry.KnotVector

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        Degree_  (1,1) double   % Polynomial degree p
        Knots_   (1,:) double   % Knot vector [1 x (n + p + 1)]
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = BSplineCurve(controlPoints, options)
            % BSPLINECURVE  Construct a B-spline curve.
            %
            %   curve = BSplineCurve(pts)
            %   curve = BSplineCurve(pts, 'Degree', 3)
            %   curve = BSplineCurve(pts, 'Degree', 3, 'Knots', myKnots)
            %
            %   Inputs:
            %     controlPoints - [K x d] matrix, K >= 2, d in {2, 3}
            %
            %   Options:
            %     Degree - polynomial degree (default: 3, capped at K-1)
            %     Knots  - custom knot vector (default: clamped uniform)
            arguments
                controlPoints  (:,:) double
                options.Degree (1,1) double {mustBeNonnegative, mustBeInteger} = 3
                options.Knots  (1,:) double = []
            end

            K = size(controlPoints, 1);
            d = size(controlPoints, 2);

            if K < 2
                error('OceanMath:Geometry:BSplineCurve:TooFewPoints', ...
                    'At least 2 control points required. Got %d.', K);
            end
            if d < 2 || d > 3
                error('OceanMath:Geometry:BSplineCurve:BadDimension', ...
                    'Control points must be 2D or 3D. Got d = %d.', d);
            end

            % Cap degree at K-1 (cannot have degree >= number of points)
            p = min(options.Degree, K - 1);

            obj.ControlPoints_ = controlPoints;
            obj.Dimension_ = d;
            obj.Degree_ = p;

            % Build or validate knot vector
            if isempty(options.Knots)
                obj.Knots_ = OceanMath.Geometry.KnotVector.clamped(K, p);
            else
                if ~OceanMath.Geometry.KnotVector.validate(options.Knots, K, p)
                    error('OceanMath:Geometry:BSplineCurve:BadKnots', ...
                        'Knot vector is invalid for n=%d, p=%d. Expected length %d.', ...
                        K, p, K + p + 1);
                end
                obj.Knots_ = options.Knots;
            end
        end

        % -------------------------------------------------------------- %
        function P = evaluate(obj, t)
            % EVALUATE  Map parameter t in [0,1] to [N x d] points.
            %
            %   C(t) = sum_i N_{i,p}(t) * P_i = B * P
            %
            %   where B is [N x K] basis matrix, P is [K x d] control points.
            arguments
                obj
                t (:,1) double
            end

            B = OceanMath.Geometry.BSplineBasis.evaluate(obj.Knots_, obj.Degree_, t);
            P = B * obj.ControlPoints_;
        end

        % -------------------------------------------------------------- %
        function T = tangent(obj, t)
            % TANGENT  First derivative dC/dt at parameter values t.
            %
            %   dC/dt = sum_i dN_{i,p}/dt * P_i = dB * P
            arguments
                obj
                t (:,1) double
            end

            dB = OceanMath.Geometry.BSplineBasis.derivative(obj.Knots_, obj.Degree_, t);
            T = dB * obj.ControlPoints_;
        end

        % -------------------------------------------------------------- %
        function D2 = secondDerivative(obj, t)
            % SECONDDERIVATIVE  Second derivative d2C/dt2.
            %
            %   Computed via central finite difference on the first derivative.
            %   (Analytical second derivative of B-spline basis requires
            %   evaluating basis at degree p-2, which is handled here via FD
            %   for robustness.)
            arguments
                obj
                t (:,1) double
            end

            dt = 1e-6;
            N = numel(t);
            d = obj.Dimension_;
            D2 = zeros(N, d);

            for i = 1:N
                ti = t(i);
                t_plus  = min(ti + dt, 1);
                t_minus = max(ti - dt, 0);
                h = t_plus - t_minus;
                if h < 1e-15
                    continue;
                end
                T_plus  = obj.tangent(t_plus);
                T_minus = obj.tangent(t_minus);
                D2(i, :) = (T_plus - T_minus) / h;
            end
        end

        % -------------------------------------------------------------- %
        function n = getNumSegments(obj)
            % GETNUMSEGMENTS  Number of non-zero knot spans.
            k = obj.Knots_;
            n = sum(diff(k) > 0);
        end

        function setControlPoint(obj, idx, newPos)
            % SETCONTROLPOINT  Move a control point (local effect only).
            %
            %   Unlike Bezier, moving one control point only affects nearby
            %   segments (at most p+1 spans). This is the key advantage of
            %   B-splines for interactive editing.
            arguments
                obj
                idx    (1,1) double {mustBePositive, mustBeInteger}
                newPos (1,:) double
            end
            K = size(obj.ControlPoints_, 1);
            if idx > K
                error('OceanMath:Geometry:BSplineCurve:IndexOutOfRange', ...
                    'Index %d exceeds number of control points (%d).', idx, K);
            end
            if numel(newPos) ~= obj.Dimension_
                error('OceanMath:Geometry:BSplineCurve:DimMismatch', ...
                    'New position must have %d elements.', obj.Dimension_);
            end
            obj.ControlPoints_(idx, :) = newPos;
            % No rebuild needed — basis depends only on knots, not control points
        end

        function addControlPoint(obj, newPos, options)
            % ADDCONTROLPOINT  Append a control point and rebuild knots.
            %
            %   obj.addControlPoint([6, 3])
            %   obj.addControlPoint([6, 3], 'RebuildKnots', true)
            arguments
                obj
                newPos  (1,:) double
                options.RebuildKnots (1,1) logical = true
            end
            if numel(newPos) ~= obj.Dimension_
                error('OceanMath:Geometry:BSplineCurve:DimMismatch', ...
                    'New position must have %d elements.', obj.Dimension_);
            end
            obj.ControlPoints_ = [obj.ControlPoints_; newPos];
            if options.RebuildKnots
                K = size(obj.ControlPoints_, 1);
                obj.Knots_ = OceanMath.Geometry.KnotVector.clamped(K, obj.Degree_);
            end
        end

        function removeControlPoint(obj, idx, options)
            % REMOVECONTROLPOINT  Remove a control point by index.
            arguments
                obj
                idx  (1,1) double {mustBePositive, mustBeInteger}
                options.RebuildKnots (1,1) logical = true
            end
            K = size(obj.ControlPoints_, 1);
            if K <= obj.Degree_ + 1
                error('OceanMath:Geometry:BSplineCurve:TooFewPoints', ...
                    'Cannot remove: minimum %d control points required for degree %d.', ...
                    obj.Degree_ + 1, obj.Degree_);
            end
            if idx > K
                error('OceanMath:Geometry:BSplineCurve:IndexOutOfRange', ...
                    'Index %d exceeds number of control points (%d).', idx, K);
            end
            obj.ControlPoints_(idx, :) = [];
            if options.RebuildKnots
                K = size(obj.ControlPoints_, 1);
                obj.Knots_ = OceanMath.Geometry.KnotVector.clamped(K, obj.Degree_);
            end
        end

        function desc = describe(obj)
            % DESCRIBE  Print a human-readable summary.
            K = size(obj.ControlPoints_, 1);
            L = obj.totalLength();
            desc = sprintf(['BSplineCurve:\n' ...
                '  Dimension:      %dD\n' ...
                '  Control Points: %d\n' ...
                '  Degree:         %d\n' ...
                '  Knot spans:     %d\n' ...
                '  Total Length:   %.4f\n'], ...
                obj.Dimension_, K, obj.Degree_, obj.getNumSegments(), L);
            fprintf('%s', desc);
        end
    end
end
