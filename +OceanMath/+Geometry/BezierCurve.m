classdef BezierCurve < OceanMath.Geometry.IParametricCurve
    % BEZIERCURVE  Bezier curve of arbitrary degree via Bernstein basis.
    %
    % A Bezier curve of degree n is defined by (n+1) control points:
    %
    %   C(t) = sum_{i=0}^{n} B_{i,n}(t) * P_i,   t in [0,1]
    %
    % where B_{i,n} are the Bernstein polynomials.
    %
    % Properties:
    %   - Passes through first and last control point only
    %   - Tangent at t=0 is proportional to (P1 - P0)
    %   - Tangent at t=1 is proportional to (Pn - Pn-1)
    %   - Degree = number of control points - 1
    %
    % Usage:
    %   pts = [0 0; 1 2; 3 1; 5 0];
    %   curve = OceanMath.Geometry.BezierCurve(pts);
    %   P = curve.evaluate(linspace(0,1,200)');
    %   T = curve.tangent(0.5);
    %
    % See also: OceanMath.Geometry.IParametricCurve,
    %           OceanMath.Geometry.BernsteinBasis

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        Degree_  (1,1) double  % Polynomial degree (= K-1)
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = BezierCurve(controlPoints)
            % BEZIERCURVE  Construct a Bezier curve from control points.
            %
            %   curve = BezierCurve(pts)
            %
            %   Input: pts - [K x d] control points, K >= 2, d in {2, 3}
            arguments
                controlPoints  (:,:) double
            end

            K = size(controlPoints, 1);
            d = size(controlPoints, 2);

            if K < 2
                error('OceanMath:Geometry:BezierCurve:TooFewPoints', ...
                    'At least 2 control points required. Got %d.', K);
            end
            if d < 2 || d > 3
                error('OceanMath:Geometry:BezierCurve:BadDimension', ...
                    'Control points must be 2D or 3D. Got d = %d.', d);
            end

            obj.ControlPoints_ = controlPoints;
            obj.Dimension_ = d;
            obj.Degree_ = K - 1;
        end

        % -------------------------------------------------------------- %
        function P = evaluate(obj, t)
            % EVALUATE  Map parameter t in [0,1] to [N x d] points.
            %
            %   Uses de Casteljau's algorithm for numerical stability.
            arguments
                obj
                t (:,1) double
            end

            P = obj.deCasteljau_(t, 0);
        end

        % -------------------------------------------------------------- %
        function T = tangent(obj, t)
            % TANGENT  First derivative dC/dt at parameter values t.
            %
            %   The derivative of a degree-n Bezier is a degree-(n-1) Bezier
            %   with control points Q_i = n * (P_{i+1} - P_i).
            arguments
                obj
                t (:,1) double
            end

            T = obj.deCasteljau_(t, 1);
        end

        % -------------------------------------------------------------- %
        function D2 = secondDerivative(obj, t)
            % SECONDDERIVATIVE  Second derivative d2C/dt2.
            %
            %   The second derivative is a degree-(n-2) Bezier with
            %   control points: n*(n-1)*(P_{i+2} - 2*P_{i+1} + P_i)
            arguments
                obj
                t (:,1) double
            end

            D2 = obj.deCasteljau_(t, 2);
        end

        % -------------------------------------------------------------- %
        function n = getNumSegments(~)
            % GETNUMSEGMENTS  Bezier is a single segment.
            n = 1;
        end

        function setControlPoint(obj, idx, newPos)
            % SETCONTROLPOINT  Move a control point.
            arguments
                obj
                idx    (1,1) double {mustBePositive, mustBeInteger}
                newPos (1,:) double
            end
            K = size(obj.ControlPoints_, 1);
            if idx > K
                error('OceanMath:Geometry:BezierCurve:IndexOutOfRange', ...
                    'Index %d exceeds number of control points (%d).', idx, K);
            end
            if numel(newPos) ~= obj.Dimension_
                error('OceanMath:Geometry:BezierCurve:DimMismatch', ...
                    'New position must have %d elements.', obj.Dimension_);
            end
            obj.ControlPoints_(idx, :) = newPos;
        end

        function desc = describe(obj)
            % DESCRIBE  Print a human-readable summary.
            K = size(obj.ControlPoints_, 1);
            L = obj.totalLength();
            desc = sprintf(['BezierCurve:\n' ...
                '  Dimension:      %dD\n' ...
                '  Control Points: %d\n' ...
                '  Degree:         %d\n' ...
                '  Total Length:   %.4f\n'], ...
                obj.Dimension_, K, obj.Degree_, L);
            fprintf('%s', desc);
        end
    end

    % ------------------------------------------------------------------ %
    methods (Access = private)

        function result = deCasteljau_(obj, t, derivOrder)
            % DECASTELJAU_  Unified evaluation via de Casteljau's algorithm.
            %
            % For position (derivOrder=0): standard de Casteljau on P.
            % For tangent (derivOrder=1): de Casteljau on derivative control
            %   points Q_i = n*(P_{i+1} - P_i), which form a degree-(n-1) Bezier.
            % For second derivative (derivOrder=2): de Casteljau on
            %   R_i = n*(n-1)*(P_{i+2} - 2*P_{i+1} + P_i), degree-(n-2) Bezier.

            pts = obj.ControlPoints_;
            n = obj.Degree_;

            % Build derivative control points if needed
            if derivOrder >= 1
                % First difference: Q_i = n * (P_{i+1} - P_i)
                pts = n * diff(pts, 1);
                if derivOrder >= 2 && size(pts, 1) >= 2
                    % Second difference: R_i = (n-1) * (Q_{i+1} - Q_i)
                    pts = (n - 1) * diff(pts, 1);
                end
            end

            K = size(pts, 1);
            if K == 0
                result = zeros(numel(t), obj.Dimension_);
                return;
            end

            N = numel(t);
            d = obj.Dimension_;
            result = zeros(N, d);

            % Vectorized Bernstein polynomial evaluation
            nPoly = K - 1;
            for i = 0:nPoly
                % B = nchoosek(nPoly, i) * t^i * (1-t)^(nPoly-i)
                B = nchoosek(nPoly, i) .* (t.^i) .* ((1 - t).^(nPoly - i)); % [N x 1]
                result = result + B * pts(i+1, :); % [N x 1] * [1 x d] -> [N x d]
            end
        end
    end
end
