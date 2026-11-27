classdef NURBSCurve < OceanMath.Geometry.IParametricCurve
    % NURBSCURVE  Non-Uniform Rational B-Spline curve.
    %
    % NURBS = B-spline + per-control-point weights. The rational form:
    %
    %   C(t) = sum_i  w_i * N_{i,p}(t) * P_i
    %          ---------------------------------
    %          sum_i  w_i * N_{i,p}(t)
    %
    % Advantages over standard B-splines:
    %   - Can represent exact conic sections (circles, ellipses, parabolas)
    %   - Invariant under projective transformations
    %   - Reduces to B-spline when all weights = 1
    %
    % Usage:
    %   % Quarter circle (exact, not approximate)
    %   pts = [1 0; 1 1; 0 1];
    %   w   = [1; 1/sqrt(2); 1];
    %   curve = OceanMath.Geometry.NURBSCurve(pts, w, 'Degree', 2);
    %   P = curve.evaluate(linspace(0,1,200)');
    %
    % See also: OceanMath.Geometry.BSplineCurve,
    %           OceanMath.Geometry.BSplineBasis,
    %           OceanMath.Geometry.KnotVector

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        Degree_   (1,1) double   % Polynomial degree p
        Knots_    (1,:) double   % Knot vector [1 x (n + p + 1)]
        Weights_  (:,1) double   % [K x 1] weights per control point
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = NURBSCurve(controlPoints, weights, options)
            % NURBSCURVE  Construct a NURBS curve.
            %
            %   curve = NURBSCurve(pts, weights)
            %   curve = NURBSCurve(pts, weights, 'Degree', 2)
            %   curve = NURBSCurve(pts, weights, 'Degree', 3, 'Knots', k)
            %
            %   Inputs:
            %     controlPoints - [K x d] matrix, K >= 2, d in {2, 3}
            %     weights       - [K x 1] positive weights
            %
            %   Options:
            %     Degree - polynomial degree (default: 3, capped at K-1)
            %     Knots  - custom knot vector (default: clamped uniform)
            arguments
                controlPoints  (:,:) double
                weights        (:,1) double
                options.Degree (1,1) double {mustBeNonnegative, mustBeInteger} = 3
                options.Knots  (1,:) double = []
            end

            K = size(controlPoints, 1);
            d = size(controlPoints, 2);

            if K < 2
                error('OceanMath:Geometry:NURBSCurve:TooFewPoints', ...
                    'At least 2 control points required. Got %d.', K);
            end
            if d < 2 || d > 3
                error('OceanMath:Geometry:NURBSCurve:BadDimension', ...
                    'Control points must be 2D or 3D. Got d = %d.', d);
            end
            if numel(weights) ~= K
                error('OceanMath:Geometry:NURBSCurve:WeightMismatch', ...
                    'Number of weights (%d) must equal control points (%d).', ...
                    numel(weights), K);
            end
            if any(weights <= 0)
                error('OceanMath:Geometry:NURBSCurve:NonPositiveWeight', ...
                    'All weights must be positive.');
            end

            p = min(options.Degree, K - 1);

            obj.ControlPoints_ = controlPoints;
            obj.Dimension_ = d;
            obj.Degree_ = p;
            obj.Weights_ = weights;

            if isempty(options.Knots)
                obj.Knots_ = OceanMath.Geometry.KnotVector.clamped(K, p);
            else
                if ~OceanMath.Geometry.KnotVector.validate(options.Knots, K, p)
                    error('OceanMath:Geometry:NURBSCurve:BadKnots', ...
                        'Knot vector is invalid for n=%d, p=%d.', K, p);
                end
                obj.Knots_ = options.Knots;
            end
        end

        % -------------------------------------------------------------- %
        function P = evaluate(obj, t)
            % EVALUATE  Rational B-spline evaluation.
            %
            %   C(t) = sum(w_i * N_i(t) * P_i) / sum(w_i * N_i(t))
            %
            %   This is equivalent to projecting a (d+1)-dimensional
            %   B-spline back into d dimensions.
            arguments
                obj
                t (:,1) double
            end

            B = OceanMath.Geometry.BSplineBasis.evaluate(obj.Knots_, obj.Degree_, t);
            % [N x K] basis, [K x 1] weights -> [N x K] weighted basis
            Bw = B .* obj.Weights_';      % broadcast: each column scaled by w_i
            denom = sum(Bw, 2);            % [N x 1] sum of weighted basis

            % Avoid division by zero
            denom(denom < 1e-15) = 1e-15;

            % Numerator: sum_i w_i * N_i * P_i = Bw * P
            P = (Bw * obj.ControlPoints_) ./ denom;
        end

        % -------------------------------------------------------------- %
        function T = tangent(obj, t)
            % TANGENT  First derivative of rational B-spline.
            %
            %   Uses the quotient rule on C(t) = A(t) / W(t):
            %     C'(t) = (A'(t) * W(t) - A(t) * W'(t)) / W(t)^2
            %
            %   where A(t) = sum(w_i * N_i * P_i), W(t) = sum(w_i * N_i)
            arguments
                obj
                t (:,1) double
            end

            B  = OceanMath.Geometry.BSplineBasis.evaluate(obj.Knots_, obj.Degree_, t);
            dB = OceanMath.Geometry.BSplineBasis.derivative(obj.Knots_, obj.Degree_, t);

            w = obj.Weights_';
            P = obj.ControlPoints_;

            % Weighted basis and derivatives
            Bw  = B  .* w;        % [N x K]
            dBw = dB .* w;        % [N x K]

            % W(t) and W'(t)
            W  = sum(Bw, 2);      % [N x 1]
            dW = sum(dBw, 2);     % [N x 1]

            % A(t) = Bw * P,  A'(t) = dBw * P
            A  = Bw  * P;         % [N x d]
            dA = dBw * P;         % [N x d]

            % Quotient rule: C' = (A'W - AW') / W^2
            W(abs(W) < 1e-15) = 1e-15;
            T = (dA .* W - A .* dW) ./ (W.^2);
        end

        % -------------------------------------------------------------- %
        function D2 = secondDerivative(obj, t)
            % SECONDDERIVATIVE  Second derivative via central FD on tangent.
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
            n = sum(diff(obj.Knots_) > 0);
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
                error('OceanMath:Geometry:NURBSCurve:IndexOutOfRange', ...
                    'Index %d exceeds number of control points (%d).', idx, K);
            end
            if numel(newPos) ~= obj.Dimension_
                error('OceanMath:Geometry:NURBSCurve:DimMismatch', ...
                    'New position must have %d elements.', obj.Dimension_);
            end
            obj.ControlPoints_(idx, :) = newPos;
        end

        function setWeight(obj, idx, w)
            % SETWEIGHT  Change the weight of a control point.
            %
            %   Higher weight pulls the curve closer to that control point.
            %   Weight = 1 is neutral. Weight > 1 attracts, weight < 1 repels.
            arguments
                obj
                idx (1,1) double {mustBePositive, mustBeInteger}
                w   (1,1) double {mustBePositive}
            end
            K = size(obj.ControlPoints_, 1);
            if idx > K
                error('OceanMath:Geometry:NURBSCurve:IndexOutOfRange', ...
                    'Index %d exceeds number of control points (%d).', idx, K);
            end
            obj.Weights_(idx) = w;
        end

        function desc = describe(obj)
            % DESCRIBE  Print a human-readable summary.
            K = size(obj.ControlPoints_, 1);
            L = obj.totalLength();
            desc = sprintf(['NURBSCurve:\n' ...
                '  Dimension:      %dD\n' ...
                '  Control Points: %d\n' ...
                '  Degree:         %d\n' ...
                '  Knot spans:     %d\n' ...
                '  Weights:        [%s]\n' ...
                '  Total Length:   %.4f\n'], ...
                obj.Dimension_, K, obj.Degree_, obj.getNumSegments(), ...
                strjoin(arrayfun(@(x) sprintf('%.3f', x), obj.Weights_, 'UniformOutput', false), ', '), L);
            fprintf('%s', desc);
        end
    end
end
