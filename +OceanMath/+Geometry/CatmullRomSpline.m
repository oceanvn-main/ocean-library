classdef CatmullRomSpline < OceanMath.Geometry.IParametricCurve
    % CATMULLROMSPLINE  Centripetal Catmull-Rom spline through N control points.
    %
    % A Catmull-Rom spline is a piecewise cubic curve that passes exactly
    % through every control point (interpolating spline). The "centripetal"
    % variant (alpha = 0.5) prevents cusps and self-intersections that
    % plague the uniform (alpha = 0) and chordal (alpha = 1) variants.
    %
    % Mathematics:
    %   Given 4 consecutive control points P0, P1, P2, P3 and their
    %   knot parameters t0, t1, t2, t3 (derived from inter-point distances
    %   raised to the power alpha):
    %
    %     t_{i+1} = t_i + ||P_{i+1} - P_i||^alpha
    %
    %   The segment between P1 and P2 is evaluated via Barry-Goldman
    %   recursive linear interpolation:
    %
    %     A1 = (t1-t)/(t1-t0)*P0 + (t-t0)/(t1-t0)*P1
    %     A2 = (t2-t)/(t2-t1)*P1 + (t-t1)/(t2-t1)*P2
    %     A3 = (t3-t)/(t3-t2)*P2 + (t-t2)/(t3-t2)*P3
    %     B1 = (t2-t)/(t2-t0)*A1 + (t-t0)/(t2-t0)*A2
    %     B2 = (t3-t)/(t3-t1)*A2 + (t-t1)/(t3-t1)*A3
    %     C  = (t2-t)/(t2-t1)*B1 + (t-t1)/(t2-t1)*B2
    %
    % Usage:
    %   pts = [0 0; 1 2; 3 1; 5 3; 7 0];
    %   spline = OceanMath.Geometry.CatmullRomSpline(pts);
    %   P = spline.evaluate(linspace(0,1,200)');
    %   plot(P(:,1), P(:,2), '-');
    %
    % See also: OceanMath.Geometry.ISpline, OceanMath.Geometry.FrenetFrame

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        % Alpha_ : parameterization exponent.
        %   0   = uniform  (can produce cusps)
        %   0.5 = centripetal (recommended, no cusps)
        %   1   = chordal  (can overshoot)
        Alpha_  (1,1) double = 0.5

        % KnotParams_ : [K x 1] cumulative knot parameter values
        %   derived from inter-point distances raised to Alpha_.
        KnotParams_  double

        % SegmentMap_ : maps global t in [0,1] to (segmentIndex, localT)
        %   SegmentMap_(i,:) = [tStart, tEnd] for segment i
        SegmentMap_  double

        % Padded_ : [K+2 x d] control points with phantom endpoints
        %   added to allow the first and last segments to be computed.
        Padded_  double
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = CatmullRomSpline(controlPoints, options)
            % CATMULLROMSPLINE  Construct a centripetal Catmull-Rom spline.
            %
            %   spline = CatmullRomSpline(pts)
            %   spline = CatmullRomSpline(pts, 'Alpha', 0.5)
            %
            %   Inputs:
            %     controlPoints - [K x d] matrix, K >= 2, d in {2, 3}
            %
            %   Options:
            %     Alpha - parameterization exponent (default: 0.5)
            arguments
                controlPoints  (:,:) double
                options.Alpha  (1,1) double {mustBeInRange(options.Alpha, 0, 1)} = 0.5
            end

            K = size(controlPoints, 1);
            d = size(controlPoints, 2);

            if K < 2
                error('OceanMath:Geometry:CatmullRomSpline:TooFewPoints', ...
                    'At least 2 control points are required. Got %d.', K);
            end
            if d < 2 || d > 3
                error('OceanMath:Geometry:CatmullRomSpline:BadDimension', ...
                    'Control points must be 2D or 3D. Got d = %d.', d);
            end

            obj.ControlPoints_ = controlPoints;
            obj.Dimension_ = d;
            obj.Alpha_ = options.Alpha;

            obj.rebuild_();
        end
        % PHYSICS ENGINE QUERY
        % -------------------------------------------------------------- %
        function P = evaluate(obj, t)
            % EVALUATE  Map parameter t in [0,1] to [N x d] spatial coordinates.
            arguments
                obj
                t (:,1) double
            end
            P = obj.evaluateDerivative_(t, 0);
        end

        % -------------------------------------------------------------- %
        function T = tangent(obj, t)
            % TANGENT  First derivative dP/dt at parameter values t.
            arguments
                obj
                t (:,1) double
            end
            T = obj.evaluateDerivative_(t, 1);
        end

        % -------------------------------------------------------------- %
        function D2 = secondDerivative(obj, t)
            % SECONDDERIVATIVE  Second derivative d2P/dt2 at parameter values t.
            arguments
                obj
                t (:,1) double
            end
            D2 = obj.evaluateDerivative_(t, 2);
        end

        % -------------------------------------------------------------- %
        function n = getNumSegments(obj)
            % GETNUMSEGMENTS  Number of cubic segments.
            n = size(obj.SegmentMap_, 1);
        end

        function setControlPoint(obj, idx, newPos)
            % SETCONTROLPOINT  Move a control point and recompute the spline.
            %
            %   obj.setControlPoint(3, [4, 2])
            %
            %   This is the method called when a user drags a knot in the GUI.
            arguments
                obj
                idx     (1,1) double {mustBePositive, mustBeInteger}
                newPos  (1,:) double
            end

            K = size(obj.ControlPoints_, 1);
            if idx > K
                error('OceanMath:Geometry:CatmullRomSpline:IndexOutOfRange', ...
                    'Index %d exceeds number of control points (%d).', idx, K);
            end
            if numel(newPos) ~= obj.Dimension_
                error('OceanMath:Geometry:CatmullRomSpline:DimMismatch', ...
                    'New position must have %d elements.', obj.Dimension_);
            end

            % Update the control point
            obj.ControlPoints_(idx, :) = newPos;

            obj.rebuild_();
        end

        function addControlPoint(obj, newPos)
            % ADDCONTROLPOINT  Append a new control point at the end.
            %
            %   obj.addControlPoint([6, 3])
            arguments
                obj
                newPos  (1,:) double
            end
            if numel(newPos) ~= obj.Dimension_
                error('OceanMath:Geometry:CatmullRomSpline:DimMismatch', ...
                    'New position must have %d elements.', obj.Dimension_);
            end
            obj.ControlPoints_ = [obj.ControlPoints_; newPos];

            obj.rebuild_();
        end

        function removeControlPoint(obj, idx)
            % REMOVECONTROLPOINT  Remove a control point by index.
            arguments
                obj
                idx  (1,1) double {mustBePositive, mustBeInteger}
            end
            K = size(obj.ControlPoints_, 1);
            if K <= 2
                error('OceanMath:Geometry:CatmullRomSpline:TooFewPoints', ...
                    'Cannot remove: minimum 2 control points required.');
            end
            if idx > K
                error('OceanMath:Geometry:CatmullRomSpline:IndexOutOfRange', ...
                    'Index %d exceeds number of control points (%d).', idx, K);
            end
            obj.ControlPoints_(idx, :) = [];
            K = K - 1;

            obj.rebuild_();
        end

        function desc = describe(obj)
            % DESCRIBE  Print a human-readable summary.
            K = size(obj.ControlPoints_, 1);
            L = obj.totalLength();
            desc = sprintf(['CatmullRomSpline:\n' ...
                '  Dimension:      %dD\n' ...
                '  Control Points: %d\n' ...
                '  Segments:       %d\n' ...
                '  Alpha:          %.2f (centripetal)\n' ...
                '  Total Length:   %.4f\n'], ...
                obj.Dimension_, K, obj.getNumSegments(), obj.Alpha_, L);
            fprintf('%s', desc);
        end
    end

    % ------------------------------------------------------------------ %
    methods (Access = private)
        function rebuild_(obj)
            % REBUILD_  Recompute phantom endpoints, knot parameters, and segment map.
            %
            % Called whenever ControlPoints_ change (constructor, set, add, remove).
            % This is the single source of truth for the spline's internal state.
            cp = obj.ControlPoints_;
            K  = size(cp, 1);

            % --- 1. Phantom endpoints (momentum-preserving reflection) ---
            % Suppose we take a step backward at P1: pStart = P1 + (P1 - P2).
            % Rearrange: pStart = 2*P1 - P2.  Same idea for pEnd.
            pStart = 2 * cp(1,:) - cp(2,:);
            pEnd   = 2 * cp(K,:) - cp(K-1,:);
            obj.Padded_ = [pStart; cp; pEnd];

            % --- 2. Knot parameters (centripetal parameterization) ---
            % Recompute because knot values depend on physical distances.
            obj.KnotParams_ = obj.computeKnotParams_(obj.Padded_, obj.Alpha_);

            % --- 3. Segment map ---
            % Each segment i uses padded points [i, i+1, i+2, i+3]
            % and interpolates between points i+1 and i+2 (the t range).
            nSeg = K - 1;
            kp   = obj.KnotParams_;
            obj.SegmentMap_ = zeros(nSeg, 2);
            for s = 1:nSeg
                obj.SegmentMap_(s, :) = [kp(s+1), kp(s+2)];
            end
        end

        function result = evaluateDerivative_(obj, t, derivOrder)
            % EVALUATEDERIVATIVE_  Unified evaluation of position / 1st / 2nd derivative.
            %
            % Uses the Barry-Goldman algorithm for position, and analytical
            % differentiation of the same recursive formula for derivatives.
            %
            % The chain rule is applied carefully because global t in [0,1]
            % must be mapped to local knot-parameter space per segment.
            %
            % derivOrder: 0 = position, 1 = tangent, 2 = second derivative.

            N = numel(t);
            d = obj.Dimension_;
            result = zeros(N, d);

            % Total knot-parameter range
            kp = obj.KnotParams_;
            tMin = kp(2);           % First actual control point
            tMax = kp(end - 1);     % Last actual control point
            tRange = tMax - tMin;

            nSeg  = obj.getNumSegments();
            pts   = obj.Padded_;

            for i = 1:N
                % --- Map global t in [0,1] to knot-parameter space ---
                %Fraction to physical distance
                tGlobal = tMin + t(i) * tRange;
                tGlobal = max(tMin, min(tMax, tGlobal));  % clamp

                % --- Find which segment this t belongs to --- 
                segIdx = 1;
                for s = 1:nSeg
                    if tGlobal >= obj.SegmentMap_(s,1) && ...
                       (tGlobal < obj.SegmentMap_(s,2) || s == nSeg)
                        segIdx = s;
                        break;
                    end
                end

                % --- Extract the 4 control points and their knot params ---
                P0 = pts(segIdx,     :);
                P1 = pts(segIdx + 1, :);
                P2 = pts(segIdx + 2, :);
                P3 = pts(segIdx + 3, :);

                %Time interval of this segment
                t0 = kp(segIdx);
                t1 = kp(segIdx + 1);
                t2 = kp(segIdx + 2);
                t3 = kp(segIdx + 3);

                u = tGlobal;  % local knot-parameter value

                raw = obj.barryGoldmanEval_(u, P0,P1,P2,P3, t0,t1,t2,t3, derivOrder);
                result(i,:) = raw * tRange^derivOrder;
            end
        end
    end

    % ------------------------------------------------------------------ %
    methods (Static, Access = private)
        function kp = computeKnotParams_(points, alpha)
            % COMPUTEKNOTPARAMS_  Centripetal knot parameterization.
            %   t_{i+1} = t_i + ||P_{i+1} - P_i||^alpha
            nPts = size(points, 1);
            kp = zeros(nPts, 1);
            for i = 2:nPts
                dist = sqrt(sum((points(i,:) - points(i-1,:)).^2));
                kp(i) = kp(i-1) + dist^alpha;
            end
        end

        function out = barryGoldmanEval_(u, P0,P1,P2,P3, t0,t1,t2,t3, derivOrder)
            % BARRYGOLDMANEVAL_  Unified Barry-Goldman evaluation.
            %
            %   derivOrder 0 = position (C)
            %   derivOrder 1 = first derivative (dC/du)
            %   derivOrder 2 = second derivative (d2C/du2)
            %
            % Each level is a lerp:  f(u) = ((b-u)/(b-a))*X + ((u-a)/(b-a))*Y
            % with derivative:      f'(u) = (Y - X) / (b - a)

            % --- Level 1: A values (shared by all orders) ---
            A1 = ((t1 - u)/(t1 - t0)) * P0 + ((u - t0)/(t1 - t0)) * P1;
            A2 = ((t2 - u)/(t2 - t1)) * P1 + ((u - t1)/(t2 - t1)) * P2;
            A3 = ((t3 - u)/(t3 - t2)) * P2 + ((u - t2)/(t3 - t2)) * P3;

            % --- Level 2: B values (shared by all orders) ---
            B1 = ((t2 - u)/(t2 - t0)) * A1 + ((u - t0)/(t2 - t0)) * A2;
            B2 = ((t3 - u)/(t3 - t1)) * A2 + ((u - t1)/(t3 - t1)) * A3;

            % --- Level 3: Position ---
            if derivOrder == 0
                out = ((t2 - u)/(t2 - t1)) * B1 + ((u - t1)/(t2 - t1)) * B2;
                return;
            end

            % --- Level 1 derivatives (needed for 1st and 2nd order) ---
            dA1 = (P1 - P0) / (t1 - t0);
            dA2 = (P2 - P1) / (t2 - t1);
            dA3 = (P3 - P2) / (t3 - t2);

            % --- Level 2 derivatives (product rule) ---
            dB1 = (-1/(t2 - t0)) * A1 + ((t2 - u)/(t2 - t0)) * dA1 ...
                 + ( 1/(t2 - t0)) * A2 + ((u  - t0)/(t2 - t0)) * dA2;
            dB2 = (-1/(t3 - t1)) * A2 + ((t3 - u)/(t3 - t1)) * dA2 ...
                 + ( 1/(t3 - t1)) * A3 + ((u  - t1)/(t3 - t1)) * dA3;

            % --- Level 3: First derivative ---
            if derivOrder == 1
                out = (-1/(t2 - t1)) * B1 + ((t2 - u)/(t2 - t1)) * dB1 ...
                    + ( 1/(t2 - t1)) * B2 + ((u  - t1)/(t2 - t1)) * dB2;
                return;
            end

            % --- Level 2 second derivatives ---
            % d2A = 0 (A's are linear in u), so only cross-terms survive.
            d2B1 = 2 * (-1/(t2 - t0)) * dA1 + 2 * (1/(t2 - t0)) * dA2;
            d2B2 = 2 * (-1/(t3 - t1)) * dA2 + 2 * (1/(t3 - t1)) * dA3;

            % --- Level 3: Second derivative ---
            %   -2/(t2-t1)*dB1 + (t2-u)/(t2-t1)*d2B1
            %   +2/(t2-t1)*dB2 + (u-t1)/(t2-t1)*d2B2
            out = -2/(t2 - t1) * dB1 + ((t2 - u)/(t2 - t1)) * d2B1 ...
                +  2/(t2 - t1) * dB2 + ((u  - t1)/(t2 - t1)) * d2B2;
        end
    end
end
