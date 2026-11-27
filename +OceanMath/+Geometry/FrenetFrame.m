classdef FrenetFrame < handle
    % FRENETFRAME  Computes the Frenet-Serret frame along a parametric curve.
    %
    % Given any ISpline object, this class calculates the local coordinate
    % frame (Tangent, Normal, Binormal) at any point along the curve:
    %   T x  N = B, B x T = N, N x B = T (Right-handed coordinate system)
    %   T(t) = dP/dt / ||dP/dt||           (unit tangent — direction of travel) r'(t) / ||r'(t)|| (Textbook definition)
    %   B(t) = (dP/dt x d2P/dt2) / ||...|| (unit binormal — axis of local bend) ( T x N)
    %   N(t) = B(t) x T(t)                 (unit normal — points toward center of curvature). T' / ||T'|| (Textbook definition)
    %
    % The curvature kappa and torsion tau are also computable:
    %   kappa(t) = ||dP/dt x d2P/dt2|| / ||dP/dt||^3
    %   tau(t)   = dot(dP/dt x d2P/dt2, d3P/dt3) / ||dP/dt x d2P/dt2||^2
    %
    % Integration with OceanMath Quaternions:
    %   The TNB frame defines a rotation from the world frame to the local
    %   frame. This rotation is exported as a unit quaternion [w,x,y,z] via
    %   the existing OceanMath.Quat library, enabling seamless 3D orientation
    %   for rendering trains, rollercoasters, or cameras.
    %
    % Usage:
    %   spline = OceanMath.Geometry.CatmullRomSpline(pts);
    %   frenet = OceanMath.Geometry.FrenetFrame(spline);
    %   [T, N, B] = frenet.frame(0.5);
    %   q = frenet.quaternion(0.5);
    %   kappa = frenet.curvature(0.5);
    %
    % See also: OceanMath.Geometry.ISpline, OceanMath.Quat.create_from_euler

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        % Spline_ : the ISpline object to compute frames along
        Spline_
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = FrenetFrame(spline)
            % FRENETFRAME  Construct from an ISpline.
            %
            %   frenet = FrenetFrame(spline)
            arguments
                spline  OceanMath.Geometry.IParametricCurve
            end
            obj.Spline_ = spline;
        end

        % -------------------------------------------------------------- %
        function [T, N, B] = frame(obj, t)
            % FRAME  Compute the Frenet-Serret frame at parameter value(s) t.
            %
            %   [T, N, B] = frenet.frame(t)
            %
            %   Inputs:
            %     t - [M x 1] parameter values in [0, 1]
            %
            %   Outputs:
            %     T - [M x d] unit tangent vectors
            %     N - [M x d] unit normal vectors
            %     B - [M x d] unit binormal vectors (only meaningful for 3D)
            arguments
                obj
                t (:,1) double
            end

            d = obj.Spline_.getDimension();
            M = numel(t);

            % Raw derivatives from the parametric curve r(t)
            %   v1 = r'(t),  v2 = r''(t)
            dP  = obj.Spline_.tangent(t);            % [M x d]
            d2P = obj.Spline_.secondDerivative(t);   % [M x d]

            T = zeros(M, d);
            N = zeros(M, d);
            B = zeros(M, d);

            % ----- Construction order: T -> B -> N -----
            %
            % Textbook (Frenet-Serret formulas):
            %   T = r'/|r'|,  N = T'(s)/|T'(s)|,  B = T x N
            %
            % Computing N from the textbook requires dT/ds, which needs
            % the chain rule on a normalized vector (quotient rule). Messy.
            %
            % Shortcut (Frenet-Serret identity, Exercise 63b):
            %   r'(t) x r''(t) = kappa * |r'|^3 * B
            %
            % So r' x r'' is already parallel to B. Normalizing it gives
            % B directly, then N = B x T. No dT/ds needed.
            %
            % Result is mathematically identical to the textbook frame.

            for i = 1:M
                v1 = dP(i, :);   % r'(t)
                v2 = d2P(i, :);  % r''(t)

                % --- Step 1: Unit tangent T = r'(t) / |r'(t)| ---
                % Same as textbook definition.
                normV1 = sqrt(sum(v1.^2));
                if normV1 < 1e-12
                    % Degenerate: zero velocity (cusp or stationary point)
                    T(i, :) = zeros(1, d);
                    N(i, :) = zeros(1, d);
                    B(i, :) = zeros(1, d);
                    continue;
                end
                Ti = v1 / normV1;

                if d == 3
                    % --- Step 2: Binormal B via r' x r'' ---
                    %Calculus by Stewart, 8th edition, page 870, Exercise 63b
                    % From identity: r' x r'' = kappa * |r'|^3 * B
                    % So: B = (r' x r'') / |r' x r''|
                    crossVec = obj.cross3_(v1, v2);
                    normCross = sqrt(sum(crossVec.^2));

                    if normCross < 1e-12
                        % Straight line (kappa = 0): r' x r'' = 0
                        % The frame is undefined — pick an arbitrary
                        % perpendicular direction to keep the frame valid.
                        Bi = obj.arbitraryPerpendicular_(Ti);
                        Ni = obj.cross3_(Bi, Ti);
                        normNi = sqrt(sum(Ni.^2));
                        if normNi > 1e-12
                            Ni = Ni / normNi;
                        end
                    else
                        % Normal case: B from the cross product
                        Bi = crossVec / normCross;

                        % --- Step 3: Normal N = B x T ---
                        % Equivalent to textbook N = T'/|T'|, but computed
                        % as a simple cross product instead of dT/ds.
                        Ni = obj.cross3_(Bi, Ti);
                        normNi = sqrt(sum(Ni.^2));
                        if normNi > 1e-12
                            Ni = Ni / normNi;
                        end
                    end
                else
                    % --- 2D case ---
                    % Normal is 90-degree CCW rotation of tangent.
                    % Binormal has no geometric meaning in 2D.
                    Ni = [-Ti(2), Ti(1)];
                    Bi = [0, 0];
                end

                T(i, :) = Ti;
                N(i, :) = Ni;
                B(i, :) = Bi;
            end
        end

        % -------------------------------------------------------------- %
        function kappa = curvature(obj, t)
            % CURVATURE  Scalar curvature at parameter value(s) t.
            %   curvature = rate of change of tangent vector with respect to arc length
            %   kappa = frenet.curvature(t)
            %
            %   kappa(t) = ||dP/dt x d2P/dt2|| / ||dP/dt||^3
            %
            %   Input:  t - [M x 1]
            %   Output: kappa - [M x 1]
            arguments
                obj
                t (:,1) double
            end

            dP  = obj.Spline_.tangent(t);
            d2P = obj.Spline_.secondDerivative(t);
            M   = numel(t);
            d   = obj.Spline_.getDimension();
            kappa = zeros(M, 1);

            for i = 1:M
                v1 = dP(i, :);
                v2 = d2P(i, :);
                normV1 = sqrt(sum(v1.^2));

                if normV1 < 1e-12
                    kappa(i) = 0;
                    continue;
                end

                if d == 3
                    %Theorem 10, page 865
                    crossVec = obj.cross3_(v1, v2);
                    kappa(i) = sqrt(sum(crossVec.^2)) / (normV1^3);
                else %choose x as parameter r(x) = x(i) + f(x)(j)
                    % 2D curvature: |x'*y'' - y'*x''| / (x'^2 + y'^2)^(3/2)
                    kappa(i) = abs(v1(1)*v2(2) - v1(2)*v2(1)) / (normV1^3);
                end
            end
        end

        % -------------------------------------------------------------- %
        function q = quaternion(obj, t)
            % QUATERNION  Orientation quaternion at parameter value(s) t.
            %
            %   q = frenet.quaternion(t)
            %
            %   Returns [M x 4] quaternions in [w, x, y, z] format
            %   representing the rotation from the world frame to the
            %   local Frenet frame (TNB). Compatible with OceanMath.Quat.
            %
            %   For 2D curves, the rotation is around the Z-axis only.
            arguments
                obj
                t (:,1) double
            end

            d = obj.Spline_.getDimension();
            M = numel(t);

            [T_vecs, N_vecs, B_vecs] = obj.frame(t);
            q = zeros(M, 4);

            for i = 1:M
                if d == 3
                    % Build rotation matrix R = [T; N; B]' maps world -> local
                    R = [T_vecs(i,:)', N_vecs(i,:)', B_vecs(i,:)'];
                    
                    q(i, :) = obj.rotmToQuat_(R);
                else
                    % 2D: angle of tangent vector from X-axis
                    theta = atan2(T_vecs(i,2), T_vecs(i,1));
                    % Quaternion for pure Z-axis rotation
                    q(i, :) = [cos(theta/2), 0, 0, sin(theta/2)];
                end
            end
        end

        % -------------------------------------------------------------- %
        function info = describe(obj)
            % DESCRIBE  Print summary.
            L = obj.Spline_.totalLength();
            d = obj.Spline_.getDimension();
            info = sprintf(['FrenetFrame:\n' ...
                '  Spline:    %s\n' ...
                '  Dimension: %dD\n' ...
                '  Arc Length: %.4f\n'], ...
                class(obj.Spline_), d, L);
            fprintf('%s', info);
        end
    end

    % ------------------------------------------------------------------ %
    methods (Access = private)
        function c = cross3_(~, a, b)
            % CROSS3_  Manual 3D cross product (no built-in dependency).
            c = [a(2)*b(3) - a(3)*b(2), ...
                 a(3)*b(1) - a(1)*b(3), ...
                 a(1)*b(2) - a(2)*b(1)];
        end

        function n = arbitraryPerpendicular_(obj, v)
            % ARBITRARYPERPENDICULAR_  Find a unit vector perpendicular to v.
            %   Used when the curve is locally straight (zero curvature).
            if abs(v(1)) < 0.9
                candidate = [1, 0, 0];
            else
                candidate = [0, 1, 0];
            end
            c = obj.cross3_(v, candidate);
            normC = sqrt(sum(c.^2));
            if normC < 1e-12
                n = [0, 0, 1];
            else
                n = c / normC;
            end
        end

        function q = rotmToQuat_(~, R)
            % ROTMTOQUAT_  Convert 3x3 rotation matrix to unit quaternion [w,x,y,z].
            %
            % Uses the Shepperd method for numerical stability.
            % This avoids dependency on any built-in rotm2quat.
            tr = R(1,1) + R(2,2) + R(3,3);

            if tr > 0
                s = 0.5 / sqrt(tr + 1.0);
                w = 0.25 / s;
                x = (R(3,2) - R(2,3)) * s;
                y = (R(1,3) - R(3,1)) * s;
                z = (R(2,1) - R(1,2)) * s;
            elseif R(1,1) > R(2,2) && R(1,1) > R(3,3)
                s = 2.0 * sqrt(1.0 + R(1,1) - R(2,2) - R(3,3));
                w = (R(3,2) - R(2,3)) / s;
                x = 0.25 * s;
                y = (R(1,2) + R(2,1)) / s;
                z = (R(1,3) + R(3,1)) / s;
            elseif R(2,2) > R(3,3)
                s = 2.0 * sqrt(1.0 + R(2,2) - R(1,1) - R(3,3));
                w = (R(1,3) - R(3,1)) / s;
                x = (R(1,2) + R(2,1)) / s;
                y = 0.25 * s;
                z = (R(2,3) + R(3,2)) / s;
            else
                s = 2.0 * sqrt(1.0 + R(3,3) - R(1,1) - R(2,2));
                w = (R(2,1) - R(1,2)) / s;
                x = (R(1,3) + R(3,1)) / s;
                y = (R(2,3) + R(3,2)) / s;
                z = 0.25 * s;
            end

            q = [w, x, y, z];
            % Ensure unit norm
            q = q / sqrt(sum(q.^2));
        end
    end
end
