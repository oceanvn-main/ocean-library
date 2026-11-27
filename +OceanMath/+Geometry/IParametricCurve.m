classdef (Abstract) IParametricCurve < handle
    % IPARAMETRICCURVE  Abstract interface for parametric curves in N-dimensional space.
    %
    % A parametric curve maps a scalar parameter t in [0, 1] to a point
    % in R^d (typically d = 2 or 3). This is fundamentally different from
    % the statistical univariate smoothers in +OceanStats/+NonParametric
    % (which fit y = f(x)). Here, every coordinate is a function of t:
    %
    %   P(t) = [x(t), y(t), z(t)]    t in [0, 1]
    %
    % This enables curves that loop, overlap, and trace arbitrary 3D paths
    % (rollercoasters, transit routes, etc.).
    %
    % Contract:
    %   evaluate(t)        -> [N x d] points on the curve
    %   tangent(t)         -> [N x d] first derivative dP/dt
    %   secondDerivative(t)-> [N x d] second derivative d2P/dt2
    %   arcLength(t0, t1)  -> scalar, exact curve distance
    %   totalLength()      -> scalar, full arc length from t=0 to t=1
    %   getControlPoints() -> [K x d] the defining knots
    %   getDimension()     -> scalar d (2 or 3)
    %   getNumSegments()   -> number of piecewise segments
    %
    % Design:
    %   - Strategy-compatible: inject into simulators, mappers, renderers.
    %   - Pure math: no MATLAB built-ins (spline, interp1, etc.).
    %   - All derivatives are analytical, not finite-difference.
    %   - Arc length uses OceanMath.SimpsonEstimate for numerical integration.
    %
    % See also: OceanMath.Geometry.CatmullRomSpline,
    %           OceanMath.Geometry.FrenetFrame

    % ------------------------------------------------------------------ %
    properties (SetAccess = protected)
        % ControlPoints_ : [K x d] matrix of knots defining the curve
        ControlPoints_  double

        % Dimension_ : spatial dimension (2 or 3)
        Dimension_  (1,1) double {mustBePositive, mustBeInteger} = 2
        
        % Memory Caches for Ultra-Fast Monotonic Calculus LUT Arrays
        LUT_S_ double = []
        LUT_T_ double = []
    end

    % ------------------------------------------------------------------ %
    methods (Abstract)
        % EVALUATE  Map parameter values to spatial coordinates.
        %
        %   P = obj.evaluate(t)
        %
        %   Input:  t - [N x 1] or [1 x N] parameter values in [0, 1]
        %   Output: P - [N x d] matrix of spatial coordinates
        P = evaluate(obj, t)

        % TANGENT  First derivative of the curve (velocity direction).
        %
        %   T = obj.tangent(t)
        %
        %   Input:  t - [N x 1] parameter values in [0, 1]
        %   Output: T - [N x d] tangent vectors (not normalized)
        T = tangent(obj, t)

        % SECONDDERIVATIVE  Second derivative of the curve (curvature direction).
        %
        %   D2 = obj.secondDerivative(t)
        %
        %   Input:  t - [N x 1] parameter values in [0, 1]
        %   Output: D2 - [N x d] second derivative vectors
        D2 = secondDerivative(obj, t)
    end

    % ------------------------------------------------------------------ %
    methods
        function k = curvature(obj, t)
            % CURVATURE  Compute the scalar curvature k at parameter t.
            %   k = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
            T = obj.tangent(t);
            D2 = obj.secondDerivative(t);
            
            % Safe guard against divide by zero if curve completely stops
            normT3 = (T(:,1).^2 + T(:,2).^2).^(1.5);
            normT3(normT3 < 1e-12) = 1e-12;
            
            k = abs(T(:,1).*D2(:,2) - T(:,2).*D2(:,1)) ./ normT3;
        end

        function L = arcLength(obj, t0, t1)
            % ARCLENGTH  Compute exact curve distance between two parameter values.
            %
            %   L = obj.arcLength(t0, t1)
            %
            %   Uses Simpson's 1/3 rule to integrate ||dP/dt|| from t0 to t1.
            %   This gives the true curved distance, not the straight-line distance.
            %
            %   Inputs:
            %     t0 - scalar, start parameter (default: 0)
            %     t1 - scalar, end parameter   (default: 1)
            %
            %   Output:
            %     L  - scalar, arc length in the same units as ControlPoints_
            arguments
                obj
                t0 (1,1) double = 0
                t1 (1,1) double = 1
            end

            if t0 == t1
                L = 0;
                return;
            end

            % Speed function: ||dP/dt|| at parameter t
            speedFn = @(tVals) obj.computeSpeed_(tVals);

            % Use our own Simpson integration (100 subintervals)
            L = OceanMath.SimpsonEstimate(speedFn, t0, t1, 100);
        end

        function L = totalLength(obj)
            % TOTALLENGTH  Full arc length of the curve from t=0 to t=1.
            L = obj.arcLength(0, 1);
        end

        function t = getParameterAtArcLength(obj, targetL, ~)
            % GETPARAMETERATARCLENGTH  Find curve parameter t perfectly matching arc length L.
            %
            %   t = obj.getParameterAtArcLength(targetL)
            %
            %   Uses a high-precision pre-computed Look-Up-Table (LUT) to guarantee
            %   perfect monotonic evaluation, utterly eliminating "integration noise jitter".
            arguments
                obj
                targetL (1,1) double
                ~       % ignored tGuess
            end
            
            % 1. Lazy-Initialize the Calculus Memory Matrix (LUT)
            if isempty(obj.LUT_S_)
                obj.buildLUT_();
            end
            
            % 2. Safe Extrapolation Guards
            maxL = obj.LUT_S_(end);
            targetL = max(0, min(targetL, maxL));
            if targetL == 0, t = 0; return; end
            if targetL >= maxL, t = 1; return; end
            
            % 3. Ultra-Fast Matrix Interpolation natively avoids Integration Numerical Noise
            t = interp1(obj.LUT_S_, obj.LUT_T_, targetL, 'linear', 'extrap');
            t = max(0, min(t, 1));
        end

        function pts = getControlPoints(obj)
            % GETCONTROLPOINTS  Return the defining knots [K x d].
            pts = obj.ControlPoints_;
        end

        function d = getDimension(obj)
            % GETDIMENSION  Return spatial dimension (2 or 3).
            d = obj.Dimension_;
        end

        function n = getNumSegments(obj)
            % GETNUMSEGMENTS  Number of piecewise polynomial segments.
            %   For most curves this is (number of control points - 1),
            %   but subclasses may override.
            n = size(obj.ControlPoints_, 1) - 1;
        end

        function tVals = parameterize(obj, nSamples)
            % PARAMETERIZE  Generate N uniformly-spaced parameter values.
            %
            %   tVals = obj.parameterize(100)
            %
            %   Useful for quick plotting: P = obj.evaluate(obj.parameterize(200))
            arguments
                obj
                nSamples (1,1) double {mustBePositive, mustBeInteger} = 100
            end
            tVals = linspace(0, 1, nSamples)';
        end

        function tVals = parameterizeByArcLength(obj, nSamples)
            % PARAMETERIZEBYARCLENGTH  Generate N parameter values that are
            % equally spaced along the actual curve length (not equally
            % spaced in t). This prevents bunching on sharp bends.
            %
            %   tVals = obj.parameterizeByArcLength(100)
            arguments
                obj
                nSamples (1,1) double {mustBePositive, mustBeInteger} = 100
            end

            % Step 1: Dense uniform parameterization
            nDense = max(nSamples * 10, 1000);
            tDense = linspace(0, 1, nDense)';

            % Step 2: Cumulative arc length at each dense sample
            speeds = obj.computeSpeed_(tDense');
            dt = tDense(2) - tDense(1);
            %Total distance traveld fro mthe start up to dense sample point i
            cumLen = [0; cumsum(speeds(1:end-1)' * dt)];
            
            totalLen = cumLen(end);

            % Step 3: Target equally-spaced arc length positions
            targetLengths = linspace(0, totalLen, nSamples)';

            % Step 4: Invert the cumulative length mapping via linear interp
            % (manual, not using interp1 built-in)
            tVals = zeros(nSamples, 1);
            tVals(1) = 0;
            tVals(end) = 1;
            idx = 1;
            for k = 2:nSamples-1
                sTarget = targetLengths(k);
                while idx < nDense && cumLen(idx+1) < sTarget
                    idx = idx + 1;
                end
                % Linear interpolation between tDense(idx) and tDense(idx+1)
                frac = (sTarget - cumLen(idx)) / max(cumLen(idx+1) - cumLen(idx), eps);
                tVals(k) = tDense(idx) + frac * dt;
            end
        end
    end

    % ------------------------------------------------------------------ %
    methods (Access = protected)
        function buildLUT_(obj)
            % BUILDLUT_ Pre-computes deterministic distance map to prevent numerical jitter
            nDense = 5000; % Extremely high resolution lookup mapping
            tDense = linspace(0, 1, nDense)';
            
            speeds = obj.computeSpeed_(tDense');
            dt = tDense(2) - tDense(1);
            
            % Cumulative geometric distance mapping
            sDense = [0; cumsum(speeds(1:end-1)' * dt)];
            
            % Force strict monotonicity to prevent interp1 failures due to float identicals
            [sUnique, uniqueIdx] = unique(sDense);
            tUnique = tDense(uniqueIdx);
            
            obj.LUT_S_ = sUnique;
            obj.LUT_T_ = tUnique;
        end
        function speeds = computeSpeed_(obj, tVals)
            % COMPUTESPEED_  ||dP/dt|| at each parameter value.
            %   Internal helper for arc length integration.
            %
            %   Input:  tVals - [1 x N] row vector of parameter values
            %   Output: speeds - [1 x N] row of scalar speeds
            T = obj.tangent(tVals(:));   % [N x d]
            speeds = sqrt(sum(T.^2, 2))';  % [1 x N]
        end
    end
end
