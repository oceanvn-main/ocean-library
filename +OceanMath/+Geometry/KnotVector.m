classdef KnotVector
    % KNOTVECTOR  Static utilities for B-spline knot sequence generation.
    %
    % A knot vector is a non-decreasing sequence of parameter values that
    % controls where B-spline segments join and how smooth the joins are.
    %
    % Terminology:
    %   n      - number of control points (1-indexed count)
    %   p      - polynomial degree (order = p + 1)
    %   knots  - [1 x (n + p + 1)] non-decreasing sequence
    %
    % Types:
    %   uniform  - equally spaced, all basis functions identical shape
    %   clamped  - repeated boundary knots, curve touches first/last control point
    %   open     - alias for clamped (standard naming convention)
    %
    % Usage:
    %   k = OceanMath.Geometry.KnotVector.clamped(6, 3);
    %   k = OceanMath.Geometry.KnotVector.uniform(6, 3);
    %   tf = OceanMath.Geometry.KnotVector.validate(k, 6, 3);
    %
    % See also: OceanMath.Geometry.BSplineBasis, OceanMath.Geometry.BSplineCurve

    methods (Static)

        function k = uniform(n, p)
            % UNIFORM  Generate a uniform (periodic) knot vector.
            %
            %   k = KnotVector.uniform(n, p)
            %
            %   Equally spaced knots in [0, 1]. No multiplicity at boundaries,
            %   so the curve does NOT start/end at the first/last control point.
            %
            %   Length: n + p + 1
            arguments
                n (1,1) double {mustBePositive, mustBeInteger}
                p (1,1) double {mustBeNonnegative, mustBeInteger}
            end

            m = n + p;  % last knot index (0-based: 0..m)
            k = (0:m) / m;
        end

        function k = clamped(n, p)
            % CLAMPED  Generate a clamped (open) knot vector.
            %
            %   k = KnotVector.clamped(n, p)
            %
            %   The first and last knots are repeated (p+1) times, ensuring
            %   the curve starts at the first control point and ends at the last.
            %   Interior knots are uniformly spaced.
            %
            %   Length: n + p + 1
            arguments
                n (1,1) double {mustBePositive, mustBeInteger}
                p (1,1) double {mustBeNonnegative, mustBeInteger}
            end

            nInternal = n - p - 1;  % number of internal knots
            if nInternal < 0
                nInternal = 0;
            end

            % Boundary: (p+1) zeros, then uniform internals, then (p+1) ones
            internal = linspace(0, 1, nInternal + 2);
            internal = internal(2:end-1);  % strip 0 and 1

            k = [zeros(1, p+1), internal, ones(1, p+1)];
        end

        function k = open(n, p)
            % OPEN  Alias for clamped (common convention).
            k = OceanMath.Geometry.KnotVector.clamped(n, p);
        end

        function k = fromInteriorKnots(interiorKnots, p)
            % FROMINTERIORKNOTS  Build clamped knot vector from explicit interior knots.
            %
            %   k = KnotVector.fromInteriorKnots([0.25, 0.5, 0.75], 3)
            %
            %   This gives full control over knot placement (non-uniform).
            arguments
                interiorKnots (1,:) double
                p             (1,1) double {mustBeNonnegative, mustBeInteger}
            end

            k = [zeros(1, p+1), sort(interiorKnots), ones(1, p+1)];
        end

        function tf = validate(k, n, p)
            % VALIDATE  Check if a knot vector is valid for (n, p).
            %
            %   tf = KnotVector.validate(k, n, p)
            %
            %   Checks:
            %     1. Length == n + p + 1
            %     2. Non-decreasing
            %     3. Values in [0, 1]
            arguments
                k (1,:) double
                n (1,1) double {mustBePositive, mustBeInteger}
                p (1,1) double {mustBeNonnegative, mustBeInteger}
            end

            tf = true;

            % Length check
            if numel(k) ~= n + p + 1
                tf = false;
                return;
            end

            % Non-decreasing
            if any(diff(k) < 0)
                tf = false;
                return;
            end

            % Range [0, 1]
            if k(1) < 0 || k(end) > 1
                tf = false;
            end
        end

        function spans = findSpans(k, p, t)
            % FINDSPANS  Find the knot span index for each parameter value.
            %
            %   spans = KnotVector.findSpans(k, p, t)
            %
            %   For each t(i), returns the index j such that k(j) <= t(i) < k(j+1).
            %   Uses 1-based indexing. Handles boundary case t == k(end).
            %
            %   Output: [1 x N] vector of span indices
            arguments
                k (1,:) double
                p (1,1) double {mustBeNonnegative, mustBeInteger}
                t (:,1) double
            end

            N = numel(t);
            m = numel(k);
            n = m - p - 1;  % number of basis functions
            spans = zeros(1, N);

            for i = 1:N
                ti = t(i);
                % Handle right boundary
                if ti >= k(end)
                    spans(i) = n;  % last valid span
                    continue;
                end
                if ti <= k(1)
                    spans(i) = p + 1;  % first valid span
                    continue;
                end
                % Binary search
                lo = p + 1;
                hi = n + 1;
                while lo < hi
                    mid = floor((lo + hi) / 2);
                    if ti < k(mid)
                        hi = mid;
                    else
                        lo = mid + 1;
                    end
                end
                spans(i) = lo - 1;
            end
        end
    end
end
