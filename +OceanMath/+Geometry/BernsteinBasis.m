classdef BernsteinBasis
    % BERNSTEINBASIS  Bernstein polynomial basis functions.
    %
    % The Bernstein polynomials of degree n are:
    %
    %   B_{i,n}(t) = C(n,i) * t^i * (1-t)^(n-i),   i = 0,...,n
    %
    % They form a partition of unity: sum_{i=0}^{n} B_{i,n}(t) = 1.
    % They are the mathematical basis of Bezier curves.
    %
    % All methods are static — no state, pure math.
    % Backward. Let's data, constraint speak for the fix point P
    %
    % Usage:
    %   B = OceanMath.Geometry.BernsteinBasis.evaluate(3, 0:3, 0.5);
    %   Bmat = OceanMath.Geometry.BernsteinBasis.matrix(3, linspace(0,1,100)');
    %
    % See also: OceanMath.Geometry.BezierCurve

    methods (Static)

        function B = evaluate(n, i, t)
            % EVALUATE  Evaluate Bernstein basis B_{i,n}(t).
            %
            %   B = BernsteinBasis.evaluate(n, i, t)
            %
            %   Inputs:
            %     n - degree (non-negative integer)
            %     i - basis index or vector of indices (0-based, 0 <= i <= n). Index of control points
            %     t - [N x 1] parameter values in [0, 1]
            %
            %   Output:
            %     B - [N x numel(i)] matrix of basis values
            arguments
                n (1,1) double {mustBeNonnegative, mustBeInteger}
                i (1,:) double {mustBeNonnegative, mustBeInteger}
                t (:,1) double
            end

            N = numel(t);
            nI = numel(i);
            B = zeros(N, nI);

            for k = 1:nI
                idx = i(k);
                % Binomial coefficient via nchoosek
                coeff = nchoosek(n, idx);
                B(:, k) = coeff .* (t .^ idx) .* ((1 - t) .^ (n - idx));
            end
        end

        function Bmat = matrix(n, t)
            % MATRIX  Full basis matrix for degree n at parameter values t.
            %
            %   Bmat = BernsteinBasis.matrix(3, linspace(0,1,100)')
            %
            %   Output: [N x (n+1)] matrix where column j = B_{j-1, n}(t)
            %
            %   Rows sum to 1 (partition of unity).
            arguments
                n (1,1) double {mustBeNonnegative, mustBeInteger}
                t (:,1) double
            end

            Bmat = OceanMath.Geometry.BernsteinBasis.evaluate(n, 0:n, t);
        end

        function dB = derivative(n, i, t)
            % DERIVATIVE  First derivative of Bernstein basis dB_{i,n}/dt.
            %
            %   dB = BernsteinBasis.derivative(n, i, t)
            %
            %   Uses the identity:
            %     dB_{i,n}/dt = n * (B_{i-1,n-1}(t) - B_{i,n-1}(t))
            %
            %   Where B_{-1,n-1} = 0 and B_{n,n-1} = 0 by convention.
            %
            %   Output: [N x numel(i)] matrix of derivative values
            arguments
                n (1,1) double {mustBeNonnegative, mustBeInteger}
                i (1,:) double {mustBeNonnegative, mustBeInteger}
                t (:,1) double
            end

            N = numel(t);
            nI = numel(i);
            dB = zeros(N, nI);

            if n == 0
                return;  % Constant → derivative is zero
            end

            for k = 1:nI
                idx = i(k);
                % B_{idx-1, n-1}(t) — zero if idx == 0
                if idx > 0
                    left = OceanMath.Geometry.BernsteinBasis.evaluate(n-1, idx-1, t);
                else
                    left = zeros(N, 1);
                end
                % B_{idx, n-1}(t) — zero if idx == n
                if idx < n
                    right = OceanMath.Geometry.BernsteinBasis.evaluate(n-1, idx, t);
                else
                    right = zeros(N, 1);
                end
                dB(:, k) = n * (left - right);
            end
        end

        function dBmat = derivativeMatrix(n, t)
            % DERIVATIVEMATRIX  Full derivative matrix for degree n.
            %
            %   dBmat = BernsteinBasis.derivativeMatrix(3, linspace(0,1,100)')
            %
            %   Output: [N x (n+1)] matrix where column j = dB_{j-1,n}/dt
            arguments
                n (1,1) double {mustBeNonnegative, mustBeInteger}
                t (:,1) double
            end

            dBmat = OceanMath.Geometry.BernsteinBasis.derivative(n, 0:n, t);
        end

        function d2B = secondDerivative(n, i, t)
            % SECONDDERIVATIVE  Second derivative of Bernstein basis.
            %
            %   d2B_{i,n}/dt2 = n*(n-1) * (B_{i-2,n-2} - 2*B_{i-1,n-2} + B_{i,n-2})
            %
            %   Output: [N x numel(i)] matrix
            arguments
                n (1,1) double {mustBeNonnegative, mustBeInteger}
                i (1,:) double {mustBeNonnegative, mustBeInteger}
                t (:,1) double
            end

            N = numel(t);
            nI = numel(i);
            d2B = zeros(N, nI);

            if n <= 1
                return;  % Linear or constant → second derivative is zero
            end

            for k = 1:nI
                idx = i(k);
                % Evaluate B_{idx-2, n-2}, B_{idx-1, n-2}, B_{idx, n-2}
                % with boundary convention: out-of-range = 0
                left = zeros(N, 1);
                mid  = zeros(N, 1);
                right = zeros(N, 1);

                if idx >= 2
                    left = OceanMath.Geometry.BernsteinBasis.evaluate(n-2, idx-2, t);
                end
                if idx >= 1 && idx <= n-1
                    mid = OceanMath.Geometry.BernsteinBasis.evaluate(n-2, idx-1, t);
                end
                if idx <= n-2
                    right = OceanMath.Geometry.BernsteinBasis.evaluate(n-2, idx, t);
                end

                d2B(:, k) = n * (n-1) * (left - 2*mid + right);
            end
        end
    end
end
