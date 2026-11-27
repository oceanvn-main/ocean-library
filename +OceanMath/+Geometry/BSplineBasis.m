classdef BSplineBasis
    % BSPLINEBASIS  B-spline basis function evaluation (iterative Cox-de Boor).
    %
    % Replaces the recursive bspline.m with an iterative, vectorized algorithm.
    % No stack overflow risk, O(n*p) per evaluation point instead of O(2^p).
    %
    % The i-th B-spline basis function of degree p on knot vector k is:
    %
    %   N_{i,0}(t) = 1 if k(i) <= t < k(i+1), else 0
    %   N_{i,p}(t) = [(t - k(i)) / (k(i+p) - k(i))] * N_{i,p-1}(t)
    %              + [(k(i+p+1) - t) / (k(i+p+1) - k(i+1))] * N_{i+1,p-1}(t)
    %
    % Convention: 0/0 = 0 (for zero-width knot spans).
    %
    % Usage:
    %   k = OceanMath.Geometry.KnotVector.clamped(6, 3);
    %   B = OceanMath.Geometry.BSplineBasis.evaluate(k, 3, linspace(0,1,100)');
    %
    % See also: OceanMath.Geometry.KnotVector, OceanMath.Geometry.BSplineCurve

    methods (Static)

        function B = evaluate(knots, degree, t)
            % EVALUATE  Evaluate all B-spline basis functions at parameter values.
            %
            %   B = BSplineBasis.evaluate(knots, degree, t)
            %
            %   Inputs:
            %     knots  - [1 x m] knot vector
            %     degree - polynomial degree p
            %     t      - [N x 1] parameter values
            %
            %   Output:
            %     B - [N x n] matrix, where n = m - p - 1 (number of basis fns)
            %         B(j, i) = N_{i,p}(t(j))
            arguments
                knots  (1,:) double
                degree (1,1) double {mustBeNonnegative, mustBeInteger}
                t      (:,1) double
            end

            m = numel(knots);
            n = m - degree - 1;  % number of basis functions
            N = numel(t);
            p = degree;

            % --- Degree 0: step functions ---
            B = zeros(N, m - 1);
            for i = 1:m-1
                if knots(i) == knots(i+1)
                    continue;  % zero-width span
                end
                mask = (t >= knots(i)) & (t < knots(i+1));
                B(mask, i) = 1;
            end
            % Handle right boundary: t == knots(end)
            rightMask = (t >= knots(end));
            if any(rightMask)
                % Find last non-zero-width span
                for i = m-1:-1:1
                    if knots(i) < knots(i+1)
                        B(rightMask, i) = 1;
                        break;
                    end
                end
            end

            % --- Build up degree by degree (triangular table) ---
            for d = 1:p
                nBasis = m - d - 1;
                Bnew = zeros(N, nBasis);
                for i = 1:nBasis
                    % Left term: (t - k_i) / (k_{i+d} - k_i) * N_{i,d-1}
                    denom1 = knots(i+d) - knots(i);
                    if denom1 > 0
                        Bnew(:, i) = Bnew(:, i) + ...
                            ((t - knots(i)) / denom1) .* B(:, i);
                    end

                    % Right term: (k_{i+d+1} - t) / (k_{i+d+1} - k_{i+1}) * N_{i+1,d-1}
                    denom2 = knots(i+d+1) - knots(i+1);
                    if denom2 > 0
                        Bnew(:, i) = Bnew(:, i) + ...
                            ((knots(i+d+1) - t) / denom2) .* B(:, i+1);
                    end
                end
                B = Bnew;
            end

            % B is now [N x n]
        end

        function dB = derivative(knots, degree, t)
            % DERIVATIVE  First derivative of all B-spline basis functions.
            %
            %   dB = BSplineBasis.derivative(knots, degree, t)
            %
            %   Uses the identity:
            %     N'_{i,p}(t) = p/(k_{i+p}-k_i) * N_{i,p-1}(t)
            %                 - p/(k_{i+p+1}-k_{i+1}) * N_{i+1,p-1}(t)
            %
            %   Output: [N x n] matrix of derivative values
            arguments
                knots  (1,:) double
                degree (1,1) double {mustBeNonnegative, mustBeInteger}
                t      (:,1) double
            end

            if degree == 0
                dB = zeros(numel(t), numel(knots) - 1);
                return;
            end

            m = numel(knots);
            n = m - degree - 1;
            N = numel(t);
            p = degree;

            % Evaluate basis of degree (p-1)
            B_lower = OceanMath.Geometry.BSplineBasis.evaluate(knots, p-1, t);
            nLower = size(B_lower, 2);

            dB = zeros(N, n);
            for i = 1:n
                % Left term
                denom1 = knots(i+p) - knots(i);
                if denom1 > 0 && i <= nLower
                    dB(:, i) = dB(:, i) + (p / denom1) .* B_lower(:, i);
                end

                % Right term (subtract)
                denom2 = knots(i+p+1) - knots(i+1);
                if denom2 > 0 && (i+1) <= nLower
                    dB(:, i) = dB(:, i) - (p / denom2) .* B_lower(:, i+1);
                end
            end
        end

        function Bmat = basisMatrix(x, knots, degree)
            % BASISMATRIX  Convenience alias matching statistics convention.
            %
            %   Bmat = BSplineBasis.basisMatrix(x, knots, degree)
            %
            %   Same as evaluate() but with x as first argument for
            %   compatibility with +OceanStats code.
            arguments
                x      (:,1) double
                knots  (1,:) double
                degree (1,1) double {mustBeNonnegative, mustBeInteger}
            end

            Bmat = OceanMath.Geometry.BSplineBasis.evaluate(knots, degree, x);
        end
    end
end
