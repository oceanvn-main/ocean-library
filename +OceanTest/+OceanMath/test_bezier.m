classdef test_bezier < matlab.unittest.TestCase
    % TEST_BEZIER  Tests for BernsteinBasis and BezierCurve.

    % ============================================================== %
    %  BernsteinBasis Tests
    % ============================================================== %

    methods (Test)

        function test_bernstein_partition_of_unity(testCase)
            % All Bernstein basis values must sum to 1 for any t.
            t = linspace(0, 1, 50)';
            for n = 1:5
                Bmat = OceanMath.Geometry.BernsteinBasis.matrix(n, t);
                rowSums = sum(Bmat, 2);
                testCase.assertEqual(rowSums, ones(50, 1), 'AbsTol', 1e-12, ...
                    sprintf('Partition of unity failed for degree %d.', n));
            end
        end

        function test_bernstein_endpoints(testCase)
            % B_{0,n}(0) = 1, B_{n,n}(1) = 1
            n = 4;
            B0 = OceanMath.Geometry.BernsteinBasis.evaluate(n, 0, 0);
            testCase.assertEqual(B0, 1, 'AbsTol', 1e-12);

            Bn = OceanMath.Geometry.BernsteinBasis.evaluate(n, n, 1);
            testCase.assertEqual(Bn, 1, 'AbsTol', 1e-12);
        end

        function test_bernstein_symmetry(testCase)
            % B_{i,n}(t) = B_{n-i,n}(1-t)
            n = 3;
            t = linspace(0.1, 0.9, 20)';
            for i = 0:n
                left  = OceanMath.Geometry.BernsteinBasis.evaluate(n, i, t);
                right = OceanMath.Geometry.BernsteinBasis.evaluate(n, n-i, 1-t);
                testCase.assertEqual(left, right, 'AbsTol', 1e-12, ...
                    sprintf('Symmetry failed for B_{%d,%d}.', i, n));
            end
        end

        function test_bernstein_derivative(testCase)
            % Verify derivative against finite difference.
            n = 3;
            t = linspace(0.1, 0.9, 20)';
            dt = 1e-7;

            for i = 0:n
                analytic = OceanMath.Geometry.BernsteinBasis.derivative(n, i, t);
                f_plus  = OceanMath.Geometry.BernsteinBasis.evaluate(n, i, t + dt);
                f_minus = OceanMath.Geometry.BernsteinBasis.evaluate(n, i, t - dt);
                fd = (f_plus - f_minus) / (2 * dt);
                testCase.assertEqual(analytic, fd, 'AbsTol', 1e-5, ...
                    sprintf('Derivative mismatch for B_{%d,%d}.', i, n));
            end
        end

        function test_bernstein_second_derivative(testCase)
            % Verify second derivative against finite difference.
            n = 4;
            t = linspace(0.1, 0.9, 20)';
            dt = 1e-5;

            for i = 0:n
                analytic = OceanMath.Geometry.BernsteinBasis.secondDerivative(n, i, t);
                d1_plus  = OceanMath.Geometry.BernsteinBasis.derivative(n, i, t + dt);
                d1_minus = OceanMath.Geometry.BernsteinBasis.derivative(n, i, t - dt);
                fd2 = (d1_plus - d1_minus) / (2 * dt);
                testCase.assertEqual(analytic, fd2, 'AbsTol', 1e-3, ...
                    sprintf('Second derivative mismatch for B_{%d,%d}.', i, n));
            end
        end

        function test_bernstein_matrix_size(testCase)
            n = 5;
            t = linspace(0, 1, 30)';
            Bmat = OceanMath.Geometry.BernsteinBasis.matrix(n, t);
            testCase.assertEqual(size(Bmat), [30, 6]);
        end

        % ============================================================== %
        %  BezierCurve Tests
        % ============================================================== %

        function test_bezier_endpoint_interpolation(testCase)
            % Bezier must pass through first and last control point.
            pts = [0 0; 1 3; 4 2; 6 0];
            c = OceanMath.Geometry.BezierCurve(pts);

            P0 = c.evaluate(0);
            P1 = c.evaluate(1);
            testCase.assertEqual(P0, [0 0], 'AbsTol', 1e-12);
            testCase.assertEqual(P1, [6 0], 'AbsTol', 1e-12);
        end

        function test_bezier_linear(testCase)
            % Degree-1 Bezier = straight line.
            pts = [0 0; 10 5];
            c = OceanMath.Geometry.BezierCurve(pts);

            t = linspace(0, 1, 11)';
            P = c.evaluate(t);

            expected = t .* [10 5];
            testCase.assertEqual(P, expected, 'AbsTol', 1e-12);
        end

        function test_bezier_midpoint_quadratic(testCase)
            % Degree-2 Bezier: P(0.5) = 0.25*P0 + 0.5*P1 + 0.25*P2
            pts = [0 0; 2 4; 4 0];
            c = OceanMath.Geometry.BezierCurve(pts);

            P_half = c.evaluate(0.5);
            expected = 0.25*pts(1,:) + 0.5*pts(2,:) + 0.25*pts(3,:);
            testCase.assertEqual(P_half, expected, 'AbsTol', 1e-12);
        end

        function test_bezier_tangent_at_endpoints(testCase)
            % Tangent at t=0 = n*(P1-P0), at t=1 = n*(Pn-Pn-1)
            pts = [0 0; 1 3; 4 2; 6 0];
            c = OceanMath.Geometry.BezierCurve(pts);
            n = 3;

            T0 = c.tangent(0);
            T1 = c.tangent(1);

            testCase.assertEqual(T0, n * (pts(2,:) - pts(1,:)), 'AbsTol', 1e-10);
            testCase.assertEqual(T1, n * (pts(4,:) - pts(3,:)), 'AbsTol', 1e-10);
        end

        function test_bezier_tangent_vs_finite_diff(testCase)
            % Verify tangent against finite difference.
            pts = [0 0; 1 3; 4 2; 6 0];
            c = OceanMath.Geometry.BezierCurve(pts);

            t = linspace(0.1, 0.9, 10)';
            dt = 1e-7;
            T_analytic = c.tangent(t);

            for i = 1:numel(t)
                P_plus  = c.evaluate(t(i) + dt);
                P_minus = c.evaluate(t(i) - dt);
                T_fd = (P_plus - P_minus) / (2 * dt);
                testCase.assertEqual(T_analytic(i,:), T_fd, 'AbsTol', 1e-4, ...
                    sprintf('Tangent mismatch at t=%.2f.', t(i)));
            end
        end

        function test_bezier_second_derivative_vs_fd(testCase)
            % Verify second derivative against finite difference.
            pts = [0 0; 1 3; 4 2; 6 0; 8 1];
            c = OceanMath.Geometry.BezierCurve(pts);

            t = linspace(0.1, 0.9, 10)';
            dt = 1e-5;

            for i = 1:numel(t)
                D2_analytic = c.secondDerivative(t(i));
                T_plus  = c.tangent(t(i) + dt);
                T_minus = c.tangent(t(i) - dt);
                D2_fd = (T_plus - T_minus) / (2 * dt);
                testCase.assertEqual(D2_analytic, D2_fd, 'AbsTol', 1e-2, ...
                    sprintf('SecondDeriv mismatch at t=%.2f.', t(i)));
            end
        end

        function test_bezier_arc_length_straight_line(testCase)
            % Arc length of a straight line = Euclidean distance.
            pts = [0 0; 3 4];
            c = OceanMath.Geometry.BezierCurve(pts);
            L = c.totalLength();
            testCase.assertEqual(L, 5, 'AbsTol', 1e-6);
        end

        function test_bezier_3d(testCase)
            % 3D Bezier curve.
            pts = [0 0 0; 1 1 1; 2 0 2; 3 1 0];
            c = OceanMath.Geometry.BezierCurve(pts);

            testCase.assertEqual(c.getDimension(), 3);
            P0 = c.evaluate(0);
            P1 = c.evaluate(1);
            testCase.assertEqual(P0, [0 0 0], 'AbsTol', 1e-12);
            testCase.assertEqual(P1, [3 1 0], 'AbsTol', 1e-12);
        end

        function test_bezier_degree(testCase)
            pts = [0 0; 1 2; 3 1; 5 0; 7 2];
            c = OceanMath.Geometry.BezierCurve(pts);
            testCase.assertEqual(c.Degree_, 4);
            testCase.assertEqual(c.getNumSegments(), 1);
        end

        function test_bezier_iparametric_interface(testCase)
            % BezierCurve implements IParametricCurve.
            pts = [0 0; 3 4; 6 0];
            c = OceanMath.Geometry.BezierCurve(pts);
            testCase.assertClass(c, 'OceanMath.Geometry.BezierCurve');
            testCase.assertTrue(isa(c, 'OceanMath.Geometry.IParametricCurve'));
        end

        function test_bezier_set_control_point(testCase)
            pts = [0 0; 3 4; 6 0];
            c = OceanMath.Geometry.BezierCurve(pts);
            c.setControlPoint(2, [3, 6]);
            P_mid = c.evaluate(0.5);
            expected = 0.25*[0 0] + 0.5*[3 6] + 0.25*[6 0];
            testCase.assertEqual(P_mid, expected, 'AbsTol', 1e-12);
        end
    end
end
