classdef test_bspline_curve < matlab.unittest.TestCase
    % TEST_BSPLINE_CURVE  Tests for BSplineCurve.

    methods (Test)

        function test_clamped_endpoint_interpolation(testCase)
            % Clamped B-spline must start at P1 and end at Pn.
            pts = [0 0; 1 3; 3 4; 5 2; 7 0; 9 1];
            c = OceanMath.Geometry.BSplineCurve(pts, 'Degree', 3);

            P0 = c.evaluate(0);
            P1 = c.evaluate(1);
            testCase.assertEqual(P0, [0 0], 'AbsTol', 1e-10, ...
                'Clamped B-spline must start at first control point.');
            testCase.assertEqual(P1, [9 1], 'AbsTol', 1e-10, ...
                'Clamped B-spline must end at last control point.');
        end

        function test_degree1_is_polyline(testCase)
            % Degree 1 B-spline = piecewise linear (polyline).
            pts = [0 0; 2 4; 6 2; 10 0];
            c = OceanMath.Geometry.BSplineCurve(pts, 'Degree', 1);

            % Midpoint of first segment should be average of P1 and P2
            % With clamped degree-1 knots = [0 0 1/3 2/3 1 1]
            % At t corresponding to midpoint of segment 1
            P0 = c.evaluate(0);
            P1 = c.evaluate(1);
            testCase.assertEqual(P0, [0 0], 'AbsTol', 1e-10);
            testCase.assertEqual(P1, [10 0], 'AbsTol', 1e-10);
        end

        function test_partition_of_unity_means_within_hull(testCase)
            % All curve points must lie within the convex hull of control points.
            pts = [0 0; 2 5; 5 5; 8 0];
            c = OceanMath.Geometry.BSplineCurve(pts, 'Degree', 3);

            t = linspace(0, 1, 100)';
            P = c.evaluate(t);

            % All x-coordinates within [0, 8], y within [0, 5]
            testCase.assertTrue(all(P(:,1) >= -1e-10 & P(:,1) <= 8 + 1e-10));
            testCase.assertTrue(all(P(:,2) >= -1e-10 & P(:,2) <= 5 + 1e-10));
        end

        function test_tangent_vs_finite_diff(testCase)
            % Verify tangent against finite difference.
            pts = [0 0; 1 3; 3 4; 5 2; 7 0; 9 1];
            c = OceanMath.Geometry.BSplineCurve(pts, 'Degree', 3);

            t = linspace(0.05, 0.95, 15)';
            dt = 1e-7;
            T = c.tangent(t);

            for i = 1:numel(t)
                P_plus  = c.evaluate(t(i) + dt);
                P_minus = c.evaluate(t(i) - dt);
                T_fd = (P_plus - P_minus) / (2 * dt);
                testCase.assertEqual(T(i,:), T_fd, 'AbsTol', 1e-3, ...
                    sprintf('Tangent mismatch at t=%.2f.', t(i)));
            end
        end

        function test_second_derivative_vs_fd(testCase)
            % Verify second derivative against finite difference of tangent.
            pts = [0 0; 1 3; 3 4; 5 2; 7 0; 9 1];
            c = OceanMath.Geometry.BSplineCurve(pts, 'Degree', 3);

            t = linspace(0.1, 0.9, 10)';
            dt = 1e-5;

            for i = 1:numel(t)
                D2 = c.secondDerivative(t(i));
                T_plus  = c.tangent(t(i) + dt);
                T_minus = c.tangent(t(i) - dt);
                D2_fd = (T_plus - T_minus) / (2 * dt);
                testCase.assertEqual(D2, D2_fd, 'AbsTol', 0.5, ...
                    sprintf('SecondDeriv mismatch at t=%.2f.', t(i)));
            end
        end

        function test_arc_length_straight_line(testCase)
            % Degree 1 straight line: arc length = Euclidean distance.
            pts = [0 0; 3 4];
            c = OceanMath.Geometry.BSplineCurve(pts, 'Degree', 1);
            L = c.totalLength();
            testCase.assertEqual(L, 5, 'AbsTol', 1e-4);
        end

        function test_local_control(testCase)
            % Moving one control point should not affect distant parts.
            pts = [0 0; 1 3; 3 4; 5 2; 7 0; 9 1];
            c = OceanMath.Geometry.BSplineCurve(pts, 'Degree', 3);

            % Evaluate far from point 2 (near t=0.9)
            P_before = c.evaluate(0.9);

            % Move control point 2
            c.setControlPoint(2, [1, 6]);
            P_after = c.evaluate(0.9);

            % Should be very similar (local control property)
            testCase.assertEqual(P_before, P_after, 'AbsTol', 1e-6, ...
                'Moving a control point should not affect distant segments.');
        end

        function test_3d_curve(testCase)
            % 3D B-spline curve.
            pts = [0 0 0; 1 1 1; 2 0 2; 3 1 0; 4 0 1];
            c = OceanMath.Geometry.BSplineCurve(pts, 'Degree', 3);

            testCase.assertEqual(c.getDimension(), 3);
            P0 = c.evaluate(0);
            P1 = c.evaluate(1);
            testCase.assertEqual(P0, [0 0 0], 'AbsTol', 1e-10);
            testCase.assertEqual(P1, [4 0 1], 'AbsTol', 1e-10);
        end

        function test_iparametric_interface(testCase)
            % BSplineCurve implements IParametricCurve.
            pts = [0 0; 2 4; 5 3; 8 0];
            c = OceanMath.Geometry.BSplineCurve(pts, 'Degree', 3);
            testCase.assertTrue(isa(c, 'OceanMath.Geometry.IParametricCurve'));
        end

        function test_degree_capped(testCase)
            % Degree should be capped at K-1.
            pts = [0 0; 5 5];  % Only 2 points
            c = OceanMath.Geometry.BSplineCurve(pts, 'Degree', 5);
            testCase.assertEqual(c.Degree_, 1, ...
                'Degree should be capped at K-1 = 1.');
        end

        function test_add_control_point(testCase)
            pts = [0 0; 3 4; 6 0];
            c = OceanMath.Geometry.BSplineCurve(pts, 'Degree', 2);
            c.addControlPoint([9, 2]);
            testCase.assertEqual(size(c.getControlPoints(), 1), 4);
            P1 = c.evaluate(1);
            testCase.assertEqual(P1, [9 2], 'AbsTol', 1e-10);
        end

        function test_remove_control_point(testCase)
            pts = [0 0; 2 4; 4 3; 6 0; 8 1];
            c = OceanMath.Geometry.BSplineCurve(pts, 'Degree', 3);
            c.removeControlPoint(3);
            testCase.assertEqual(size(c.getControlPoints(), 1), 4);
        end

        function test_custom_knots(testCase)
            % User can supply custom knot vector.
            % n=5, p=3 -> knot length = 5+3+1 = 9, so 1 interior knot
            pts = [0 0; 1 3; 3 4; 5 2; 7 0];
            k = OceanMath.Geometry.KnotVector.fromInteriorKnots([0.5], 3);
            c = OceanMath.Geometry.BSplineCurve(pts, 'Degree', 3, 'Knots', k);
            testCase.assertTrue(OceanMath.Geometry.KnotVector.validate(c.Knots_, 5, 3));

            % Should still interpolate endpoints (clamped)
            P0 = c.evaluate(0);
            P1 = c.evaluate(1);
            testCase.assertEqual(P0, [0 0], 'AbsTol', 1e-10);
            testCase.assertEqual(P1, [7 0], 'AbsTol', 1e-10);
        end

        function test_frenet_compatibility(testCase)
            % BSplineCurve should work with FrenetFrame.
            pts = [0 0 0; 1 2 1; 3 1 3; 5 3 0; 7 0 2];
            c = OceanMath.Geometry.BSplineCurve(pts, 'Degree', 3);
            frenet = OceanMath.Geometry.FrenetFrame(c);

            [T, N, B] = frenet.frame(0.5);
            % T, N, B should be unit vectors
            testCase.assertEqual(norm(T), 1, 'AbsTol', 1e-6);
            testCase.assertEqual(norm(N), 1, 'AbsTol', 1e-6);
            testCase.assertEqual(norm(B), 1, 'AbsTol', 1e-6);
            % Orthogonality: T.N = 0, T.B = 0, N.B = 0
            testCase.assertEqual(dot(T, N), 0, 'AbsTol', 1e-6);
            testCase.assertEqual(dot(T, B), 0, 'AbsTol', 1e-6);
            testCase.assertEqual(dot(N, B), 0, 'AbsTol', 1e-6);
        end
    end
end
