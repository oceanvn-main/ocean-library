classdef test_nurbs_curve < matlab.unittest.TestCase
    % TEST_NURBS_CURVE  Tests for NURBSCurve.

    methods (Test)

        function test_unit_weights_equals_bspline(testCase)
            % All weights = 1 should produce identical results to BSplineCurve.
            pts = [0 0; 1 3; 3 4; 5 2; 7 0; 9 1];
            w = ones(6, 1);

            nurbs = OceanMath.Geometry.NURBSCurve(pts, w, 'Degree', 3);
            bspl  = OceanMath.Geometry.BSplineCurve(pts, 'Degree', 3);

            t = linspace(0, 1, 50)';
            P_nurbs = nurbs.evaluate(t);
            P_bspl  = bspl.evaluate(t);

            testCase.assertEqual(P_nurbs, P_bspl, 'AbsTol', 1e-10, ...
                'NURBS with unit weights should equal BSplineCurve.');
        end

        function test_clamped_endpoint_interpolation(testCase)
            % Must start at P1 and end at Pn regardless of weights.
            pts = [0 0; 1 3; 3 4; 5 2; 7 0];
            w = [1; 2; 0.5; 3; 1];

            c = OceanMath.Geometry.NURBSCurve(pts, w, 'Degree', 3);
            P0 = c.evaluate(0);
            P1 = c.evaluate(1);

            testCase.assertEqual(P0, [0 0], 'AbsTol', 1e-10);
            testCase.assertEqual(P1, [7 0], 'AbsTol', 1e-10);
        end

        function test_quarter_circle(testCase)
            % Degree-2 NURBS with specific weights = exact quarter circle.
            %
            % Control points for a 90-degree arc from (1,0) to (0,1):
            %   P0 = (1,0), w0 = 1
            %   P1 = (1,1), w1 = 1/sqrt(2)
            %   P2 = (0,1), w2 = 1
            pts = [1 0; 1 1; 0 1];
            w = [1; 1/sqrt(2); 1];

            c = OceanMath.Geometry.NURBSCurve(pts, w, 'Degree', 2);
            t = linspace(0, 1, 100)';
            P = c.evaluate(t);

            % Every point should lie on the unit circle: x^2 + y^2 = 1
            radii = sqrt(P(:,1).^2 + P(:,2).^2);
            testCase.assertEqual(radii, ones(100,1), 'AbsTol', 1e-6, ...
                'Quarter circle: all points should have radius 1.');

            % Check endpoints
            testCase.assertEqual(P(1,:), [1, 0], 'AbsTol', 1e-10);
            testCase.assertEqual(P(end,:), [0, 1], 'AbsTol', 1e-10);
        end

        function test_weight_attraction(testCase)
            % Higher weight pulls curve toward that control point.
            pts = [0 0; 3 5; 6 0];
            w_low  = [1; 1; 1];
            w_high = [1; 10; 1];

            c_low  = OceanMath.Geometry.NURBSCurve(pts, w_low,  'Degree', 2);
            c_high = OceanMath.Geometry.NURBSCurve(pts, w_high, 'Degree', 2);

            % At midpoint, high weight should pull curve closer to P2=[3,5]
            P_low  = c_low.evaluate(0.5);
            P_high = c_high.evaluate(0.5);

            testCase.assertGreaterThan(P_high(2), P_low(2), ...
                'Higher weight on P2 should pull curve upward.');
        end

        function test_tangent_vs_finite_diff(testCase)
            % Verify tangent against finite difference.
            pts = [0 0; 1 3; 3 4; 5 2; 7 0];
            w = [1; 2; 0.5; 3; 1];
            c = OceanMath.Geometry.NURBSCurve(pts, w, 'Degree', 3);

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
            pts = [0 0; 1 3; 3 4; 5 2; 7 0];
            w = [1; 2; 0.5; 3; 1];
            c = OceanMath.Geometry.NURBSCurve(pts, w, 'Degree', 3);

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

        function test_3d_nurbs(testCase)
            pts = [0 0 0; 1 1 1; 2 0 2; 3 1 0];
            w = [1; 2; 1; 1];
            c = OceanMath.Geometry.NURBSCurve(pts, w, 'Degree', 3);

            testCase.assertEqual(c.getDimension(), 3);
            P0 = c.evaluate(0);
            P1 = c.evaluate(1);
            testCase.assertEqual(P0, [0 0 0], 'AbsTol', 1e-10);
            testCase.assertEqual(P1, [3 1 0], 'AbsTol', 1e-10);
        end

        function test_iparametric_interface(testCase)
            pts = [0 0; 3 4; 6 0];
            w = [1; 1; 1];
            c = OceanMath.Geometry.NURBSCurve(pts, w, 'Degree', 2);
            testCase.assertTrue(isa(c, 'OceanMath.Geometry.IParametricCurve'));
        end

        function test_set_weight(testCase)
            pts = [0 0; 3 5; 6 0];
            w = [1; 1; 1];
            c = OceanMath.Geometry.NURBSCurve(pts, w, 'Degree', 2);

            P_before = c.evaluate(0.5);
            c.setWeight(2, 5);
            P_after = c.evaluate(0.5);

            % Increasing weight on P2 should pull curve toward P2
            testCase.assertGreaterThan(P_after(2), P_before(2));
        end

        function test_frenet_compatibility(testCase)
            % NURBSCurve should work with FrenetFrame.
            pts = [0 0 0; 1 2 1; 3 1 3; 5 3 0; 7 0 2];
            w = [1; 2; 1; 1.5; 1];
            c = OceanMath.Geometry.NURBSCurve(pts, w, 'Degree', 3);
            frenet = OceanMath.Geometry.FrenetFrame(c);

            [T, N, B] = frenet.frame(0.5);
            testCase.assertEqual(norm(T), 1, 'AbsTol', 1e-6);
            testCase.assertEqual(norm(N), 1, 'AbsTol', 1e-6);
            testCase.assertEqual(norm(B), 1, 'AbsTol', 1e-6);
            testCase.assertEqual(dot(T, N), 0, 'AbsTol', 1e-6);
        end

        function test_arc_length_quarter_circle(testCase)
            % Quarter circle arc length should be pi/2.
            pts = [1 0; 1 1; 0 1];
            w = [1; 1/sqrt(2); 1];
            c = OceanMath.Geometry.NURBSCurve(pts, w, 'Degree', 2);

            L = c.totalLength();
            testCase.assertEqual(L, pi/2, 'AbsTol', 1e-3, ...
                'Quarter circle arc length should be pi/2.');
        end
    end
end
