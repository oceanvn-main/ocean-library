classdef test_geometry < OceanTest.BaseTestCase
    % TEST_GEOMETRY  Unit tests for +OceanMath/+Geometry (Phase 1).
    %
    % Tests CatmullRomSpline and FrenetFrame with known geometric shapes.

    methods (Test)

        % ============================================================== %
        %  CatmullRomSpline — Basic 2D
        % ============================================================== %

        function test_evaluate_passes_through_control_points(testCase)
            % The spline must interpolate: P(0) ≈ first control point,
            % P(1) ≈ last control point.
            pts = [0 0; 1 2; 3 1; 5 3; 7 0];
            s = OceanMath.Geometry.CatmullRomSpline(pts);

            P0 = s.evaluate(0);
            P1 = s.evaluate(1);

            testCase.assertLessThan(max(abs(P0 - pts(1,:))), 1e-8, ...
                'Spline must pass through the first control point.');
            testCase.assertLessThan(max(abs(P1 - pts(end,:))), 1e-8, ...
                'Spline must pass through the last control point.');
        end

        function test_evaluate_at_internal_points(testCase)
            % For K control points, t = (i-1)/(K-1) should approximately
            % pass through control point i (for centripetal, very close).
            pts = [0 0; 2 4; 4 0];
            s = OceanMath.Geometry.CatmullRomSpline(pts);

            P_mid = s.evaluate(0.5);
            % Should be close to the middle control point [2, 4]
            testCase.assertLessThan(max(abs(P_mid - [2, 4])), 0.5, ...
                'Midpoint should be near the middle control point.');
        end

        function test_dimension_detection(testCase)
            pts2D = [0 0; 1 1; 2 0];
            s2D = OceanMath.Geometry.CatmullRomSpline(pts2D);
            testCase.assertEqual(s2D.getDimension(), 2);

            pts3D = [0 0 0; 1 1 1; 2 0 0];
            s3D = OceanMath.Geometry.CatmullRomSpline(pts3D);
            testCase.assertEqual(s3D.getDimension(), 3);
        end

        function test_too_few_points_errors(testCase)
            testCase.assertError( ...
                @() OceanMath.Geometry.CatmullRomSpline([1 2]), ...
                'OceanMath:Geometry:CatmullRomSpline:TooFewPoints');
        end

        % ============================================================== %
        %  CatmullRomSpline — Arc Length
        % ============================================================== %

        function test_arc_length_straight_line(testCase)
            % A straight line from [0,0] to [3,4] has exact length 5.
            pts = [0 0; 1 4/3; 2 8/3; 3 4];
            s = OceanMath.Geometry.CatmullRomSpline(pts);
            L = s.totalLength();
            testCase.assertLessThan(abs(L - 5), 0.15, ...
                'Straight-line arc length should be approximately 5.');
        end

        function test_arc_length_positive(testCase)
            pts = [0 0; 1 2; 3 1; 5 3; 7 0];
            s = OceanMath.Geometry.CatmullRomSpline(pts);
            L = s.totalLength();
            testCase.assertGreaterThan(L, 0, 'Arc length must be positive.');
        end

        % ============================================================== %
        %  CatmullRomSpline — Tangent
        % ============================================================== %

        function test_tangent_nonzero(testCase)
            pts = [0 0; 2 3; 5 1];
            s = OceanMath.Geometry.CatmullRomSpline(pts);
            T = s.tangent(0.5);
            testCase.assertGreaterThan(sqrt(sum(T.^2)), 1e-6, ...
                'Tangent at midpoint must be non-zero.');
        end

        function test_tangent_direction_straight_line(testCase)
            % For a line along [1,0], tangent should point in positive x.
            pts = [0 0; 1 0; 2 0; 3 0];
            s = OceanMath.Geometry.CatmullRomSpline(pts);
            T = s.tangent(0.5);
            direction = T / sqrt(sum(T.^2));
            testCase.assertLessThan(abs(direction(2)), 0.1, ...
                'Y-component of tangent should be near zero for horizontal line.');
            testCase.assertGreaterThan(direction(1), 0.5, ...
                'X-component of tangent should be positive for rightward line.');
        end

        % ============================================================== %
        %  CatmullRomSpline — Mutation
        % ============================================================== %

        function test_set_control_point(testCase)
            pts = [0 0; 1 1; 2 0];
            s = OceanMath.Geometry.CatmullRomSpline(pts);
            s.setControlPoint(2, [1, 3]);
            newPts = s.getControlPoints();
            testCase.assertEqual(newPts(2,:), [1, 3]);
        end

        function test_add_control_point(testCase)
            pts = [0 0; 1 1; 2 0];
            s = OceanMath.Geometry.CatmullRomSpline(pts);
            s.addControlPoint([4, 2]);
            testCase.assertEqual(size(s.getControlPoints(), 1), 4);
        end

        function test_remove_control_point(testCase)
            pts = [0 0; 1 1; 2 0; 3 1];
            s = OceanMath.Geometry.CatmullRomSpline(pts);
            s.removeControlPoint(2);
            testCase.assertEqual(size(s.getControlPoints(), 1), 3);
        end

        % ============================================================== %
        %  CatmullRomSpline — Arc Length Parameterization
        % ============================================================== %

        function test_arc_length_parameterization(testCase)
            pts = [0 0; 1 2; 3 1; 5 3; 7 0];
            s = OceanMath.Geometry.CatmullRomSpline(pts);
            tVals = s.parameterizeByArcLength(50);
            testCase.assertEqual(numel(tVals), 50);
            testCase.assertLessThan(abs(tVals(1)), 1e-8, 'First t must be 0.');
            testCase.assertLessThan(abs(tVals(end) - 1), 1e-8, 'Last t must be 1.');
        end

        % ============================================================== %
        %  CatmullRomSpline — 3D
        % ============================================================== %

        function test_3d_helix_positive_length(testCase)
            % Helical path in 3D
            nPts = 10;
            theta = linspace(0, 2*pi, nPts)';
            pts = [cos(theta), sin(theta), theta/(2*pi)];
            s = OceanMath.Geometry.CatmullRomSpline(pts);
            L = s.totalLength();
            testCase.assertGreaterThan(L, 2*pi*0.5, ...
                'Helix arc length should be substantial.');
        end

        % ============================================================== %
        %  FrenetFrame — 2D
        % ============================================================== %

        function test_frenet_2d_frame(testCase)
            pts = [0 0; 2 3; 5 1];
            s = OceanMath.Geometry.CatmullRomSpline(pts);
            frenet = OceanMath.Geometry.FrenetFrame(s);
            [T, N, ~] = frenet.frame(0.5);

            % T and N should be unit vectors
            testCase.assertLessThan(abs(sqrt(sum(T.^2)) - 1), 1e-6, ...
                'Tangent must be unit length.');
            testCase.assertLessThan(abs(sqrt(sum(N.^2)) - 1), 1e-6, ...
                'Normal must be unit length.');

            % T and N should be perpendicular
            dotTN = sum(T .* N);
            testCase.assertLessThan(abs(dotTN), 1e-6, ...
                'Tangent and Normal must be perpendicular.');
        end

        function test_frenet_2d_curvature_straight_line(testCase)
            pts = [0 0; 1 0; 2 0; 3 0];
            s = OceanMath.Geometry.CatmullRomSpline(pts);
            frenet = OceanMath.Geometry.FrenetFrame(s);
            kappa = frenet.curvature(0.5);
            testCase.assertLessThan(kappa, 0.1, ...
                'Curvature of a straight line should be near zero.');
        end

        function test_frenet_2d_quaternion(testCase)
            pts = [0 0; 2 0; 4 0];
            s = OceanMath.Geometry.CatmullRomSpline(pts);
            frenet = OceanMath.Geometry.FrenetFrame(s);
            q = frenet.quaternion(0.5);
            % Should be unit quaternion
            testCase.assertLessThan(abs(sqrt(sum(q.^2)) - 1), 1e-6, ...
                'Quaternion must be unit length.');
        end

        % ============================================================== %
        %  FrenetFrame — 3D
        % ============================================================== %

        function test_frenet_3d_frame_orthogonal(testCase)
            % Helical curve in 3D
            nPts = 8;
            theta = linspace(0, pi, nPts)';
            pts = [cos(theta), sin(theta), theta];
            s = OceanMath.Geometry.CatmullRomSpline(pts);
            frenet = OceanMath.Geometry.FrenetFrame(s);

            [T, N, B] = frenet.frame(0.5);

            % All unit length
            testCase.assertLessThan(abs(sqrt(sum(T.^2)) - 1), 1e-5);
            testCase.assertLessThan(abs(sqrt(sum(N.^2)) - 1), 1e-5);
            testCase.assertLessThan(abs(sqrt(sum(B.^2)) - 1), 1e-5);

            % All perpendicular
            testCase.assertLessThan(abs(sum(T.*N)), 1e-5, 'T . N = 0');
            testCase.assertLessThan(abs(sum(T.*B)), 1e-5, 'T . B = 0');
            testCase.assertLessThan(abs(sum(N.*B)), 1e-5, 'N . B = 0');
        end

        function test_frenet_3d_quaternion_unit(testCase)
            nPts = 6;
            theta = linspace(0, pi, nPts)';
            pts = [cos(theta), sin(theta), theta];
            s = OceanMath.Geometry.CatmullRomSpline(pts);
            frenet = OceanMath.Geometry.FrenetFrame(s);

            tSamples = linspace(0.1, 0.9, 5)';
            Q = frenet.quaternion(tSamples);
            for i = 1:size(Q, 1)
                testCase.assertLessThan(abs(sqrt(sum(Q(i,:).^2)) - 1), 1e-5, ...
                    'All quaternions must be unit length.');
            end
        end

        function test_frenet_3d_curvature_positive(testCase)
            nPts = 8;
            theta = linspace(0, pi, nPts)';
            pts = [cos(theta), sin(theta), theta];
            s = OceanMath.Geometry.CatmullRomSpline(pts);
            frenet = OceanMath.Geometry.FrenetFrame(s);
            kappa = frenet.curvature(0.5);
            testCase.assertGreaterThan(kappa, 0, ...
                'Curvature of a helix must be positive.');
        end

        % ============================================================== %
        %  Multiple evaluation points (vectorized)
        % ============================================================== %

        function test_batch_evaluate(testCase)
            pts = [0 0; 1 2; 3 1; 5 3; 7 0];
            s = OceanMath.Geometry.CatmullRomSpline(pts);
            tVals = linspace(0, 1, 50)';
            P = s.evaluate(tVals);
            testCase.assertEqual(size(P), [50, 2]);
        end

        function test_batch_frenet(testCase)
            pts = [0 0 0; 1 1 1; 2 0 0; 3 1 1];
            s = OceanMath.Geometry.CatmullRomSpline(pts);
            frenet = OceanMath.Geometry.FrenetFrame(s);
            tVals = linspace(0.1, 0.9, 20)';
            [T, N, B] = frenet.frame(tVals);
            testCase.assertEqual(size(T), [20, 3]);
            testCase.assertEqual(size(N), [20, 3]);
            testCase.assertEqual(size(B), [20, 3]);
        end
    end
end
