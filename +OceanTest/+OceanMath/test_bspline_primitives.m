classdef test_bspline_primitives < matlab.unittest.TestCase
    % TEST_BSPLINE_PRIMITIVES  Tests for KnotVector and BSplineBasis.

    % ============================================================== %
    %  KnotVector Tests
    % ============================================================== %

    methods (Test)

        function test_clamped_length(testCase)
            % Clamped knot vector length = n + p + 1
            for n = [4, 6, 8]
                for p = [1, 2, 3]
                    k = OceanMath.Geometry.KnotVector.clamped(n, p);
                    testCase.assertEqual(numel(k), n + p + 1, ...
                        sprintf('Wrong length for n=%d, p=%d.', n, p));
                end
            end
        end

        function test_clamped_boundary_multiplicity(testCase)
            % First (p+1) knots = 0, last (p+1) knots = 1
            n = 6; p = 3;
            k = OceanMath.Geometry.KnotVector.clamped(n, p);
            testCase.assertEqual(k(1:p+1), zeros(1, p+1));
            testCase.assertEqual(k(end-p:end), ones(1, p+1));
        end

        function test_clamped_non_decreasing(testCase)
            k = OceanMath.Geometry.KnotVector.clamped(8, 3);
            testCase.assertTrue(all(diff(k) >= 0));
        end

        function test_uniform_length(testCase)
            n = 5; p = 3;
            k = OceanMath.Geometry.KnotVector.uniform(n, p);
            testCase.assertEqual(numel(k), n + p + 1);
        end

        function test_uniform_equally_spaced(testCase)
            k = OceanMath.Geometry.KnotVector.uniform(5, 3);
            diffs = diff(k);
            testCase.assertEqual(diffs, repmat(diffs(1), 1, numel(diffs)), ...
                'AbsTol', 1e-12);
        end

        function test_open_is_clamped(testCase)
            n = 6; p = 3;
            k1 = OceanMath.Geometry.KnotVector.clamped(n, p);
            k2 = OceanMath.Geometry.KnotVector.open(n, p);
            testCase.assertEqual(k1, k2);
        end

        function test_from_interior_knots(testCase)
            k = OceanMath.Geometry.KnotVector.fromInteriorKnots([0.25, 0.5, 0.75], 3);
            testCase.assertEqual(k(1:4), [0 0 0 0]);
            testCase.assertEqual(k(5:7), [0.25 0.5 0.75]);
            testCase.assertEqual(k(8:11), [1 1 1 1]);
        end

        function test_validate_good(testCase)
            k = OceanMath.Geometry.KnotVector.clamped(6, 3);
            testCase.assertTrue(OceanMath.Geometry.KnotVector.validate(k, 6, 3));
        end

        function test_validate_bad_length(testCase)
            k = [0 0 0 1 1 1];  % too short for n=6, p=3
            testCase.assertFalse(OceanMath.Geometry.KnotVector.validate(k, 6, 3));
        end

        function test_validate_decreasing(testCase)
            k = [0 0 0 0 0.5 0.3 1 1 1 1];  % not non-decreasing
            testCase.assertFalse(OceanMath.Geometry.KnotVector.validate(k, 6, 3));
        end

        function test_findspans(testCase)
            k = OceanMath.Geometry.KnotVector.clamped(5, 2);
            % k = [0 0 0 0.333 0.667 1 1 1] (approx)
            spans = OceanMath.Geometry.KnotVector.findSpans(k, 2, [0; 0.5; 1]);
            % All spans should be valid indices
            testCase.assertTrue(all(spans >= 1));
            testCase.assertTrue(all(spans <= numel(k)));
        end

        % ============================================================== %
        %  BSplineBasis Tests
        % ============================================================== %

        function test_bspline_partition_of_unity(testCase)
            % All B-spline basis values must sum to 1 for clamped knots.
            n = 6; p = 3;
            k = OceanMath.Geometry.KnotVector.clamped(n, p);
            t = linspace(0, 1, 50)';
            B = OceanMath.Geometry.BSplineBasis.evaluate(k, p, t);

            rowSums = sum(B, 2);
            testCase.assertEqual(rowSums, ones(50, 1), 'AbsTol', 1e-12, ...
                'Partition of unity failed for B-spline basis.');
        end

        function test_bspline_non_negative(testCase)
            % B-spline basis values are always >= 0.
            k = OceanMath.Geometry.KnotVector.clamped(6, 3);
            t = linspace(0, 1, 100)';
            B = OceanMath.Geometry.BSplineBasis.evaluate(k, 3, t);
            testCase.assertTrue(all(B(:) >= -1e-15), ...
                'B-spline basis values should be non-negative.');
        end

        function test_bspline_output_size(testCase)
            % Output should be [N x n] where n = m - p - 1.
            n = 7; p = 3;
            k = OceanMath.Geometry.KnotVector.clamped(n, p);
            t = linspace(0, 1, 30)';
            B = OceanMath.Geometry.BSplineBasis.evaluate(k, p, t);
            testCase.assertEqual(size(B), [30, n]);
        end

        function test_bspline_degree0_step(testCase)
            % Degree 0: piecewise constant (step functions).
            k = [0 0.25 0.5 0.75 1];
            B = OceanMath.Geometry.BSplineBasis.evaluate(k, 0, [0.1; 0.3; 0.6; 0.9]);
            % t=0.1 in [0, 0.25), t=0.3 in [0.25, 0.5), etc.
            testCase.assertEqual(B(1,:), [1 0 0 0], 'AbsTol', 1e-12);
            testCase.assertEqual(B(2,:), [0 1 0 0], 'AbsTol', 1e-12);
            testCase.assertEqual(B(3,:), [0 0 1 0], 'AbsTol', 1e-12);
            testCase.assertEqual(B(4,:), [0 0 0 1], 'AbsTol', 1e-12);
        end

        function test_bspline_degree1_linear(testCase)
            % Degree 1 clamped: piecewise linear hat functions.
            k = OceanMath.Geometry.KnotVector.clamped(3, 1);
            % k = [0 0 0.5 1 1]
            t = [0; 0.25; 0.5; 0.75; 1];
            B = OceanMath.Geometry.BSplineBasis.evaluate(k, 1, t);
            % At t=0: N1(0)=1, N2(0)=0, N3(0)=0
            testCase.assertEqual(B(1,:), [1 0 0], 'AbsTol', 1e-12);
            % At t=0.5: midpoint
            testCase.assertEqual(B(3,1), 0, 'AbsTol', 1e-12);
            testCase.assertEqual(B(3,2), 1, 'AbsTol', 1e-12);
            testCase.assertEqual(B(3,3), 0, 'AbsTol', 1e-12);
            % At t=1: N3(1)=1
            testCase.assertEqual(B(5,:), [0 0 1], 'AbsTol', 1e-12);
        end

        function test_bspline_consistency_across_degrees(testCase)
            % Verify that increasing degree produces smoother basis.
            % Degree 0 is piecewise constant, degree 3 is smooth cubic.
            k1 = OceanMath.Geometry.KnotVector.clamped(6, 1);
            k3 = OceanMath.Geometry.KnotVector.clamped(6, 3);
            t = linspace(0, 1, 100)';

            B1 = OceanMath.Geometry.BSplineBasis.evaluate(k1, 1, t);
            B3 = OceanMath.Geometry.BSplineBasis.evaluate(k3, 3, t);

            % Both should have 6 basis functions (n=6)
            testCase.assertEqual(size(B1, 2), 6);
            testCase.assertEqual(size(B3, 2), 6);

            % Both should partition unity
            testCase.assertEqual(sum(B1, 2), ones(100, 1), 'AbsTol', 1e-12);
            testCase.assertEqual(sum(B3, 2), ones(100, 1), 'AbsTol', 1e-12);
        end

        function test_bspline_derivative_vs_fd(testCase)
            % Verify derivative against finite difference.
            k = OceanMath.Geometry.KnotVector.clamped(6, 3);
            t = linspace(0.05, 0.95, 20)';
            dt = 1e-7;

            dB = OceanMath.Geometry.BSplineBasis.derivative(k, 3, t);
            B_plus  = OceanMath.Geometry.BSplineBasis.evaluate(k, 3, t + dt);
            B_minus = OceanMath.Geometry.BSplineBasis.evaluate(k, 3, t - dt);
            dB_fd = (B_plus - B_minus) / (2 * dt);

            testCase.assertEqual(dB, dB_fd, 'AbsTol', 1e-4, ...
                'Derivative should match finite difference.');
        end

        function test_basisMatrix_alias(testCase)
            % basisMatrix should be identical to evaluate.
            k = OceanMath.Geometry.KnotVector.clamped(5, 3);
            t = linspace(0, 1, 15)';
            B1 = OceanMath.Geometry.BSplineBasis.evaluate(k, 3, t);
            B2 = OceanMath.Geometry.BSplineBasis.basisMatrix(t, k, 3);
            testCase.assertEqual(B1, B2);
        end
    end
end
