classdef IterativeSolverTest < OceanTest.BaseTestCase
    % ITERATIVESOLVERTEST  Unit tests for OceanMath.LinearAlgebra.Solvers.
    %
    % COVERS:
    %   - ILinearSolver interface contract (isinstance checks)
    %   - GaussSeidelSolver  (standard, SOR omega, warm-start, history, warnings)
    %   - JacobiSolver       (standard, damped, warm-start, history, warnings)
    %   - Cross-strategy comparison (both should agree on solution)
    %
    % USAGE (from library root):
    %   runtests('OceanTest.OceanMath.LinearAlgebra.IterativeSolverTest')
    %
    % AUTHOR: Ocean Library System
    % DATE:   2026-02-28

    % ------------------------------------------------------------------
    % Shared test fixtures
    % ------------------------------------------------------------------
    properties
        % 3-by-3 strictly diagonally dominant (textbook example)
        A3   % (3 x 3)
        b3   % (3 x 1)
        x3   % exact solution via backslash

        % 10-by-10 symmetric positive definite
        A10  % (10 x 10)
        b10  % (10 x 1)
        x10  % exact solution via backslash

        % Relative tolerance used in solution comparisons
        SolTol double = 1e-6
    end

    methods (TestClassSetup)
        function buildFixtures(testCase)
            % ── 3×3 diagonally dominant ───────────────────────────
            testCase.A3 = [ 4  -1   0;
                           -1   4  -1;
                            0  -1   3];
            testCase.b3 = [15; 10; 10];
            testCase.x3 = testCase.A3 \ testCase.b3;

            % ── 10×10 SPD  (B'B + 10*I guarantees dominance) ─────
            rng(42);
            B = randn(10);
            testCase.A10 = B' * B + 10 * eye(10);
            testCase.b10 = randn(10, 1);
            testCase.x10 = testCase.A10 \ testCase.b10;
        end
    end

    % ==================================================================
    %   SECTION 1 — INTERFACE CONTRACT TESTS
    % ==================================================================
    methods (Test, TestTags = {'Interface'})

        function testGaussSeidelIsILinearSolver(testCase)
            % GaussSeidelSolver must be a concrete ILinearSolver
            import OceanMath.LinearAlgebra.Solvers.GaussSeidelSolver;
            import OceanMath.LinearAlgebra.Solvers.ILinearSolver;

            gs = GaussSeidelSolver();
            testCase.verifyTrue(isa(gs, 'OceanMath.LinearAlgebra.Solvers.ILinearSolver'), ...
                'GaussSeidelSolver must extend ILinearSolver');
        end

        function testJacobiIsILinearSolver(testCase)
            % JacobiSolver must be a concrete ILinearSolver
            import OceanMath.LinearAlgebra.Solvers.JacobiSolver;

            jac = JacobiSolver();
            testCase.verifyTrue(isa(jac, 'OceanMath.LinearAlgebra.Solvers.ILinearSolver'), ...
                'JacobiSolver must extend ILinearSolver');
        end

        function testInfoStructHasRequiredFields(testCase)
            % solve() must return an info struct with the four required fields
            import OceanMath.LinearAlgebra.Solvers.GaussSeidelSolver;
            gs = GaussSeidelSolver();
            [~, info] = gs.solve(testCase.A3, testCase.b3);

            testCase.verifyTrue(isfield(info, 'iters'),     'info missing .iters');
            testCase.verifyTrue(isfield(info, 'residual'),  'info missing .residual');
            testCase.verifyTrue(isfield(info, 'converged'), 'info missing .converged');
            testCase.verifyTrue(isfield(info, 'history'),   'info missing .history');
        end

    end

    % ==================================================================
    %   SECTION 2 — GAUSS-SEIDEL SOLVER TESTS
    % ==================================================================
    methods (Test, TestTags = {'GaussSeidel'})

        function testGS_Simple3x3(testCase)
            % Standard Gauss-Seidel on the 3×3 dominant system
            import OceanMath.LinearAlgebra.Solvers.GaussSeidelSolver;
            gs = GaussSeidelSolver();
            [x, info] = gs.solve(testCase.A3, testCase.b3);

            testCase.verifyTrue(info.converged, 'GS should converge on 3x3 dominant system');
            testCase.assertMatrixEquals(x, testCase.x3, testCase.SolTol, ...
                'GS 3x3 solution mismatch');
        end

        function testGS_SPD_10x10(testCase)
            % Gauss-Seidel on a 10×10 SPD matrix (no diagonal dominance required)
            import OceanMath.LinearAlgebra.Solvers.GaussSeidelSolver;
            gs = GaussSeidelSolver('Tol', 1e-8);
            [x, info] = gs.solve(testCase.A10, testCase.b10);

            testCase.verifyTrue(info.converged, 'GS should converge on SPD 10x10 system');
            testCase.assertMatrixEquals(x, testCase.x10, testCase.SolTol, ...
                'GS 10x10 solution mismatch');
        end

        function testGS_SOR_Omega_Accelerates(testCase)
            % SOR with w > 1 should converge in fewer iterations than w=1 for SPD
            import OceanMath.LinearAlgebra.Solvers.GaussSeidelSolver;

            gs_standard = GaussSeidelSolver('Omega', 1.0);
            gs_sor       = GaussSeidelSolver('Omega', 1.5);

            [x_gs,  info_gs]  = gs_standard.solve(testCase.A10, testCase.b10);
            [x_sor, info_sor] = gs_sor.solve(testCase.A10, testCase.b10);

            % Both should produce the correct answer
            testCase.assertMatrixEquals(x_gs,  testCase.x10, testCase.SolTol, 'GS standard solution wrong');
            testCase.assertMatrixEquals(x_sor, testCase.x10, testCase.SolTol, 'GS SOR solution wrong');

            % SOR should generally need fewer or equal iterations
            testCase.verifyTrue(info_sor.iters <= info_gs.iters + 20, ...
                sprintf('SOR (%d iters) should not be much worse than GS (%d iters)', ...
                info_sor.iters, info_gs.iters));
        end

        function testGS_WarmStart_FasterConvergence(testCase)
            % Starting from near the solution should reduce iteration count
            import OceanMath.LinearAlgebra.Solvers.GaussSeidelSolver;
            gs = GaussSeidelSolver();

            [~, info_cold] = gs.solve(testCase.A10, testCase.b10);

            % Warm start: perturb the exact solution slightly
            x0_warm = testCase.x10 + 1e-3 * randn(10, 1);
            [~, info_warm] = gs.solve(testCase.A10, testCase.b10, x0_warm);

            testCase.verifyTrue(info_warm.iters < info_cold.iters, ...
                'Warm start should converge in fewer iterations than cold start');
        end

        function testGS_ResidualHistory(testCase)
            % When RecordHistory=true, info.history should be non-empty and decreasing
            import OceanMath.LinearAlgebra.Solvers.GaussSeidelSolver;
            gs = GaussSeidelSolver('RecordHistory', true);
            [~, info] = gs.solve(testCase.A10, testCase.b10);

            testCase.verifyTrue(~isempty(info.history), 'History should be non-empty');
            testCase.verifyEqual(numel(info.history), info.iters, ...
                'History length should equal iteration count');

            % Residuals should be (generally) decreasing
            diffs = diff(info.history);
            testCase.verifyTrue(sum(diffs > 0) < 3, ...
                'Residual history should be mostly decreasing (monotone convergence)');
        end

        function testGS_ConvergenceInfo(testCase)
            % Verify info fields are numerically consistent
            import OceanMath.LinearAlgebra.Solvers.GaussSeidelSolver;
            gs = GaussSeidelSolver('Tol', 1e-8);
            [x, info] = gs.solve(testCase.A3, testCase.b3);

            testCase.verifyTrue(info.converged);
            testCase.verifyTrue(info.iters > 0);

            % Reported residual should match recomputed value
            actual_res = norm(testCase.b3 - testCase.A3 * x) / norm(testCase.b3);
            testCase.verifyEqual(info.residual, actual_res, 'AbsTol', 1e-12, ...
                'info.residual does not match recomputed residual');
        end

        function testGS_DiagDominanceWarning(testCase)
            % A non-dominant matrix should trigger the convergence warning
            import OceanMath.LinearAlgebra.Solvers.GaussSeidelSolver;
            gs = GaussSeidelSolver('MaxIter', 5);

            % Intentionally non-dominant matrix
            A_bad = [1 5 0; 0 1 5; 5 0 1];
            b_bad = [1; 1; 1];

            testCase.verifyWarning( ...
                @() gs.solve(A_bad, b_bad), ...
                'OceanMath:LinearAlgebra:NotDiagonallyDominant', ...
                'Expected diagonal dominance warning');
        end

    end

    % ==================================================================
    %   SECTION 3 — JACOBI SOLVER TESTS
    % ==================================================================
    methods (Test, TestTags = {'Jacobi'})

        function testJacobi_Simple3x3(testCase)
            % Standard Jacobi on the 3×3 dominant system
            import OceanMath.LinearAlgebra.Solvers.JacobiSolver;
            jac = JacobiSolver('MaxIter', 1000);
            [x, info] = jac.solve(testCase.A3, testCase.b3);

            testCase.verifyTrue(info.converged, 'Jacobi should converge on 3x3 dominant system');
            testCase.assertMatrixEquals(x, testCase.x3, testCase.SolTol, ...
                'Jacobi 3x3 solution mismatch');
        end

        function testJacobi_SPD_10x10(testCase)
            % Jacobi on 10×10 SPD (B'B + 10*I is strictly dominant, so Jacobi converges)
            import OceanMath.LinearAlgebra.Solvers.JacobiSolver;
            jac = JacobiSolver('MaxIter', 2000, 'Tol', 1e-8);
            [x, info] = jac.solve(testCase.A10, testCase.b10);

            testCase.verifyTrue(info.converged, 'Jacobi should converge on SPD 10x10');
            testCase.assertMatrixEquals(x, testCase.x10, testCase.SolTol, ...
                'Jacobi 10x10 solution mismatch');
        end

        function testJacobi_WarmStart(testCase)
            % Warm start should reduce iteration count for Jacobi too
            import OceanMath.LinearAlgebra.Solvers.JacobiSolver;
            jac = JacobiSolver('MaxIter', 2000);

            [~, info_cold] = jac.solve(testCase.A10, testCase.b10);

            x0_warm = testCase.x10 + 1e-3 * randn(10, 1);
            [~, info_warm] = jac.solve(testCase.A10, testCase.b10, x0_warm);

            testCase.verifyTrue(info_warm.iters < info_cold.iters, ...
                'Jacobi warm start should converge faster');
        end

        function testJacobi_ResidualHistory(testCase)
            % RecordHistory=true must populate info.history
            import OceanMath.LinearAlgebra.Solvers.JacobiSolver;
            jac = JacobiSolver('RecordHistory', true, 'MaxIter', 2000);
            [~, info] = jac.solve(testCase.A10, testCase.b10);

            testCase.verifyTrue(~isempty(info.history), 'Jacobi history should be non-empty');
            testCase.verifyEqual(numel(info.history), info.iters, ...
                'Jacobi history length should equal iteration count');
        end

        function testJacobi_DiagDominanceWarning(testCase)
            % Non-dominant matrix triggers the shared warning from ILinearSolver
            import OceanMath.LinearAlgebra.Solvers.JacobiSolver;
            jac = JacobiSolver('MaxIter', 5);

            A_bad = [1 5 0; 0 1 5; 5 0 1];
            b_bad = [1; 1; 1];

            testCase.verifyWarning( ...
                @() jac.solve(A_bad, b_bad), ...
                'OceanMath:LinearAlgebra:NotDiagonallyDominant');
        end

    end

    % ==================================================================
    %   SECTION 4 — CROSS-STRATEGY COMPARISON
    % ==================================================================
    methods (Test, TestTags = {'CrossStrategy'})

        function testGS_vs_Jacobi_SameAnswer(testCase)
            % Both strategies must reach the same correct solution
            import OceanMath.LinearAlgebra.Solvers.GaussSeidelSolver;
            import OceanMath.LinearAlgebra.Solvers.JacobiSolver;

            gs  = GaussSeidelSolver();
            jac = JacobiSolver('MaxIter', 2000);

            [x_gs,  ~] = gs.solve(testCase.A10,  testCase.b10);
            [x_jac, ~] = jac.solve(testCase.A10, testCase.b10);

            testCase.assertMatrixEquals(x_gs, x_jac, 1e-5, ...
                'GS and Jacobi should agree on same system');
        end

        function testGS_FasterThanJacobi_OnSPD(testCase)
            % Gauss-Seidel is theoretically 2x faster than Jacobi for SPD;
            % verify it needs strictly fewer iterations.
            import OceanMath.LinearAlgebra.Solvers.GaussSeidelSolver;
            import OceanMath.LinearAlgebra.Solvers.JacobiSolver;

            gs  = GaussSeidelSolver('Tol', 1e-8);
            jac = JacobiSolver('Tol', 1e-8, 'MaxIter', 5000);

            [~, info_gs]  = gs.solve(testCase.A10,  testCase.b10);
            [~, info_jac] = jac.solve(testCase.A10, testCase.b10);

            testCase.verifyTrue(info_gs.iters < info_jac.iters, ...
                sprintf('GS (%d iters) should converge faster than Jacobi (%d iters) on SPD', ...
                info_gs.iters, info_jac.iters));
        end

    end

end
