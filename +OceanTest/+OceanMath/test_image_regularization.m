classdef test_image_regularization < OceanTest.BaseTestCase
    % TEST_IMAGE_REGULARIZATION  Unit tests for Szeliski §4.2 Robust Regularization.
    %
    %   7 test methods, each cross-validated against MATLAB built-in functions
    %   so that correctness is provable, not just plausible.
    %
    %   Run via: OceanTest.TestRunner.run('test_image_regularization')
    %
    %   Test inventory:
    %     1. testDifferenceMatrix          — D * f(:) vs MATLAB diff()
    %     2. testPenaltyValues             — evaluate() vs closed-form operators
    %     3. testIRLSWeightFunction        — weight() vs rho(d)/d^2 + isfinite at d=0
    %     4. testL2TikhonovVsDirectSolve   — IRLS(Quadratic) vs MATLAB backslash
    %     5. testIRLSvsPrimalDualEquiv     — IRLS-TV == Primal-Dual-TV
    %     6. testPrimalDualStepCondition   — tau*sigma*normest(D)^2 <= 1
    %     7. testGNCLimitEqualsQuadratic   — GNC(V→inf) == Quadratic
    %
    %   Author: Ocean Library  |  2026-08-05

    % ── Shared aliases ────────────────────────────────────────────────────────
    methods (TestMethodSetup)
        function importPkg(~)
            % Nothing needed — using full package paths for clarity.
        end
    end

    % ═════════════════════════════════════════════════════════════════════════
    % Test 1: Difference Matrix vs MATLAB diff()
    % ═════════════════════════════════════════════════════════════════════════
    methods (Test)
        function testDifferenceMatrix(testCase)
            % Verify that D * f(:) produces the SAME result as
            %   [diff(f,1,2)(:);  diff(f,1,1)(:)]
            % for a random image.  This is the foundation: if D is wrong,
            % every solver is wrong.

            rng(1);
            rows = 6;  cols = 8;
            f = rand(rows, cols);

            % ── MATLAB ground truth ──────────────────────────────────────────
            dx_ref = diff(f, 1, 2);        % horizontal: (rows x cols-1)
            dy_ref = diff(f, 1, 1);        % vertical:   (rows-1 x cols)
            ref    = [dx_ref(:); dy_ref(:)];

            % ── Our implementation ───────────────────────────────────────────
            pen    = OceanMath.Regularization.QuadraticPenalty();
            energy = OceanMath.Image.Variational.ImageRegularizationEnergy(pen);
            D      = energy.buildDifferenceMatrix(rows, cols);
            out    = D * f(:);

            testCase.verifyEqual(out, ref, 'AbsTol', 1e-12, ...
                ['buildDifferenceMatrix result must exactly match ' ...
                 '[diff(f,1,2)(:); diff(f,1,1)(:)].']);
        end

        % ═════════════════════════════════════════════════════════════════════
        % Test 2: Penalty evaluate() vs closed-form MATLAB operators
        % ═════════════════════════════════════════════════════════════════════
        function testPenaltyValues(testCase)
            % Each penalty.evaluate(d) must equal the textbook formula
            % computed directly with MATLAB array operators.

            d = linspace(-3, 3, 201)';

            % 2a — Quadratic
            pen = OceanMath.Regularization.QuadraticPenalty();
            testCase.verifyEqual(pen.evaluate(d), d.^2, 'AbsTol', 1e-12, ...
                'QuadraticPenalty.evaluate must equal d.^2');

            % 2b — L1
            pen = OceanMath.Regularization.L1Penalty();
            testCase.verifyEqual(pen.evaluate(d), abs(d), 'AbsTol', 1e-12, ...
                'L1Penalty.evaluate must equal abs(d)');

            % 2c — PNorm (p=0.8)
            pen = OceanMath.Regularization.PNormPenalty('P', 0.8);
            testCase.verifyEqual(pen.evaluate(d), abs(d).^0.8, 'AbsTol', 1e-12, ...
                'PNormPenalty(p=0.8).evaluate must equal abs(d).^0.8');

            % 2d — Huber (sigma=1)
            sigma = 1.0;
            pen   = OceanMath.Regularization.HuberPenalty('Sigma', sigma);
            ad    = abs(d);
            ref_huber = 0.5 .* d.^2 .* (ad <= sigma) + ...
                        sigma .* (ad - 0.5.*sigma) .* (ad > sigma);
            testCase.verifyEqual(pen.evaluate(d), ref_huber, 'AbsTol', 1e-12, ...
                'HuberPenalty(sigma=1).evaluate must match piecewise formula');

            % 2e — GNC (V=4)
            V   = 4;
            pen = OceanMath.Regularization.GNCPenalty('V', V);
            testCase.verifyEqual(pen.evaluate(d), min(d.^2, V), 'AbsTol', 1e-12, ...
                'GNCPenalty(V=4).evaluate must equal min(d.^2, 4)');
        end

        % ═════════════════════════════════════════════════════════════════════
        % Test 3: IRLS weight function — ratio and finite-at-zero
        % ═════════════════════════════════════════════════════════════════════
        function testIRLSWeightFunction(testCase)
            % For each penalty, verify:
            %   (a) weight(d) is finite at d = 0
            %   (b) at d = [0.5, 1.0, 2.0], weight = derivative(d) ./ d
            %       This is the weight for which the IRLS fixed point of
            %       (I + lambda*D'WD) f = g is a stationary point of
            %       (1/2)||f-g||^2 + lambda*sum rho(Df).

            d_test = [0.5; 1.0; 2.0];

            penalties = { ...
                OceanMath.Regularization.QuadraticPenalty(), ...
                OceanMath.Regularization.L1Penalty(), ...
                OceanMath.Regularization.PNormPenalty('P', 0.8), ...
                OceanMath.Regularization.HuberPenalty('Sigma', 0.5), ...
                OceanMath.Regularization.GNCPenalty('V', 2.0) ...
            };

            for i = 1:numel(penalties)
                pen  = penalties{i};
                name = pen.getName();

                % (a) No division-by-zero at d = 0
                testCase.verifyTrue(isfinite(pen.weight(0)), ...
                    sprintf('%s: weight(0) must be finite (no division-by-zero)', name));

                % (b) At non-zero points, weight = rho'(d)/d
                w_method = pen.weight(d_test);
                w_ratio  = pen.derivative(d_test) ./ d_test;

                testCase.verifyEqual(w_method, w_ratio, 'AbsTol', 1e-6, ...
                    sprintf('%s: weight(d) must equal derivative(d)/d for d~=0', name));
            end
        end

        % ═════════════════════════════════════════════════════════════════════
        % Test 4: IRLS with QuadraticPenalty == MATLAB backslash solve
        %         PRIMARY MATLAB built-in cross-check
        % ═════════════════════════════════════════════════════════════════════
        function testL2TikhonovVsDirectSolve(testCase)
            % With QuadraticPenalty rho(d) = d^2, every IRLS weight = 2, so the
            % first (and only) iteration exactly solves the normal equations of
            % (1/2)||f-g||^2 + lambda*||Df||^2:
            %   (I + 2 * lambda * D'D) f = g
            % This must match MATLAB's \ operator on the same system.

            rng(42);
            rows   = 8;  cols = 8;
            g      = rand(rows, cols);
            lambda = 0.1;

            % ── MATLAB ground truth ──────────────────────────────────────────
            pen    = OceanMath.Regularization.QuadraticPenalty();
            energy = OceanMath.Image.Variational.ImageRegularizationEnergy(pen, ...
                         'Lambda', lambda);
            N = rows * cols;
            D = energy.buildDifferenceMatrix(rows, cols);
            LHS = speye(N) + 2 .* lambda .* (D' * D);
            RHS = g(:);
            f_ref = reshape(LHS \ RHS, rows, cols);

            % ── Our IRLS solver ──────────────────────────────────────────────
            solver = OceanMath.Image.Variational.Solvers.IRLSSolver( ...
                         'MaxIter', 50, 'Tol', 1e-12);
            f_ours = solver.solve(energy, g);

            testCase.verifyEqual(f_ours, f_ref, 'AbsTol', 1e-8, ...
                ['IRLSSolver with QuadraticPenalty must match direct MATLAB ' ...
                 'backslash solve of (I + 2*lambda*D''D)f = g']);
        end

        % ═════════════════════════════════════════════════════════════════════
        % Test 5: IRLS-TV and Primal-Dual-TV converge to the same minimum
        % ═════════════════════════════════════════════════════════════════════
        function testIRLSvsPrimalDualEquivalence(testCase)
            % For a convex penalty (L1/TV), both solvers must reach the same
            % global minimiser.  If they disagree, one is incorrect.

            rng(0);
            rows   = 12;  cols = 12;
            g      = rand(rows, cols);
            lambda = 0.2;

            pen    = OceanMath.Regularization.L1Penalty();
            energy = OceanMath.Image.Variational.ImageRegularizationEnergy(pen, ...
                         'Lambda', lambda);

            % IRLS with many iterations (tight convergence). IRLS on TV
            % converges slowly: 300 iterations leave a 4e-4 gap to the
            % minimiser, 2000 bring it to about 3e-6.
            solver_irls = OceanMath.Image.Variational.Solvers.IRLSSolver( ...
                              'MaxIter', 2000, 'Tol', 1e-13);
            f_irls = solver_irls.solve(energy, g);

            % Primal-Dual with many iterations (tight convergence)
            solver_pd = OceanMath.Image.Variational.Solvers.PrimalDualSolver( ...
                            'MaxIter', 3000, 'Tol', 1e-9);
            f_pd = solver_pd.solve(energy, g);

            testCase.verifyEqual(f_irls, f_pd, 'AbsTol', 1e-4, ...
                ['IRLS-TV and Primal-Dual-TV must converge to the same minimum. ' ...
                 'Max discrepancy = ' num2str(max(abs(f_irls(:) - f_pd(:))), '%.2e')]);
        end

        % ═════════════════════════════════════════════════════════════════════
        % Test 6: Primal-Dual step sizes satisfy convergence condition
        % ═════════════════════════════════════════════════════════════════════
        function testPrimalDualStepCondition(testCase)
            % Chambolle-Pock converges iff tau * sigma * ||D||^2 <= 1.
            % Verify with MATLAB's normest().

            rows = 10;  cols = 10;
            pen    = OceanMath.Regularization.L1Penalty();
            energy = OceanMath.Image.Variational.ImageRegularizationEnergy(pen, ...
                         'Lambda', 0.1);

            % Run a minimal solve to trigger auto-computation of Tau and Sigma
            solver = OceanMath.Image.Variational.Solvers.PrimalDualSolver( ...
                         'MaxIter', 2);
            rng(7);
            solver.solve(energy, rand(rows, cols));

            % Retrieve the auto-set step sizes
            tau   = solver.Tau;
            sigma = solver.Sigma;

            % Exact spectral norm (the matrix is small): an estimate could
            % undershoot ||D||_2 and hide a violated condition.
            D        = energy.buildDifferenceMatrix(rows, cols);
            normD_sq = norm(full(D))^2;

            condition_value = tau * sigma * normD_sq;

            testCase.verifyTrue(condition_value <= 1.0 + 1e-10, ...
                sprintf(['PrimalDualSolver step sizes violate convergence condition: ' ...
                         'tau*sigma*||D||^2 = %.6f > 1'], condition_value));
        end

        % ═════════════════════════════════════════════════════════════════════
        % Test 7: GNCPenalty(V→inf) == QuadraticPenalty
        % ═════════════════════════════════════════════════════════════════════
        function testGNCLimitEqualsQuadratic(testCase)
            % As V → inf, GNCPenalty must behave identically to QuadraticPenalty
            % because the clamp min(d^2, V) never activates for finite d.

            d = linspace(-2, 2, 200)';

            gnc  = OceanMath.Regularization.GNCPenalty('V', 1e8);
            quad = OceanMath.Regularization.QuadraticPenalty();

            testCase.verifyEqual(gnc.evaluate(d), quad.evaluate(d), 'AbsTol', 1e-6, ...
                'GNCPenalty(V=1e8).evaluate must match QuadraticPenalty for |d| <= 2');

            % Also verify solver output matches for a small test image
            rng(3);
            rows = 6; cols = 6;
            g  = rand(rows, cols);

            % GNC with very large V (one scale, no true annealing)
            pen_gnc    = OceanMath.Regularization.GNCPenalty('V', 1e8);
            energy_gnc = OceanMath.Image.Variational.ImageRegularizationEnergy(pen_gnc, ...
                             'Lambda', 0.1);
            solver_gnc = OceanMath.Image.Variational.Solvers.GNCSolver( ...
                             'NumScales', 1, 'VMax', 1e8);
            f_gnc = solver_gnc.solve(energy_gnc, g);

            % Quadratic IRLS (exact 1-iteration solve)
            pen_q    = OceanMath.Regularization.QuadraticPenalty();
            energy_q = OceanMath.Image.Variational.ImageRegularizationEnergy(pen_q, ...
                           'Lambda', 0.1);
            solver_q = OceanMath.Image.Variational.Solvers.IRLSSolver( ...
                           'MaxIter', 10, 'Tol', 1e-12);
            f_quad = solver_q.solve(energy_q, g);

            testCase.verifyEqual(f_gnc, f_quad, 'AbsTol', 1e-4, ...
                'GNCSolver with V→inf must match IRLSSolver with QuadraticPenalty');
        end
    end
end
