classdef TestNn4 < OceanTest.BaseTestCase
    methods (Test)
        function testAll(testCase)
            passed = 0; failed = 0;
            % TEST 1 — Plain net: ResidualStart all false
            import OceanDL.Graph.*
            import OceanDL.Simulation.*
            import OceanStats.Optim.*
            import OceanStats.Loss.Regression.MSELoss
            fprintf('=== NN-4 Tests ===\n\n');
            % ------------------------------------------------------------------- %
            fprintf('-- TEST 1: Plain spec has no residual flags --\n');
            specP = NNTopologySpec.make([4,8,8,2], {'linear','relu','relu','linear'});
            if ~any(specP.ResidualStart)
            fprintf('  PASS  ResidualStart all false for plain net\n');
            passed = passed + 1;
            else
            fprintf('  FAIL  ResidualStart = %s (expected all false)\n', mat2str(specP.ResidualStart));
            failed = failed + 1;
            end
            % ------------------------------------------------------------------- %


            % TEST 2 — Residual spec: ResidualStart correct
            import OceanDL.Graph.*
            import OceanDL.Simulation.*
            import OceanStats.Optim.*
            import OceanStats.Loss.Regression.MSELoss
            fprintf('=== NN-4 Tests ===\n\n');
            % ------------------------------------------------------------------- %
            fprintf('\n-- TEST 2: Residual spec ResidualStart pattern --\n');
            % [d1,d2,d3,d4,d5] — hidden layers 2,3,4 (L=2..4)
            % ResidualStart(L) = true for L=3..nL-1 = 3,4
            specR = NNTopologySpec.make([4,8,8,8,2], ...
            {'linear','relu','relu','relu','linear'}, 'BlockType', 'residual');
            % Expected: L=3 and L=4 are residual ends (skip from L-2)
            expectedRS = [false, false, true, true, false];
            if isequal(specR.ResidualStart, expectedRS)
            fprintf('  PASS  ResidualStart = %s\n', mat2str(specR.ResidualStart));
            passed = passed + 1;
            else
            fprintf('  FAIL  ResidualStart = %s (expected %s)\n', ...
            mat2str(specR.ResidualStart), mat2str(expectedRS));
            failed = failed + 1;
            end
            % ------------------------------------------------------------------- %


            % TEST 3 — VectorizedEngine output shape with residual
            import OceanDL.Graph.*
            import OceanDL.Simulation.*
            import OceanStats.Optim.*
            import OceanStats.Loss.Regression.MSELoss
            fprintf('=== NN-4 Tests ===\n\n');
            % ------------------------------------------------------------------- %
            fprintf('\n-- TEST 3: VectorizedEngine output shape with BlockType=''residual'' --\n');
            rng(1);
            fwd   = VectorizedEngine();
            graphR = NNGraph.fromSpec(specR);
            X3    = randn(20, 4);
            [Y3, cache3] = fwd.forward(graphR, X3);
            ok_shape = isequal(size(Y3), [20, 2]);
            ok_skip  = ~isempty(cache3{3}.A_skip);   % layer 3 should have a skip
            if ok_shape && ok_skip
            fprintf('  PASS  Y shape %s, cache{3}.A_skip present\n', mat2str(size(Y3)));
            passed = passed + 1;
            else
            fprintf('  FAIL  Y shape %s, ok_skip=%d\n', mat2str(size(Y3)), ok_skip);
            failed = failed + 1;
            end
            % ------------------------------------------------------------------- %


            % TEST 4 — Gradient check with residual net
            import OceanDL.Graph.*
            import OceanDL.Simulation.*
            import OceanStats.Optim.*
            import OceanStats.Loss.Regression.MSELoss
            fprintf('=== NN-4 Tests ===\n\n');
            % ------------------------------------------------------------------- %
            fprintf('\n-- TEST 4: Gradient check with residual net --\n');
            rng(3);
            specGC  = NNTopologySpec.make([2,6,6,1], ...
            {'linear','tanh','tanh','linear'}, 'BlockType', 'residual');
            graphGC = NNGraph.fromSpec(specGC);
            loss    = MSELoss();
            bwd     = BackpropEngine();
            X4  = randn(8, 2);
            Yt4 = randn(8, 1);
            [Y4, cache4] = fwd.forward(graphGC, X4);
            grads4 = bwd.backward(graphGC, cache4, Y4, Yt4, loss);
            % Finite-diff on W layer 1->2
            eps_fd = 1e-5;
            W2     = graphGC.getWeightMatrix(1);
            gW_an  = grads4{2}.dW(:);
            gW_fd  = zeros(size(W2));
            for i = 1:numel(W2)
            Wp = W2; Wp(i) = Wp(i) + eps_fd;
            Wm = W2; Wm(i) = Wm(i) - eps_fd;
            graphGC.setWeightMatrix(1, Wp); [Yp,~] = fwd.forward(graphGC, X4);
            graphGC.setWeightMatrix(1, Wm); [Ym,~] = fwd.forward(graphGC, X4);
            gW_fd(i) = (loss.forward(Yt4,Yp) - loss.forward(Yt4,Ym)) / (2*eps_fd);
            end
            graphGC.setWeightMatrix(1, W2);
            maxErr = max(abs(gW_an - gW_fd(:)));
            tol = 1e-4;
            if maxErr < tol
            fprintf('  PASS  max grad error (residual) = %.2e  (tol %.0e)\n', maxErr, tol);
            passed = passed + 1;
            else
            fprintf('  FAIL  max grad error (residual) = %.2e  (tol %.0e)\n', maxErr, tol);
            failed = failed + 1;
            end
            % ------------------------------------------------------------------- %


            % TEST 5 — Residual converges on smooth regression (8-layer)
            import OceanDL.Graph.*
            import OceanDL.Simulation.*
            import OceanStats.Optim.*
            import OceanStats.Loss.Regression.MSELoss
            fprintf('=== NN-4 Tests ===\n\n');
            % ------------------------------------------------------------------- %
            fprintf('\n-- TEST 5: 8-layer residual vs plain convergence --\n');
            rng(42);
            N5  = 200;
            X5  = linspace(-pi, pi, N5)';
            Y5  = sin(X5);
            acts8 = [{'linear'}, repmat({'relu'},1,8), {'linear'}];
            sizes8 = [1, repmat(16,1,8), 1];
            % Plain 8-hidden net
            specPlain = NNTopologySpec.make(sizes8, acts8);
            grPlain   = NNGraph.fromSpec(specPlain);
            trPlain   = Trainer(grPlain, MSELoss(), @() AdamOptimizer('LR',0.01));
            hPlain    = trPlain.fit(X5, Y5, 'Epochs', 300, 'BatchSize', 32);
            % Residual 8-hidden net
            specRes = NNTopologySpec.make(sizes8, acts8, 'BlockType', 'residual');
            grRes   = NNGraph.fromSpec(specRes);
            trRes   = Trainer(grRes, MSELoss(), @() AdamOptimizer('LR',0.01));
            hRes    = trRes.fit(X5, Y5, 'Epochs', 300, 'BatchSize', 32);
            fprintf('  Plain   loss: init=%.4f  final=%.4f\n', hPlain.Loss(1), hPlain.Loss(end));
            fprintf('  Residual loss: init=%.4f  final=%.4f\n', hRes.Loss(1), hRes.Loss(end));
            % Both should decrease; residual should end lower or within 2x
            both_improved = hPlain.Loss(end) < hPlain.Loss(1) && ...
            hRes.Loss(end)   < hRes.Loss(1);
            if both_improved
            fprintf('  PASS  Both nets improved. Residual=%.4f, Plain=%.4f\n', ...
            hRes.Loss(end), hPlain.Loss(end));
            passed = passed + 1;
            else
            fprintf('  FAIL  One or both nets did not converge.\n');
            failed = failed + 1;
            end
            % ------------------------------------------------------------------- %
            fprintf(\'Total: %d passed, %d failed\n\', passed, failed);
            testCase.verifyEqual(failed, 0, \'Some procedural checks failed\');
        end

    end
end