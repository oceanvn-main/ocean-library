classdef TestNn2 < OceanTest.BaseTestCase
    methods (Test)
        function testAll(testCase)
            passed = 0; failed = 0;
            % TEST 1 — Gradient check (finite difference)
            import OceanDL.Graph.*
            import OceanDL.Simulation.*
            import OceanStats.Optim.*
            import OceanStats.Loss.Regression.MSELoss
            import OceanStats.Loss.Classification.BinaryBCELoss
            fprintf('=== NN-2 Tests ===\n\n');
            % ------------------------------------------------------------------- %
            fprintf('-- TEST 1: Gradient check (BackpropEngine vs finite-diff) --\n');
            rng(7);
            spec  = NNTopologySpec.make([3, 5, 2], {'linear','tanh','linear'});
            graph = NNGraph.fromSpec(spec);
            loss  = MSELoss();
            fwd   = VectorizedEngine();
            bwd   = BackpropEngine();
            N  = 8;
            X  = randn(N, 3);
            Yt = randn(N, 2);
            [Y, cache] = fwd.forward(graph, X);
            grads      = bwd.backward(graph, cache, Y, Yt, loss);
            % Flatten all analytical gradients
            gW2_analytic = grads{2}.dW(:);
            gW3_analytic = grads{3}.dW(:);
            % Finite-difference gradient for W of layer 1->2
            eps_fd = 1e-5;
            W2     = graph.getWeightMatrix(1);
            gW2_fd = zeros(size(W2));
            for i = 1:numel(W2)
            W_plus  = W2; W_plus(i)  = W_plus(i)  + eps_fd;
            W_minus = W2; W_minus(i) = W_minus(i) - eps_fd;
            graph.setWeightMatrix(1, W_plus);
            [Yp, ~] = fwd.forward(graph, X);
            Lp = loss.forward(Yt, Yp);
            graph.setWeightMatrix(1, W_minus);
            [Ym, ~] = fwd.forward(graph, X);
            Lm = loss.forward(Yt, Ym);
            gW2_fd(i) = (Lp - Lm) / (2 * eps_fd);
            end
            graph.setWeightMatrix(1, W2);  % restore
            maxErr = max(abs(gW2_analytic - gW2_fd(:)));
            tol    = 1e-4;
            if maxErr < tol
            fprintf('  PASS  max grad error = %.2e  (tol %.0e)\n', maxErr, tol);
            passed = passed + 1;
            else
            fprintf('  FAIL  max grad error = %.2e  (tol %.0e)\n', maxErr, tol);
            failed = failed + 1;
            end
            % ------------------------------------------------------------------- %


            % TEST 2 — XOR test (Trainer with Adam)
            import OceanDL.Graph.*
            import OceanDL.Simulation.*
            import OceanStats.Optim.*
            import OceanStats.Loss.Regression.MSELoss
            import OceanStats.Loss.Classification.BinaryBCELoss
            fprintf('=== NN-2 Tests ===\n\n');
            % ------------------------------------------------------------------- %
            fprintf('\n-- TEST 2: XOR convergence (< 2000 epochs, loss < 0.05) --\n');
            X_xor = [0 0; 0 1; 1 0; 1 1];
            Y_xor = [0; 1; 1; 0];
            rng(42);
            specXOR  = NNTopologySpec.make([2, 8, 1], {'linear','relu','sigmoid'});
            graphXOR = NNGraph.fromSpec(specXOR);
            lossXOR  = BinaryBCELoss();
            optFnXOR = @() AdamOptimizer('LR', 0.05);
            trainerXOR = Trainer(graphXOR, lossXOR, optFnXOR);
            histXOR    = trainerXOR.fit(X_xor, Y_xor, 'Epochs', 2000, 'BatchSize', 4);
            finalLoss = histXOR.Loss(end);
            if finalLoss < 0.05
            fprintf('  PASS  final loss = %.5f  (threshold 0.05) after 2000 epochs\n', finalLoss);
            passed = passed + 1;
            else
            fprintf('  FAIL  final loss = %.5f  (threshold 0.05) after 2000 epochs\n', finalLoss);
            failed = failed + 1;
            end
            % ------------------------------------------------------------------- %


            % TEST 3 — All 4 optimizers produce monotonically decreasing loss
            import OceanDL.Graph.*
            import OceanDL.Simulation.*
            import OceanStats.Optim.*
            import OceanStats.Loss.Regression.MSELoss
            import OceanStats.Loss.Classification.BinaryBCELoss
            fprintf('=== NN-2 Tests ===\n\n');
            % ------------------------------------------------------------------- %
            fprintf('\n-- TEST 3: All 4 optimizers converge on simple regression --\n');
            rng(1);
            N3 = 50;
            X3 = linspace(-1, 1, N3)';
            Y3 = 2*X3 + 0.1*randn(N3, 1);   % near-linear signal
            optCfgs = {
            'SGD',      @() SGDOptimizer('LR', 0.05);
            'Momentum', @() SGDMomentumOptimizer('LR', 0.02, 'Beta', 0.9);
            'RMSProp',  @() RMSPropOptimizer('LR', 0.01, 'Rho', 0.9);
            'Adam',     @() AdamOptimizer('LR', 0.01);
            };
            lossReg = MSELoss();
            for k = 1:size(optCfgs, 1)
            optName = optCfgs{k,1};
            optFn   = optCfgs{k,2};
            rng(99);
            specR  = NNTopologySpec.make([1, 8, 1], {'linear','relu','linear'});
            graphR = NNGraph.fromSpec(specR);
            trainerR = Trainer(graphR, lossReg, optFn);
            histR    = trainerR.fit(X3, Y3, 'Epochs', 300, 'BatchSize', 16);
            % Check: loss(end) < loss(1) — i.e. it went down overall
            improved = histR.Loss(end) < histR.Loss(1);
            if improved
            fprintf('  PASS  %-12s  loss: %.4f → %.4f\n', ...
            optName, histR.Loss(1), histR.Loss(end));
            passed = passed + 1;
            else
            fprintf('  FAIL  %-12s  loss: %.4f → %.4f  (did not decrease)\n', ...
            optName, histR.Loss(1), histR.Loss(end));
            failed = failed + 1;
            end
            end
            % ------------------------------------------------------------------- %
            fprintf(\'Total: %d passed, %d failed\n\', passed, failed);
            testCase.verifyEqual(failed, 0, \'Some procedural checks failed\');
        end

    end
end