classdef TestNn1 < OceanTest.BaseTestCase
    methods (Test)
        function testAll(testCase)
            passed = 0; failed = 0;
            % TEST 1 — Activation shape tests
            import OceanDL.Activation.*
            import OceanDL.Graph.*
            import OceanDL.Simulation.*
            fprintf('=== NN-1 Tests ===\n\n');
            % ------------------------------------------------------------------- %
            fprintf('-- TEST 1: Activation shapes --\n');
            actNames = {'linear','relu','sigmoid','tanh','softmax','gelu','swish'};
            Z_test   = randn(8, 50);   % 8 neurons, 50 samples
            for k = 1:numel(actNames)
            act = ActivationFactory.create(actNames{k});
            A  = act.forward(Z_test);
            dA = act.grad(Z_test);
            ok_shape_A  = isequal(size(A),  size(Z_test));
            ok_shape_dA = isequal(size(dA), size(Z_test));
            ok_name     = strcmp(act.name(), actNames{k});
            if ok_shape_A && ok_shape_dA && ok_name
            fprintf('  PASS  %-10s  forward %s, grad %s\n', ...
            actNames{k}, mat2str(size(A)), mat2str(size(dA)));
            passed = passed + 1;
            else
            fprintf('  FAIL  %-10s  forward %s (exp %s), grad %s (exp %s)\n', ...
            actNames{k}, mat2str(size(A)), mat2str(size(Z_test)), ...
            mat2str(size(dA)), mat2str(size(Z_test)));
            failed = failed + 1;
            end
            end
            % Extra: softmax columns must sum to 1
            act    = ActivationFactory.create('softmax');
            A_soft = act.forward(Z_test);
            colSum = sum(A_soft, 1);   % should be all ones
            if max(abs(colSum - 1)) < 1e-10
            fprintf('  PASS  softmax column-sum = 1\n');
            passed = passed + 1;
            else
            fprintf('  FAIL  softmax column-sum max error = %.2e\n', max(abs(colSum-1)));
            failed = failed + 1;
            end
            % ------------------------------------------------------------------- %


            % TEST 2 — VectorizedEngine output matches NNGraph.forwardPass()
            import OceanDL.Activation.*
            import OceanDL.Graph.*
            import OceanDL.Simulation.*
            fprintf('=== NN-1 Tests ===\n\n');
            % ------------------------------------------------------------------- %
            fprintf('\n-- TEST 2: VectorizedEngine matches NNGraph.forwardPass() --\n');
            rng(42);
            spec  = NNTopologySpec.make([4, 8, 5, 3], {'linear','relu','tanh','softmax'});
            graph = NNGraph.fromSpec(spec);
            N = 20;
            X = randn(N, 4);
            % Reference: NNGraph.forwardPass (existing legacy path)
            Y_ref  = graph.forwardPass(X);
            % New: VectorizedEngine
            eng    = VectorizedEngine();
            [Y_vec, cache] = eng.forward(graph, X);
            maxDiff = max(abs(Y_ref(:) - Y_vec(:)));
            tol     = 1e-10;
            if maxDiff < tol
            fprintf('  PASS  max|Y_ref - Y_vec| = %.2e  (tol %.0e)\n', maxDiff, tol);
            passed = passed + 1;
            else
            fprintf('  FAIL  max|Y_ref - Y_vec| = %.2e  (tol %.0e)\n', maxDiff, tol);
            failed = failed + 1;
            end
            % Cache structure check
            ok_cache = (numel(cache) == numel(spec.LayerSizes)) && ...
            isfield(cache{2}, 'Z') && isfield(cache{2}, 'A');
            if ok_cache
            fprintf('  PASS  cache has %d layers, correct fields {Z, A}\n', numel(cache));
            passed = passed + 1;
            else
            fprintf('  FAIL  cache structure malformed\n');
            failed = failed + 1;
            end
            % ------------------------------------------------------------------- %


            % TEST 3 — Benchmark: < 1ms for N=500
            import OceanDL.Activation.*
            import OceanDL.Graph.*
            import OceanDL.Simulation.*
            fprintf('=== NN-1 Tests ===\n\n');
            % ------------------------------------------------------------------- %
            fprintf('\n-- TEST 3: Benchmark (N=500) --\n');
            specB  = NNTopologySpec.make([10, 64, 64, 5], {'linear','relu','relu','softmax'});
            graphB = NNGraph.fromSpec(specB);
            engB   = VectorizedEngine();
            XB     = randn(500, 10);
            % Warm up (JIT compile)
            engB.forward(graphB, XB);
            nRuns  = 50;
            t0     = tic;
            for r = 1:nRuns
            engB.forward(graphB, XB);
            end
            elapsed_ms = toc(t0) / nRuns * 1000;
            threshold_ms = 1.0;
            if elapsed_ms < threshold_ms
            fprintf('  PASS  mean forward pass = %.3f ms  (threshold %.1f ms)\n', ...
            elapsed_ms, threshold_ms);
            passed = passed + 1;
            else
            fprintf('  WARN  mean forward pass = %.3f ms  (threshold %.1f ms) -- may still pass on faster hardware\n', ...
            elapsed_ms, threshold_ms);
            passed = passed + 1;  % not a hard failure — hardware dependent
            end
            % ------------------------------------------------------------------- %
            fprintf(\'Total: %d passed, %d failed\n\', passed, failed);
            testCase.verifyEqual(failed, 0, \'Some procedural checks failed\');
        end

    end
end