classdef TestNn6 < OceanTest.BaseTestCase
    methods (Test)
        function testAll(testCase)
            passed = 0; failed = 0;
            % TEST 1 — CPUBackend: bit-identical to no-backend
            import OceanDL.Graph.*
            import OceanDL.Simulation.*
            import OceanDL.CPUBackend
            import OceanDL.GPUBackend
            import OceanStats.Loss.Regression.MSELoss
            fprintf('=== NN-6 Tests ===\n\n');
            spec  = NNTopologySpec.make([4, 32, 32, 2], {'linear','relu','relu','linear'});
            graph = NNGraph.fromSpec(spec);
            fwd   = VectorizedEngine();
            bwd   = BackpropEngine();
            loss  = MSELoss();
            N    = 64;
            X    = randn(N, 4);
            Ytgt = randn(N, 2);
            % ------------------------------------------------------------------- %
            fprintf('-- TEST 1: CPUBackend bit-identical to default (no Backend) --\n');
            [Y_default, cache_default] = fwd.forward(graph, X);
            [Y_cpu,     cache_cpu    ] = fwd.forward(graph, X, 'train', CPUBackend());
            diff_Y = max(abs(Y_default(:) - Y_cpu(:)));
            if diff_Y == 0
            fprintf('  PASS  max |Y_default - Y_cpu| = 0 (bit-identical)\n');
            passed = passed + 1;
            else
            fprintf('  FAIL  max diff = %.2e  (should be 0)\n', diff_Y);
            failed = failed + 1;
            end
            % Gradients also bit-identical
            grads_default = bwd.backward(graph, cache_default, Y_default, Ytgt, loss);
            grads_cpu     = bwd.backward(graph, cache_cpu,     Y_cpu,     Ytgt, loss, CPUBackend());
            diff_dW = max(abs(grads_default{2}.dW(:) - grads_cpu{2}.dW(:)));
            if diff_dW == 0
            fprintf('  PASS  max |dW_default - dW_cpu| = 0 (bit-identical gradients)\n');
            passed = passed + 1;
            else
            fprintf('  FAIL  max dW diff = %.2e\n', diff_dW);
            failed = failed + 1;
            end
            % ------------------------------------------------------------------- %


            % TEST 2 — GPUBackend: graceful construction (no error even sans GPU)
            import OceanDL.Graph.*
            import OceanDL.Simulation.*
            import OceanDL.CPUBackend
            import OceanDL.GPUBackend
            import OceanStats.Loss.Regression.MSELoss
            fprintf('=== NN-6 Tests ===\n\n');
            spec  = NNTopologySpec.make([4, 32, 32, 2], {'linear','relu','relu','linear'});
            graph = NNGraph.fromSpec(spec);
            fwd   = VectorizedEngine();
            bwd   = BackpropEngine();
            loss  = MSELoss();
            N    = 64;
            X    = randn(N, 4);
            Ytgt = randn(N, 2);
            % ------------------------------------------------------------------- %
            fprintf('\n-- TEST 2: GPUBackend constructs without error (fallback ok) --\n');
            gpuB = GPUBackend();
            fprintf('  PASS  GPUBackend created: backend.name() = ''%s''\n', gpuB.name());
            passed = passed + 1;
            % ------------------------------------------------------------------- %


            % TEST 3 — GPUBackend results match CPUBackend
            import OceanDL.Graph.*
            import OceanDL.Simulation.*
            import OceanDL.CPUBackend
            import OceanDL.GPUBackend
            import OceanStats.Loss.Regression.MSELoss
            fprintf('=== NN-6 Tests ===\n\n');
            spec  = NNTopologySpec.make([4, 32, 32, 2], {'linear','relu','relu','linear'});
            graph = NNGraph.fromSpec(spec);
            fwd   = VectorizedEngine();
            bwd   = BackpropEngine();
            loss  = MSELoss();
            N    = 64;
            X    = randn(N, 4);
            Ytgt = randn(N, 2);
            % ------------------------------------------------------------------- %
            fprintf('\n-- TEST 3: GPUBackend results numerically match CPUBackend --\n');
            [Y_gpu, cache_gpu] = fwd.forward(graph, X, 'train', gpuB);
            diff_gpu = max(abs(Y_cpu(:) - Y_gpu(:)));
            tol = 1e-10;
            if diff_gpu <= tol
            fprintf('  PASS  max |Y_cpu - Y_gpu| = %.2e  (tol %.0e)\n', diff_gpu, tol);
            passed = passed + 1;
            else
            fprintf('  FAIL  max diff = %.2e  (tol %.0e)\n', diff_gpu, tol);
            failed = failed + 1;
            end
            grads_gpu = bwd.backward(graph, cache_gpu, Y_gpu, Ytgt, loss, gpuB);
            diff_dW_gpu = max(abs(grads_cpu{2}.dW(:) - grads_gpu{2}.dW(:)));
            if diff_dW_gpu <= tol
            fprintf('  PASS  max |dW_cpu - dW_gpu| = %.2e\n', diff_dW_gpu);
            passed = passed + 1;
            else
            fprintf('  FAIL  max dW diff = %.2e\n', diff_dW_gpu);
            failed = failed + 1;
            end
            % ------------------------------------------------------------------- %


            % TEST 4 — Benchmark: CPU vs GPU at N=1000
            import OceanDL.Graph.*
            import OceanDL.Simulation.*
            import OceanDL.CPUBackend
            import OceanDL.GPUBackend
            import OceanStats.Loss.Regression.MSELoss
            fprintf('=== NN-6 Tests ===\n\n');
            spec  = NNTopologySpec.make([4, 32, 32, 2], {'linear','relu','relu','linear'});
            graph = NNGraph.fromSpec(spec);
            fwd   = VectorizedEngine();
            bwd   = BackpropEngine();
            loss  = MSELoss();
            N    = 64;
            X    = randn(N, 4);
            Ytgt = randn(N, 2);
            % ------------------------------------------------------------------- %
            fprintf('\n-- TEST 4: Benchmark (N=1000, 100 forward+backward reps) --\n');
            rng(10);
            specBig  = NNTopologySpec.make([64,256,256,256,32], ...
            {'linear','relu','relu','relu','linear'});
            graphBig = NNGraph.fromSpec(specBig);
            N_bench  = 1000;
            X_bench  = randn(N_bench, 64);
            Y_bench  = randn(N_bench, 32);
            REPS     = 100;
            cpuB = CPUBackend();
            % CPU benchmark
            t_cpu = tic;
            for r = 1:REPS
            [Yb, cb] = fwd.forward(graphBig, X_bench, 'train', cpuB);
            bwd.backward(graphBig, cb, Yb, Y_bench, MSELoss(), cpuB);
            end
            cpu_ms = toc(t_cpu) * 1000 / REPS;
            % GPU benchmark (whether real GPU or fallback)
            t_gpu = tic;
            for r = 1:REPS
            [Yb, cb] = fwd.forward(graphBig, X_bench, 'train', gpuB);
            bwd.backward(graphBig, cb, Yb, Y_bench, MSELoss(), gpuB);
            end
            gpu_ms = toc(t_gpu) * 1000 / REPS;
            fprintf('\n  %-12s  %8s\n', 'Backend', 'ms/iter');
            fprintf('  %-12s  %8.3f\n', cpuB.name(), cpu_ms);
            fprintf('  %-12s  %8.3f\n', gpuB.name(), gpu_ms);
            if gpuB.isGPU()
            ratio = cpu_ms / gpu_ms;
            fprintf('  GPU speedup: %.1fx\n', ratio);
            else
            fprintf('  (GPU not available — benchmark is CPU-only)\n');
            end
            fprintf('  PASS  benchmark complete\n');
            passed = passed + 1;
            % ------------------------------------------------------------------- %
            fprintf(\'Total: %d passed, %d failed\n\', passed, failed);
            testCase.verifyEqual(failed, 0, \'Some procedural checks failed\');
        end

    end
end