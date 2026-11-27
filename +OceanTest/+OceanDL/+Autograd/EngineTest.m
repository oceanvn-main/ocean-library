classdef EngineTest < OceanTest.BaseTestCase
    % ENGINETEST  Graph semantics of the Tensor autograd engine: multi-use
    %   accumulation, deep unrolled graphs (BPTT), PyTorch-compatible grad
    %   accumulation / release / retain rules, no-grad mode, precision.

    methods (Test)
        function testDiamondAccumulates(testCase)
            % y = x.*x + x uses x three times: dy/dx = 2x + 1.
            x = OceanDL.Autograd.Tensor([1 -2 3], 'RequiresGrad', true);
            y = sum(x .* x + x);
            y.backward();
            testCase.verifyEqual(x.Grad, 2 * [1 -2 3] + 1);
        end

        function testDeepChainNoRecursionLimit(testCase)
            % h_t = a h_{t-1} + 1, T = 3000 steps, shared a: this is BPTT on a
            % graph far deeper than MATLAB's recursion limit.
            % h_T = sum_{k<T} a^k (h_0 = 0)  =>  dh_T/da = sum_k k a^(k-1).
            a0 = 0.999;
            T = 3000;
            a = OceanDL.Autograd.Tensor(a0, 'RequiresGrad', true);
            h = OceanDL.Autograd.Tensor(0);
            for t = 1:T
                h = a .* h + 1;
            end
            h.backward();
            k = 1:T-1;
            testCase.verifyEqual(h.Data, sum(a0 .^ (0:T-1)), 'RelTol', 1e-10);
            testCase.verifyEqual(a.Grad, sum(k .* a0 .^ (k - 1)), 'RelTol', 1e-9);
        end

        function testRecurrentMatrixGradcheck(testCase)
            % A vanilla RNN unrolled 40 steps: gradient w.r.t. the SHARED W is
            % the sum over time steps (Goodfellow Eq. 10.22-10.28), checked numerically.
            s = RandStream('mt19937ar', 'Seed', 3);
            X = randn(s, 3, 2, 40);
            fn = @(W, U) OceanTest.OceanDL.Autograd.EngineTest.rnnLoss(W, U, X);
            r = OceanDL.Autograd.GradCheck.check(fn, {0.5 * randn(s, 4, 4), randn(s, 4, 3)});
            testCase.verifyTrue(r.Passed, sprintf('max err %.3g', r.MaxAbsError));
        end

        function testGradsAccumulateUntilZeroGrad(testCase)
            w = OceanDL.Autograd.Tensor(3, 'RequiresGrad', true);
            backward(w .* 2);
            backward(w .* 2);
            testCase.verifyEqual(w.Grad, 4);
            w.zeroGrad();
            backward(w .* 5);
            testCase.verifyEqual(w.Grad, 5);
        end

        function testSecondBackwardNeedsRetainGraph(testCase)
            x = OceanDL.Autograd.Tensor(2, 'RequiresGrad', true);
            y = x .* x;
            y.backward();
            testCase.verifyError(@() y.backward(), 'OceanDL:Autograd:Engine:GraphReleased');
            x2 = OceanDL.Autograd.Tensor(2, 'RequiresGrad', true);
            y2 = x2 .* x2;
            y2.backward([], 'RetainGraph', true);
            y2.backward();
            testCase.verifyEqual(x2.Grad, 8);
        end

        function testNoGradBuildsNoGraph(testCase)
            import OceanDL.Autograd.*
            w = Tensor(randn(2), 'RequiresGrad', true);
            y = GradMode.noGrad(@() tanh(w * [1; 2]));
            testCase.verifyFalse(y.RequiresGrad);
            testCase.verifyTrue(y.IsLeaf);
            testCase.verifyTrue(GradMode.isEnabled());          % restored afterwards
            c = Tensor(1) + Tensor(2);                          % constants only
            testCase.verifyTrue(c.IsLeaf);
        end

        function testRetainGradAndGradQueryOnIntermediate(testCase)
            import OceanDL.Autograd.*
            x = Tensor([0.3; -0.4], 'RequiresGrad', true);
            h = x .* 3;                                         % intermediate
            h.retainGrad();
            L = sum(tanh(h));
            L.backward([], 'RetainGraph', true);
            testCase.verifyEqual(h.Grad, 1 - tanh(3 * [0.3; -0.4]).^2, 'AbsTol', 1e-14);
            % torch.autograd.grad-style query: no .Grad side effect.
            x.zeroGrad();
            g = Engine.grad(L, {h, x});
            testCase.verifyEqual(g{1}, h.Grad, 'AbsTol', 1e-14);
            testCase.verifyEqual(g{2}, 3 * h.Grad, 'AbsTol', 1e-14);
            testCase.verifyEmpty(x.Grad);
        end

        function testNonScalarBackwardNeedsSeed(testCase)
            x = OceanDL.Autograd.Tensor([1 2], 'RequiresGrad', true);
            y = x .* x;
            testCase.verifyError(@() y.backward(), 'OceanDL:Autograd:Engine:NonScalarRoot');
            y.backward([1 10]);
            testCase.verifyEqual(x.Grad, [2 40]);
        end

        function testSinglePrecisionIsPreserved(testCase)
            W = OceanDL.Autograd.Tensor(single(randn(3)), 'RequiresGrad', true);
            L = sum(tanh(W * single([1; 2; 3])), 'all');
            L.backward();
            testCase.verifyClass(L.Data, 'single');
            testCase.verifyClass(W.Grad, 'single');
        end

        function testInPlaceAssignmentIsRejected(testCase)
            x = OceanDL.Autograd.Tensor([1 2 3]);
            testCase.verifyError(@() OceanTest.OceanDL.Autograd.EngineTest.assignInto(x), ...
                'OceanDL:Autograd:Tensor:NoInPlace');
        end

        function testDetachStopsGradient(testCase)
            x = OceanDL.Autograd.Tensor(2, 'RequiresGrad', true);
            y = x .* x.detach();                                % d/dx = x (the detached factor is a constant)
            y.backward();
            testCase.verifyEqual(x.Grad, 2);
        end
    end

    methods (Static)
        function L = rnnLoss(W, U, X)
            h = zeros(size(W.Data, 1), size(X, 2));
            for t = 1:size(X, 3)
                h = tanh(W * h + U * X(:, :, t));
            end
            L = sum(h .^ 2, 'all');
        end

        function assignInto(x)
            x(1) = 5; %#ok<NASGU>
        end
    end
end
