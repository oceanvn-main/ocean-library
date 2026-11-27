classdef TensorOpsGradTest < OceanTest.BaseTestCase
    % TENSOROPSGRADTEST  Every primitive op's VJP against central finite
    %   differences (GradCheck), including the broadcasting and
    %   repeated-index cases where hand-written gradients usually go wrong.

    methods (Test)
        function testArithmeticWithBroadcasting(testCase)
            s = RandStream('mt19937ar', 'Seed', 1);
            A = randn(s, 3, 4); col = randn(s, 3, 1); row = randn(s, 1, 4);
            testCase.gc(@(a, b) a + b, {A, col});                 % bias-style column
            testCase.gc(@(a, b) a - b, {A, row});
            testCase.gc(@(a, b) a .* b, {A, row});
            testCase.gc(@(a, b) a ./ b, {A, abs(col) + 1});
            testCase.gc(@(a, b) a .* b, {A, 2.5});                % scalar broadcast
            testCase.gc(@(a) -a, {A});
        end

        function testMatrixProductAndScalarTimes(testCase)
            s = RandStream('mt19937ar', 'Seed', 2);
            testCase.gc(@(a, b) a * b, {randn(s, 3, 4), randn(s, 4, 2)});
            testCase.gc(@(a, b) a * b, {randn(s, 3, 4), 0.7});    % scalar -> elementwise
            testCase.gc(@(a) a.', {randn(s, 3, 4)});
        end

        function testElementwiseMath(testCase)
            s = RandStream('mt19937ar', 'Seed', 3);
            P = rand(s, 3, 4) + 0.5;                              % positive
            X = randn(s, 3, 4);
            X(abs(X) < 0.1) = 0.3;                                % away from relu/abs kinks
            testCase.gc(@(a) exp(a), {X});
            testCase.gc(@(a) log(a), {P});
            testCase.gc(@(a) sqrt(a), {P});
            testCase.gc(@(a) tanh(a), {X});
            testCase.gc(@(a) sigmoid(a), {X});
            testCase.gc(@(a) relu(a), {X});
            testCase.gc(@(a) abs(a), {X});
            testCase.gc(@(a) a .^ 3, {X});
            testCase.gc(@(a) a .^ 0.5, {P});
        end

        function testReductions(testCase)
            s = RandStream('mt19937ar', 'Seed', 4);
            X = randn(s, 3, 4, 2);
            testCase.gc(@(a) sum(a), {X});
            testCase.gc(@(a) sum(a, 2), {X});
            testCase.gc(@(a) sum(a, 'all'), {X});
            testCase.gc(@(a) mean(a, 3), {X});
            testCase.gc(@(a) mean(a, 'all'), {X});
            testCase.gc(@(a) max(a, [], 2), {X + reshape(0:0.37:0.37*23, 3, 4, 2)});  % distinct values
        end

        function testShapeOps(testCase)
            s = RandStream('mt19937ar', 'Seed', 5);
            X = randn(s, 2, 3, 4);
            testCase.gc(@(a) reshape(a, 6, []), {X});
            testCase.gc(@(a) permute(a, [3 1 2]), {X});
            testCase.gc(@(a, b) [a; b], {randn(s, 2, 3), randn(s, 1, 3)});
            testCase.gc(@(a, b) [a, b, a], {randn(s, 2, 3), randn(s, 2, 1)});   % a used twice
            testCase.gc(@(a, b) cat(3, a, b), {randn(s, 2, 3), randn(s, 2, 3)});
        end

        function testIndexingIncludingRepeats(testCase)
            s = RandStream('mt19937ar', 'Seed', 6);
            X = randn(s, 4, 5);
            testCase.gc(@(a) a(2:end, [1 1 3]), {X});          % repeated column
            testCase.gc(@(a) a([3 3 7 20]), {X});               % linear, repeated
            testCase.gc(@(a) a(:), {X});
            testCase.gc(@(a) a(logical([1 0 1 1]), :), {X});
            testCase.gc(@(a) a(:, end), {X});
            testCase.gc(@(a) a(2, :, 1), {randn(s, 3, 4, 2)});
        end

        function testSoftmaxFamily(testCase)
            s = RandStream('mt19937ar', 'Seed', 7);
            Z = randn(s, 5, 3);
            testCase.gc(@(z) softmax(z, 1), {Z});
            testCase.gc(@(z) softmax(z, 2), {Z});
            testCase.gc(@(z) logSoftmax(z, 1), {Z});
        end

        function testSoftmaxIsStableForHugeLogits(testCase)
            z = OceanDL.Autograd.Tensor([1000; 0; -1000]);
            y = logSoftmax(z, 1);
            testCase.verifyTrue(all(isfinite(y.Data)));
            testCase.verifyEqual(y.Data(1), 0, 'AbsTol', 1e-12);
        end

        function testLosses(testCase)
            F = OceanDL.Autograd.F;
            s = RandStream('mt19937ar', 'Seed', 8);
            labels = [2 1 3 3];
            testCase.gc(@(z) F.crossEntropy(z, labels), {randn(s, 3, 4)});
            seqLabels = [1 2 3; 3 3 1];                           % N=2, T=3
            mask = logical([1 1 0; 1 0 0]);
            testCase.gc(@(z) F.crossEntropy(z, seqLabels, 'Mask', mask), {randn(s, 3, 2, 3)});
            testCase.gc(@(p) F.mseLoss(p, [1 2; 3 4]), {randn(s, 2, 2)});
            testCase.gc(@(z) F.bceWithLogits(z, [1 0 1]), {randn(s, 1, 3)});
        end

        function testCrossEntropyValue(testCase)
            % Hand value: logits [0; ln 3] -> p = [1/4; 3/4]; label 2 -> -ln(3/4).
            F = OceanDL.Autograd.F;
            L = F.crossEntropy(OceanDL.Autograd.Tensor([0; log(3)]), 2);
            testCase.verifyEqual(L.Data, -log(3/4), 'AbsTol', 1e-12);
        end

        function testEmbeddingAccumulatesRepeatedIds(testCase)
            F = OceanDL.Autograd.F;
            s = RandStream('mt19937ar', 'Seed', 9);
            ids = [2 5; 2 1; 4 2];                                % id 2 three times, (N=3 x T=2)
            testCase.gc(@(W) F.embedding(W, ids), {randn(s, 3, 6)});
            W = OceanDL.Autograd.Tensor(randn(s, 3, 6), 'RequiresGrad', true);
            E = F.embedding(W, ids);
            testCase.verifySize(E.Data, [3 3 2]);
            backward(sum(E, 'all'));
            testCase.verifyEqual(W.Grad(1, :), [1 3 0 1 1 0]);    % column 2 read 3x, 3 and 6 never
        end

        function testConv2dWithStridePaddingBias(testCase)
            F = OceanDL.Autograd.F;
            s = RandStream('mt19937ar', 'Seed', 10);
            testCase.gc(@(x, w, b) F.conv2d(x, w, b, 'Stride', [2 1], 'Padding', [1 0]), ...
                {randn(s, 5, 4, 2, 2), randn(s, 3, 2, 2, 3), randn(s, 3, 1)});
        end
    end

    methods (Access = private)
        function gc(testCase, fn, inputs)
            r = OceanDL.Autograd.GradCheck.check(fn, inputs);
            testCase.verifyTrue(r.Passed, sprintf('%s: max abs err %.3g, rel %.3g', ...
                func2str(fn), r.MaxAbsError, r.MaxRelError));
        end
    end
end
