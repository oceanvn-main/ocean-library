classdef RecurrentLayerTest < OceanTest.BaseTestCase
    % RECURRENTLAYERTEST  Unrolling semantics (masking, reverse,
    %   bidirectional, stacking, pooling, truncated BPTT), the Sec. 10.7
    %   diagnostic, and an end-to-end learning task that needs memory.

    methods (Test)
        function testMaskedBatchEqualsUnpaddedRuns(testCase)
            import OceanDL.Recurrent.*
            s = RandStream('mt19937ar', 'Seed', 1);
            X = randn(s, 3, 2, 6);
            mask = logical([1 1 1 1 0 0; 1 1 1 1 1 1]);                  % sequence 1 has length 4
            for rev = [false true]
                layer = RecurrentLayer(LSTMCell(3, 5, 'Seed', 2), 'Reverse', rev);
                [Y, st] = layer.forward(X, 'Mask', mask);
                [Y1, st1] = layer.forward(X(:, 1, 1:4));                   % sequence 1 alone, unpadded
                testCase.verifyEqual(st.h.Data(:, 1), st1.h.Data, 'AbsTol', 1e-14);
                testCase.verifyEqual(st.c.Data(:, 1), st1.c.Data, 'AbsTol', 1e-14);
                testCase.verifyEqual(Y.Data(:, 1, 1:4), Y1.Data, 'AbsTol', 1e-14);
                testCase.verifyEqual(Y.Data(:, 1, 5:6), zeros(5, 1, 2));    % padded outputs are 0
            end
        end

        function testReverseEqualsForwardOnFlippedSequence(testCase)
            import OceanDL.Recurrent.*
            cell = GRUCell(3, 4, 'Seed', 3);
            X = randn(3, 2, 5);
            Yr = RecurrentLayer(cell, 'Reverse', true).forward(X);
            Yf = RecurrentLayer(cell).forward(flip(X, 3));
            testCase.verifyEqual(Yr.Data, flip(Yf.Data, 3), 'AbsTol', 1e-14);
        end

        function testBidirectionalConcatenatesDirections(testCase)
            bi = OceanDL.Recurrent.Bidirectional(@() OceanDL.Recurrent.ElmanCell(3, 4));
            X = randn(3, 2, 5);
            Y = bi.forward(X);
            testCase.verifySize(Y.Data, [8 2 5]);
            testCase.verifyEqual(bi.outputSize(), 8);
            testCase.verifyEqual(Y.Data(5:8, :, :), bi.BackwardLayer.forward(X).Data, 'AbsTol', 1e-14);
            testCase.verifyEqual(numel(bi.parameters()), 6);                % two independent cells
        end

        function testStackedEvalIsDeterministicAndResidualKeepsSize(testCase)
            import OceanDL.Recurrent.*
            stack = StackedRecurrent({RecurrentLayer(GRUCell(3, 6, 'Seed', 1)), ...
                RecurrentLayer(GRUCell(6, 6, 'Seed', 2))}, 'Dropout', 0.5, 'Residual', true);
            X = randn(3, 4, 5);
            stack.eval();
            Y1 = stack.forward(X); Y2 = stack.forward(X);
            testCase.verifyEqual(Y1.Data, Y2.Data);
            testCase.verifySize(Y1.Data, [6 4 5]);
        end

        function testSequencePoolModes(testCase)
            Y = reshape(1:2*2*3, 2, 2, 3);                                % H=2, N=2, T=3
            mask = logical([1 1 0; 1 1 1]);
            import OceanDL.Recurrent.SequencePool
            last = SequencePool("last").forward(Y, mask);
            testCase.verifyEqual(last, [Y(:, 1, 2), Y(:, 2, 3)]);
            mn = SequencePool("mean").forward(Y, mask);
            testCase.verifyEqual(mn, [mean(Y(:, 1, 1:2), 3), mean(Y(:, 2, :), 3)]);
            mx = SequencePool("max").forward(-Y, mask);
            testCase.verifyEqual(mx, [-Y(:, 1, 1), -Y(:, 2, 1)]);
            r = OceanDL.Autograd.GradCheck.check(@(y) SequencePool("last").forward(y, mask), {randn(2, 2, 3)});
            testCase.verifyTrue(r.Passed);
        end

        function testTruncatedBpttCutsTheGraph(testCase)
            import OceanDL.Recurrent.*
            layer = RecurrentLayer(ElmanCell(2, 3, 'Seed', 4));
            X = randn(2, 1, 8);
            Yfull = layer.forward(X);
            x = OceanDL.Autograd.Tensor(X, 'RequiresGrad', true);
            [~, s1] = layer.forward(x(:, :, 1:4));
            Y2 = layer.forward(x(:, :, 5:8), 'InitialState', RecurrentLayer.detachState(s1));
            testCase.verifyEqual(Y2.Data, Yfull.Data(:, :, 5:8), 'AbsTol', 1e-14);   % values continue
            backward(sum(Y2, 'all'));
            testCase.verifyEqual(x.Grad(:, :, 1:4), zeros(2, 1, 4));                 % no gradient crosses the cut
            testCase.verifyTrue(any(x.Grad(:, :, 5:8) ~= 0, 'all'));
        end

        function testVanishingGradientIsVisibleAndLstmResists(testCase)
            % Loss on the last step only; g(t) = ||dL/dh_t||. Elman with a
            % contracting W: g(1) is orders of magnitude below g(T). An LSTM
            % with forget bias 3 keeps far more signal at t = 1 (Sec. 10.7, 10.10).
            import OceanDL.Recurrent.*
            T = 40;
            X = randn(RandStream('mt19937ar', 'Seed', 5), 4, 8, T);
            elman = RecurrentLayer(ElmanCell(4, 16, 'Seed', 6), 'TrackStates', true);
            elman.Cell.Wh.Data = 0.5 * elman.Cell.Wh.Data;
            lstm = RecurrentLayer(LSTMCell(4, 16, 'ForgetBias', 3, 'Seed', 6), 'TrackStates', true);
            ratio = zeros(1, 2);
            layers = {elman, lstm};
            for k = 1:2
                Y = layers{k}.forward(X);
                backward(sum(Y(:, :, T), 'all'));
                g = Diagnostics.gradientNormsOverTime(layers{k});
                ratio(k) = g(1) / g(T);
            end
            testCase.verifyLessThan(ratio(1), 1e-6);
            testCase.verifyGreaterThan(ratio(2), 1e3 * ratio(1));
        end

        function testWrongLayoutGivesHelpfulError(testCase)
            layer = OceanDL.Recurrent.RecurrentLayer(OceanDL.Recurrent.ElmanCell(3, 4));
            testCase.verifyError(@() layer.forward(randn(5, 3, 2)), ...
                'OceanDL:Recurrent:RecurrentLayer:InputSize');
        end

        function testLstmLearnsLongRangeMemoryTask(testCase)
            % Class = sign of the FIRST input, followed by 14 noise steps: only a
            % network that carries information across the whole sequence can
            % beat chance (50%).
            import OceanDL.Recurrent.*
            s = RandStream('mt19937ar', 'Seed', 7);
            T = 15; N = 64;
            rnn = RecurrentLayer(LSTMCell(1, 12, 'Seed', 8));
            head = OceanDL.nn.Linear(12, 2, 'Seed', 9);
            params = [rnn.parameters(), head.parameters()];
            opt = OceanDL.optim.ModuleOptimizer(params, @() OceanStats.Optim.AdamOptimizer('LR', 0.02));
            for it = 1:300                                   % measured: >= 97% from ~200 its on
                [X, y] = OceanTest.OceanDL.Recurrent.RecurrentLayerTest.memoryBatch(s, N, T);
                opt.zeroGrad();
                [~, st] = rnn.forward(X);
                L = OceanDL.Autograd.F.crossEntropy(head.forward(st.h), y);
                L.backward();
                OceanDL.nn.clipGradNorm(params, 1);
                opt.step();
            end
            [X, y] = OceanTest.OceanDL.Recurrent.RecurrentLayerTest.memoryBatch(s, 500, T);
            [~, st] = OceanDL.Autograd.GradMode.noGrad(@() rnn.forward(X));
            [~, pred] = max(head.forward(st.h).Data, [], 1);
            testCase.verifyGreaterThan(mean(pred == y), 0.95);
        end
    end

    methods (Static)
        function [X, y] = memoryBatch(s, N, T)
            X = 0.5 * randn(s, 1, N, T);
            y = 1 + (X(1, :, 1) > 0);                   % label from step 1 only
        end
    end
end
