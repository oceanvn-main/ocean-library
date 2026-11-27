classdef ModuleTest < OceanTest.BaseTestCase
    % MODULETEST  nn.Module parameter discovery / state dict / modes, the
    %   standard modules, gradient clipping, and a real training run
    %   (zeroGrad -> forward -> backward -> step) that must actually learn.

    methods (Test)
        function testSequentialDiscoversNamedParameters(testCase)
            net = OceanDL.nn.Sequential( ...
                OceanDL.nn.Linear(2, 5), OceanDL.nn.Activation("tanh"), ...
                OceanDL.nn.Linear(5, 1));
            named = net.namedParameters();
            testCase.verifyEqual({named.Name}, ...
                {'Layers{1}.Weight', 'Layers{1}.Bias', 'Layers{3}.Weight', 'Layers{3}.Bias'});
            testCase.verifyEqual(net.numParameters(), 2*5 + 5 + 5 + 1);
        end

        function testSharedModuleCountedOnce(testCase)
            fc = OceanDL.nn.Linear(3, 3);
            net = OceanDL.nn.Sequential(fc, OceanDL.nn.Activation("relu"), fc);
            testCase.verifyEqual(numel(net.parameters()), 2);
        end

        function testStateDictRoundTripAndShapeCheck(testCase)
            a = OceanDL.nn.Linear(3, 2, 'Seed', 1);
            b = OceanDL.nn.Linear(3, 2, 'Seed', 2);
            b.loadStateDict(a.stateDict());
            testCase.verifyEqual(b.Weight.Data, a.Weight.Data);
            bad = containers.Map({'Weight', 'Bias'}, {zeros(2, 4), zeros(2, 1)});
            testCase.verifyError(@() b.loadStateDict(bad), 'OceanDL:nn:Module:ShapeMismatch');
        end

        function testEvalModeDisablesDropout(testCase)
            net = OceanDL.nn.Sequential(OceanDL.nn.Dropout(0.5));
            x = ones(4, 100);
            yTrain = net.forward(x);
            testCase.verifyTrue(any(yTrain.Data(:) == 0) && any(yTrain.Data(:) == 2));   % inverted scaling 1/(1-p)
            net.eval();
            testCase.verifyEqual(net.forward(x).Data, x);
            testCase.verifyFalse(net.Layers{1}.Training);
        end

        function testLinearFoldsTimeDimension(testCase)
            fc = OceanDL.nn.Linear(3, 2, 'Seed', 4);
            X = randn(3, 5, 7);                                   % D x N x T
            Y = fc.forward(X);
            testCase.verifySize(Y.Data, [2 5 7]);
            testCase.verifyEqual(Y.Data(:, :, 4), fc.Weight.Data * X(:, :, 4) + fc.Bias.Data, 'AbsTol', 1e-12);
        end

        function testClipGradNormPreservesDirection(testCase)
            p = OceanDL.nn.Parameter([0 0]);
            q = OceanDL.nn.Parameter(0);
            p.Grad = [3 0]; q.Grad = 4;                           % global norm 5
            n = OceanDL.nn.clipGradNorm({p, q}, 1);
            testCase.verifyEqual(n, 5);
            testCase.verifyEqual([p.Grad q.Grad], [3 0 4] / 5, 'AbsTol', 1e-6);
        end

        function testTrainingLearnsXor(testCase)
            % Real optimisation through every layer: XOR is not linearly
            % separable, so this only succeeds if hidden-layer gradients are right.
            X = [0 0 1 1; 0 1 0 1];
            y = [1 2 2 1];                                        % class 1 = even parity
            net = OceanDL.nn.Sequential( ...
                OceanDL.nn.Linear(2, 8, 'Seed', 1), OceanDL.nn.Activation("tanh"), ...
                OceanDL.nn.Linear(8, 2, 'Seed', 2));
            opt = OceanDL.optim.ModuleOptimizer(net.parameters(), @() OceanStats.Optim.AdamOptimizer('LR', 0.05));
            for it = 1:300
                opt.zeroGrad();
                L = OceanDL.Autograd.F.crossEntropy(net.forward(X), y);
                L.backward();
                opt.step();
            end
            testCase.verifyLessThan(L.Data, 0.05);
            [~, pred] = max(net.forward(X).Data, [], 1);
            testCase.verifyEqual(pred, y);
        end

        function testEmbeddingModule(testCase)
            emb = OceanDL.nn.Embedding(10, 4, 'Seed', 1);
            E = emb.forward([1 2 3; 4 5 6]);
            testCase.verifySize(E.Data, [4 2 3]);
            testCase.verifyEqual(E.Data(:, 2, 3), emb.Weight.Data(:, 6));
            testCase.verifyError(@() emb.forward(11), 'OceanDL:nn:Embedding:BadIndex');
        end

        function testConv2dModuleShapes(testCase)
            conv = OceanDL.nn.Conv2d(3, 8, 3, 'Padding', 1, 'Stride', 2, 'Seed', 1);
            Y = conv.forward(randn(9, 9, 3, 2));
            testCase.verifySize(Y.Data, [5 5 8 2]);
        end
    end
end
