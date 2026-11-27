classdef LegacyEquivalenceTest < OceanTest.BaseTestCase
    % LEGACYEQUIVALENCETEST  The new Tensor autograd must agree with the
    %   repo's existing, already-verified machinery:
    %     - native Conv2d op  ==  legacy Conv.Conv2D forward/backward
    %     - legacy parameter-free layers run inside the new graph (nn.Legacy)
    %     - a whole CNN: gradient from the NEW engine == gradient from the
    %       OLD Tape-based CNNEngine, on identical weights.

    methods (Test)
        function testConv2dMatchesLegacyConv2D(testCase)
            s = RandStream('mt19937ar', 'Seed', 1);
            X = randn(s, 7, 6, 3, 2);
            W = randn(s, 3, 3, 3, 4); b = randn(s, 4, 1);
            legacy = OceanDL.Conv.Conv2D([3 3], 4, 'InChannels', 3, ...
                'Stride', [2 1], 'Padding', [1 1], 'Weight', W, 'Bias', b);
            Yold = legacy.forward(X);
            G = randn(s, size(Yold));
            dXold = legacy.backward(G);
            [dWold, dbold] = legacy.getWeightGrad();

            x = OceanDL.Autograd.Tensor(X, 'RequiresGrad', true);
            w = OceanDL.Autograd.Tensor(W, 'RequiresGrad', true);
            bb = OceanDL.Autograd.Tensor(b, 'RequiresGrad', true);
            Y = OceanDL.Autograd.F.conv2d(x, w, bb, 'Stride', [2 1], 'Padding', [1 1]);
            Y.backward(G);
            testCase.verifyEqual(Y.Data, Yold, 'AbsTol', 1e-12);
            testCase.verifyEqual(x.Grad, dXold, 'AbsTol', 1e-12);
            testCase.verifyEqual(w.Grad, dWold, 'AbsTol', 1e-12);
            testCase.verifyEqual(bb.Grad, dbold, 'AbsTol', 1e-12);
        end

        function testLegacyMaxPoolInsideGraph(testCase)
            pool = OceanDL.nn.Legacy(OceanDL.Pooling.MaxPool2D([2 2]));
            s = RandStream('mt19937ar', 'Seed', 2);
            X = randn(s, 4, 4, 2, 1) + reshape(0:0.01:0.31, 4, 4, 2);   % no ties
            r = OceanDL.Autograd.GradCheck.check(@(x) pool.forward(x) .^ 2, {X});
            testCase.verifyTrue(r.Passed);
        end

        function testLegacyStaleCacheIsDetected(testCase)
            pool = OceanDL.nn.Legacy(OceanDL.Pooling.MaxPool2D([2 2]));
            x = OceanDL.Autograd.Tensor(randn(4, 4, 1, 1), 'RequiresGrad', true);
            y1 = pool.forward(x);
            y2 = pool.forward(x .* 2); %#ok<NASGU>            % overwrites the layer's cache
            testCase.verifyError(@() backward(sum(y1, 'all')), ...
                'OceanDL:Autograd:LegacyLayer:StaleCache');
        end

        function testWholeCnnGradientMatchesTapeEngine(testCase)
            import OceanDL.Conv.CNNTopologySpec
            import OceanDL.Autograd.*
            spec = CNNTopologySpec.make({ ...
                CNNTopologySpec.conv2d([3 3], 2, 'Padding', [1 1]), CNNTopologySpec.relu(), ...
                CNNTopologySpec.maxpool2d([2 2]), ...
                CNNTopologySpec.conv2d([3 3], 3, 'Padding', [1 1]), CNNTopologySpec.relu(), ...
                CNNTopologySpec.flatten(), CNNTopologySpec.dense(4), CNNTopologySpec.softmax()}, ...
                'InputChannels', 1, 'LayerNames', ...
                {'conv1', 'act1', 'pool1', 'conv2', 'act2', 'flat', 'dense1', 'smax'});
            rng(11);
            X = randn(6, 6, 1, 2);
            eng = OceanDL.Conv.CNNEngine(spec);
            Yold = eng.forward(X);
            seed = randn(size(Yold));
            dXold = eng.gradient('smax', 'input', seed);

            c1 = eng.getLayerObject('conv1'); c2 = eng.getLayerObject('conv2'); d1 = eng.getLayerObject('dense1');
            pool = OceanDL.nn.Legacy(OceanDL.Pooling.MaxPool2D([2 2]));
            flat = OceanDL.nn.Legacy(OceanDL.Conv.Flatten());
            x = Tensor(X, 'RequiresGrad', true);
            h = relu(F.conv2d(x, c1.Weight, c1.Bias, 'Padding', [1 1]));
            h = relu(F.conv2d(pool.forward(h), c2.Weight, c2.Bias, 'Padding', [1 1]));
            Y = F.softmax(F.linear(flat.forward(h), d1.Weight, d1.Bias), 1);
            Y.backward(seed);

            testCase.verifyEqual(Y.Data, Yold, 'AbsTol', 1e-12);
            testCase.verifyEqual(x.Grad, dXold, 'AbsTol', 1e-12);
        end
    end
end
