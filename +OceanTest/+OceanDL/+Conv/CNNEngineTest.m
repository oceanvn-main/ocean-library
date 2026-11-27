classdef CNNEngineTest < OceanTest.BaseTestCase
    % CNNENGINETEST  Integration tests for OceanDL.Conv.CNNEngine:
    %   a full toy CNN (conv-relu-pool-conv-relu-flatten-dense-softmax)
    %   wired through the Autograd Tape, gradchecked end to end against
    %   finite differences -- this is the "crux" verification called for
    %   by the project plan (robust autograd, not four hand-rolled cases).

    properties
        Spec
    end

    methods (TestClassSetup)
        function buildSpec(testCase)
            import OceanDL.Conv.CNNTopologySpec
            testCase.Spec = CNNTopologySpec.make({ ...
                CNNTopologySpec.conv2d([3 3], 2, 'Padding', [1 1]), ...
                CNNTopologySpec.relu(), ...
                CNNTopologySpec.maxpool2d([2 2]), ...
                CNNTopologySpec.conv2d([3 3], 3, 'Padding', [1 1]), ...
                CNNTopologySpec.relu(), ...
                CNNTopologySpec.flatten(), ...
                CNNTopologySpec.dense(4), ...
                CNNTopologySpec.softmax() ...
            }, 'InputChannels', 1, 'LayerNames', ...
                {'conv1', 'act1', 'pool1', 'conv2', 'act2', 'flat', 'dense1', 'smax'});
        end
    end

    methods (Test)
        function testForwardShapesThroughFullStack(testCase)
            import OceanDL.Conv.CNNEngine
            eng = CNNEngine(testCase.Spec);
            eng.forward(zeros(6, 6, 1, 2));   % N=2 batch

            testCase.verifySize(eng.getActivations('conv1'), [6 6 2 2]);
            testCase.verifySize(eng.getActivations('pool1'), [3 3 2 2]);
            testCase.verifySize(eng.getActivations('conv2'), [3 3 3 2]);
            testCase.verifySize(eng.getActivations('flat'), [27 2]);
            Y = eng.getActivations('smax');
            testCase.verifySize(Y, [4 2]);
            testCase.verifyEqual(sum(Y, 1), ones(1, 2), 'AbsTol', 1e-10); % softmax columns sum to 1
        end

        function testGetActivationsBeforeForwardErrors(testCase)
            import OceanDL.Conv.CNNEngine
            eng = CNNEngine(testCase.Spec);
            testCase.verifyError(@() eng.getActivations('conv1'), ...
                'OceanDL:Conv:CNNEngine:NoForwardCache');
        end

        function testUnknownLayerNameErrors(testCase)
            import OceanDL.Conv.CNNEngine
            eng = CNNEngine(testCase.Spec);
            eng.forward(zeros(6, 6, 1, 1));
            testCase.verifyError(@() eng.getActivations('ghost'), ...
                'OceanDL:Conv:CNNEngine:UnknownLayer');
        end

        function testEndToEndInputGradcheckFull(testCase)
            % loss = sum(softmax_output.^2, 'all'); dLoss/dInput via the
            % Tape (seeded with 2*Y) vs. central finite differences on the
            % ACTUAL scalar loss, recomputed by a full forward() each time.
            import OceanDL.Conv.CNNEngine
            rng(99);
            X = randn(6, 6, 1, 1);
            eng = CNNEngine(testCase.Spec);
            Y = eng.forward(X);
            dX = eng.gradient('smax', 'input', 2 * Y);

            lossOf = @(Xv) sum(eng.forward(Xv).^2, 'all');

            eps_ = 1e-4;
            dXnum = zeros(size(X));
            for k = 1:numel(X)
                Xp = X; Xp(k) = Xp(k) + eps_;
                Xm = X; Xm(k) = Xm(k) - eps_;
                dXnum(k) = (lossOf(Xp) - lossOf(Xm)) / (2 * eps_);
            end
            testCase.verifyEqual(dX, dXnum, 'AbsTol', 1e-3);
        end

        function testGradientWithRespectToIntermediateConvLayer(testCase)
            % Grad-CAM-shaped query: gradient of the (pre-softmax-consuming)
            % loss w.r.t. an INTERMEDIATE conv layer's output feature map,
            % gradchecked by manually replaying the tail of the network
            % (flat -> dense1 -> softmax) on a perturbed intermediate.
            import OceanDL.Conv.CNNEngine
            rng(7);
            X = randn(6, 6, 1, 1);
            eng = CNNEngine(testCase.Spec);
            Y = eng.forward(X);
            A = eng.getActivations('conv2');          % (3 x 3 x 3 x 1)

            dA = eng.gradient('smax', 'conv2', 2 * Y); % seeded with dLoss/dY

            tailLoss = @(Av) OceanTest.OceanDL.Conv.CNNEngineTest.tailLoss(eng, Av);

            eps_ = 1e-4;
            dAnum = zeros(size(A));
            for k = 1:numel(A)
                Ap = A; Ap(k) = Ap(k) + eps_;
                Am = A; Am(k) = Am(k) - eps_;
                dAnum(k) = (tailLoss(Ap) - tailLoss(Am)) / (2 * eps_);
            end
            testCase.verifyEqual(dA, dAnum, 'AbsTol', 1e-3);
        end
    end

    methods (Static, Access = private)
        function L = tailLoss(eng, A)
            % Must replay the REAL graph tail exactly: conv2 -> act2(relu)
            % -> flat -> dense1 -> softmax. 'conv2' is the PRE-activation
            % node, so the relu here is not optional.
            flat = eng.getLayerObject('flat');
            dense1 = eng.getLayerObject('dense1');
            act2 = OceanDL.Activation.ActivationFactory.create('relu');
            a = flat.forward(act2.forward(A));
            z = dense1.forward(a);
            Y = OceanDL.Activation.SoftmaxActivation().forward(z);
            L = sum(Y.^2, 'all');
        end
    end
end
