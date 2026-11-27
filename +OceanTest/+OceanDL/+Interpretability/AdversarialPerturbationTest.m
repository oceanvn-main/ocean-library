classdef AdversarialPerturbationTest < OceanTest.BaseTestCase
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
                CNNTopologySpec.flatten(), ...
                CNNTopologySpec.dense(3), ...
                CNNTopologySpec.softmax() ...
            }, 'InputChannels', 1, 'LayerNames', {'conv1', 'act1', 'pool1', 'flat', 'dense1', 'smax'});
        end
    end

    methods (Test)
        function testFGSMPerturbationIsBoundedByEpsilonAndClipped(testCase)
            import OceanDL.Conv.CNNEngine
            import OceanDL.Interpretability.AdversarialPerturbation

            rng(31);
            eng = CNNEngine(testCase.Spec);
            X = rand(6, 6, 1, 1);   % in [0,1]
            epsilon = 0.05;
            r = AdversarialPerturbation.fgsm(eng, X, 1, epsilon);

            testCase.verifyTrue(all(abs(r.Perturbation(:)) <= epsilon + 1e-10));
            testCase.verifyTrue(all(r.AdversarialImage(:) >= 0 - 1e-10));
            testCase.verifyTrue(all(r.AdversarialImage(:) <= 1 + 1e-10));
        end

        function testFGSMLowersTrueClassScoreForSmallEpsilon(testCase)
            % First-order argument: X += eps*sign(dX) ascends the loss
            % -1*log(Y_true) w.r.t. that class's own probability, so for a
            % small enough epsilon the true-class score should not
            % increase (generically decreases, barring an exact tie in a
            % degenerate direction).
            import OceanDL.Conv.CNNEngine
            import OceanDL.Interpretability.AdversarialPerturbation

            rng(32);
            eng = CNNEngine(testCase.Spec);
            X = rand(6, 6, 1, 1);
            r = AdversarialPerturbation.fgsm(eng, X, 1, 1e-3);
            testCase.verifyLessThanOrEqual(r.AdversarialScore, r.OriginalScore + 1e-9);
        end

        function testIterativeFGSMStaysWithinEpsilonBall(testCase)
            import OceanDL.Conv.CNNEngine
            import OceanDL.Interpretability.AdversarialPerturbation

            rng(33);
            eng = CNNEngine(testCase.Spec);
            X = rand(6, 6, 1, 1);
            epsilon = 0.1;
            r = AdversarialPerturbation.iterativeFGSM(eng, X, 2, epsilon, 'NumSteps', 5);

            testCase.verifyTrue(all(abs(r.Perturbation(:)) <= epsilon + 1e-8));
            testCase.verifyTrue(all(r.AdversarialImage(:) >= 0 - 1e-10));
            testCase.verifyTrue(all(r.AdversarialImage(:) <= 1 + 1e-10));
        end
    end
end
