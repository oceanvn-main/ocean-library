classdef test_regularization < OceanTest.BaseTestCase
    % TEST_REGULARIZATION Unit tests for L1, L2, and CompositeObjective.
    %
    % Author: Ocean Library | 2026-03-13

    methods (Test)
        function testL2Regularizer(testCase)
            reg = OceanStats.Regularization.L2Regularizer();
            w = [2; -4];
            
            % 1. Evaluate
            % 0.5 * (4 + 16) = 10
            testCase.verifyEqual(reg.evaluate(w), 10.0);
            
            % 2. Gradient
            % grad = w
            testCase.verifyEqual(reg.gradient(w), [2; -4]);
            
            % 3. Proximity
            % w / (1 + t*lambda) = [2; -4] / (1 + 0.1 * 10) = [2; -4] / 2 = [1; -2]
            w_prox = reg.proximity(w, 0.1, 10.0);
            testCase.verifyEqual(w_prox, [1; -2]);
        end

        function testL1Regularizer(testCase)
            reg = OceanStats.Regularization.L1Regularizer();
            w = [2; -0.5; 0];
            
            % 1. Evaluate
            % 2 + 0.5 + 0 = 2.5
            testCase.verifyEqual(reg.evaluate(w), 2.5);
            
            % 2. Subgradient
            % sign(w)
            testCase.verifyEqual(reg.gradient(w), [1; -1; 0]);
            
            % 3. Proximity (Soft Thresholding)
            % threshold = t * lambda = 0.1 * 10 = 1.0
            % max(|w| - 1.0, 0) * sign(w)
            % 2 -> max(2-1,0)*1 = 1
            % -0.5 -> max(0.5-1,0)*-1 = 0
            % 0 -> max(0-1,0)*0 = 0
            w_prox = reg.proximity(w, 0.1, 10.0);
            testCase.verifyEqual(w_prox, [1; 0; 0]);
        end
        
        function testCompositeObjectiveEval(testCase)
            % Test that loss and regularization combine correctly
            loss = OceanStats.Loss.Regression.MSELoss();
            reg = OceanStats.Regularization.L2Regularizer();
            objFunc = OceanStats.Optimization.CompositeObjective(loss, reg, 0.1);
            
            w = [1; 2];
            Y_true = [10; 20];
            Y_pred = [8; 20];
            
            % MSE Loss = mean((Y_true - Y_pred).^2) = mean([4; 0]) = 2.0
            % L2 Reg = 0.5 * (1 + 4) = 2.5
            % Total = 2.0 + 0.1 * 2.5 = 2.25
            
            val = objFunc.evaluate(Y_true, Y_pred, w);
            testCase.verifyEqual(val, 2.25);
        end
    end
end
