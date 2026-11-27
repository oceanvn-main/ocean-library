classdef TestClassificationLossesMissing < OceanTest.BaseTestCase
    methods (Test)
        function testSetupTestData(testCase)
            % Setup Test Data
            import OceanStats.Loss.Classification.*
            % Fix Y_true to +1 for simple margin calculations (margin = Y_true * Y_pred)
            % Since y=1, margin = Y_pred.
            Y_true = [1; 1; 1; 1; 1; 1];
            Y_pred = [-2; -1; 0; 0.5; 1; 2];
        end

        function testTestMarginsquaredloss(testCase)
            % Test MarginSquaredLoss
            import OceanStats.Loss.Classification.*
            fprintf('Testing MarginSquaredLoss...\n');
            loss_ms = MarginSquaredLoss();
            % Expected forward: (1 - margin)^2
            L_exp_ms = mean([9; 4; 1; 0.25; 0; 1]);
            L_act_ms = loss_ms.forward(Y_true, Y_pred);
            assert(abs(L_exp_ms - L_act_ms) < 1e-10, 'MarginSquaredLoss: forward failed');
            % Expected gradient: -2*y*(1 - margin) = -2(1 - margin)
            g_exp_ms = [-6; -4; -2; -1; 0; 2];
            g_act_ms = loss_ms.gradient(Y_true, Y_pred);
            assert(max(abs(g_exp_ms - g_act_ms)) < 1e-10, 'MarginSquaredLoss: gradient failed');
            % Expected hessian: 2
            h_exp_ms = [2; 2; 2; 2; 2; 2];
            h_act_ms = loss_ms.hessian(Y_true, Y_pred);
            assert(max(abs(h_exp_ms - h_act_ms)) < 1e-10, 'MarginSquaredLoss: hessian failed');
            fprintf('  MarginSquaredLoss passed.\n\n');
        end

        function testTestHuberisedsquarehingeloss(testCase)
            % Test HuberisedSquareHingeLoss
            import OceanStats.Loss.Classification.*
            fprintf('Testing HuberisedSquareHingeLoss...\n');
            loss_hsh = HuberisedSquareHingeLoss();
            % Expected forward:
            % m < -1: -4m
            % m >= -1: max(0, 1 - m)^2
            % margins: -2, -1, 0, 0.5, 1, 2
            % values : 8, 4, 1, 0.25, 0, 0
            L_exp_hsh = mean([8; 4; 1; 0.25; 0; 0]);
            L_act_hsh = loss_hsh.forward(Y_true, Y_pred);
            assert(abs(L_exp_hsh - L_act_hsh) < 1e-10, 'HuberisedSquareHingeLoss: forward failed');
            % Expected gradient:
            % m < -1: -4y = -4
            % -1 <= m < 1: -2y(1-m)
            % m >= 1: 0
            g_exp_hsh = [-4; -4; -2; -1; 0; 0];
            g_act_hsh = loss_hsh.gradient(Y_true, Y_pred);
            assert(max(abs(g_exp_hsh - g_act_hsh)) < 1e-10, 'HuberisedSquareHingeLoss: gradient failed');
            % Expected hessian:
            % -1 <= m < 1: 2
            % otherwise: 0
            h_exp_hsh = [0; 2; 2; 2; 0; 0];
            h_act_hsh = loss_hsh.hessian(Y_true, Y_pred);
            assert(max(abs(h_exp_hsh - h_act_hsh)) < 1e-10, 'HuberisedSquareHingeLoss: hessian failed');
            fprintf('  HuberisedSquareHingeLoss passed.\n\n');
            fprintf('All new loss functions passed successfully!\n');
        end

    end
end