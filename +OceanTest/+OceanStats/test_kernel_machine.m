classdef test_kernel_machine < OceanTest.BaseTestCase
    % TEST_KERNEL_MACHINE  Unit tests for KernelMachine and SVMClassifier wrapper.
    %
    %   Author: Ocean Library | 2026-03-13

    properties
        X_train
        Y_train
        X_test
        Y_test
        Machine
    end

    methods (TestClassSetup)
        function generateData(testCase)
            % Generate simple linearly separable 2D data
            rng(42);
            X1 = randn(50, 2) + [2 2];
            X2 = randn(50, 2) - [2 2];
            testCase.X_train = [X1; X2];
            testCase.Y_train = [ones(50, 1); -ones(50, 1)];

            X1_te = randn(20, 2) + [2 2];
            X2_te = randn(20, 2) - [2 2];
            testCase.X_test = [X1_te; X2_te];
            testCase.Y_test = [ones(20, 1); -ones(20, 1)];
            
            testCase.Machine = OceanStats.Supervised.KernelMethods.KernelMachine(...
                'C', 1.0, ...
                'Kernel', OceanStats.Supervised.SVM.Kernels.LinearKernel());
        end
    end

    methods (Test)

        function testKernelMachineTrainPredict(testCase)
            % Should find a perfect boundary on this easy dataset
            model = testCase.Machine.train(testCase.X_train, testCase.Y_train);
            Y_hat = testCase.Machine.predict(model, testCase.X_test);
            
            acc = mean(Y_hat == testCase.Y_test);
            testCase.verifyEqual(acc, 1.0, 'KernelMachine should achieve 100% accuracy on easy data.');
            testCase.verifyGreaterThan(model.nSV, 0, 'Should have some support vectors.');
        end

        function testSVMClassifierWrapper(testCase)
            % The wrapper should behave identically
            svm = OceanStats.Supervised.SVM.SVMClassifier('C', 1.0, ...
                'Kernel', OceanStats.Supervised.SVM.Kernels.LinearKernel());
            
            model = svm.train(testCase.X_train, testCase.Y_train);
            Y_hat = svm.predict(model, testCase.X_test);
            
            acc = 1.0 - svm.evaluate(testCase.Y_test, Y_hat);
            testCase.verifyEqual(acc, 1.0, 'SVMClassifier wrapper should achieve 100% accuracy.');
        end
        
        function testSVMLabelValidation(testCase)
            % Should auto-remap {0,1} to {-1,+1}
            svm = OceanStats.Supervised.SVM.SVMClassifier('C', 1.0);
            
            % Map -1 to 0
            Y_01 = testCase.Y_train;
            Y_01(Y_01 == -1) = 0;
            
            % Verify warning is thrown about remapping
            testCase.verifyWarning(@() svm.train(testCase.X_train, Y_01), ...
                'OceanStats:Supervised:SVM:SVMClassifier:LabelRemapped');
        end

        function testCompatibleLossEnforcement(testCase)
            % SMOSolver should throw error if given an incompatible loss (e.g. MSE)
            badMachine = OceanStats.Supervised.KernelMethods.KernelMachine(...
                'Loss', OceanStats.Loss.Regression.MSELoss(), ...
                'Solver', OceanStats.Supervised.KernelMethods.SMOSolver());
            
            testCase.verifyError(@() badMachine.train(testCase.X_train, testCase.Y_train), ...
                'OceanStats:KernelMethods:SMOSolver:IncompatibleLoss');
        end
        
        function testRBFKernelIntegration(testCase)
            % Testing KernelMachine with another kernel
            rbfMachine = OceanStats.Supervised.KernelMethods.KernelMachine(...
                'C', 10.0, ...
                'Kernel', OceanStats.Supervised.SVM.Kernels.RBFKernel('Gamma', 0.1));
            
            model = rbfMachine.train(testCase.X_train, testCase.Y_train);
            Y_hat = rbfMachine.predict(model, testCase.X_test);
            
            acc = mean(Y_hat == testCase.Y_test);
            testCase.verifyGreaterThan(acc, 0.9, 'RBF KernelMachine should learn the boundary.');
        end
    end
end
