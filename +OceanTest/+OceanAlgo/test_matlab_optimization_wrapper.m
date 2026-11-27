classdef test_matlab_optimization_wrapper < OceanTest.BaseTestCase
    % TEST_MATLAB_OPTIMIZATION_WRAPPER Unit tests for MatlabOptimizationWrapper.
    %
    % AUTHOR: Ocean Library System
    % DATE: 2026-09-03
    
    methods (Test)
        function testSteepestDescentQuadratic(testCase)
            % Minimize f(x) = (x1 - 3)^2 + (x2 + 2)^2
            quadFun = @(x) (x(1) - 3)^2 + (x(2) + 2)^2;
            x0 = [0, 0];
            
            wrapper = OceanAlgo.Optimization.MatlabOptimizationWrapper();
            [x_opt, f_opt, history] = wrapper.solve(quadFun, x0, ...
                'Method', 'steepdesc', 'IsMaximization', false);
            
            testCase.assertMatrixEquals(x_opt, [3, -2], 1e-4, 'Steepest descent quadratic optimization');
            testCase.verifyLessThan(abs(f_opt), 1e-5);
            testCase.verifyGreaterThan(history.iterations, 0);
        end
        
        function testDFPQuadratic(testCase)
            quadFun = @(x) (x(1) - 3)^2 + (x(2) + 2)^2;
            x0 = [0, 0];
            
            wrapper = OceanAlgo.Optimization.MatlabOptimizationWrapper();
            [x_opt, f_opt, history] = wrapper.solve(quadFun, x0, ...
                'Method', 'dfp', 'IsMaximization', false);
            
            testCase.assertMatrixEquals(x_opt, [3, -2], 1e-4, 'DFP quadratic optimization');
            testCase.verifyLessThan(abs(f_opt), 1e-5);
            testCase.verifyGreaterThan(history.iterations, 0);
        end
        
        function testBFGSQuadratic(testCase)
            quadFun = @(x) (x(1) - 3)^2 + (x(2) + 2)^2;
            x0 = [0, 0];
            
            wrapper = OceanAlgo.Optimization.MatlabOptimizationWrapper();
            [x_opt, f_opt, history] = wrapper.solve(quadFun, x0, ...
                'Method', 'bfgs', 'IsMaximization', false);
            
            testCase.assertMatrixEquals(x_opt, [3, -2], 1e-4, 'BFGS quadratic optimization');
            testCase.verifyLessThan(abs(f_opt), 1e-5);
            testCase.verifyGreaterThan(history.iterations, 0);
        end
        
        function testNewtonQuadratic(testCase)
            quadFun = @(x) (x(1) - 3)^2 + (x(2) + 2)^2;
            x0 = [0, 0];
            
            wrapper = OceanAlgo.Optimization.MatlabOptimizationWrapper();
            [x_opt, f_opt, history] = wrapper.solve(quadFun, x0, ...
                'Method', 'newton', 'IsMaximization', false);
            
            testCase.assertMatrixEquals(x_opt, [3, -2], 1e-4, 'Newton quadratic optimization');
            testCase.verifyLessThan(abs(f_opt), 1e-5);
            testCase.verifyGreaterThan(history.iterations, 0);
        end
        
        function testTelemetryFields(testCase)
            quadFun = @(x) (x(1) - 1)^2 + (x(2) - 1)^2;
            x0 = [0, 0];
            
            wrapper = OceanAlgo.Optimization.MatlabOptimizationWrapper();
            [~, ~, history] = wrapper.solve(quadFun, x0, 'Method', 'bfgs', 'IsMaximization', false);
            
            testCase.verifyTrue(isfield(history, 'x'));
            testCase.verifyTrue(isfield(history, 'f'));
            testCase.verifyTrue(isfield(history, 'grad_norm'));
            testCase.verifyTrue(isfield(history, 'step_norm'));
            testCase.verifyTrue(isfield(history, 'iterations'));
            testCase.verifyEqual(size(history.x, 2), 2);
            testCase.verifyEqual(length(history.f), size(history.x, 1));
        end
        
        function testARMALikelihoodIntegration(testCase)
            % Test likelihood maximization on synthetic series
            y = OceanStats.Time.ARMAProcess.simulate(0.6, -0.3, 1.0, 300, 42);
            likeFun = @(p) OceanStats.Time.ARMALikelihood.logLikelihood(p, y, 1.0);
            
            wrapper = OceanAlgo.Optimization.MatlabOptimizationWrapper();
            [p_opt, f_mle, history] = wrapper.solve(likeFun, [0.0, 0.0], ...
                'Method', 'bfgs', 'IsMaximization', true);
            
            testCase.verifyEqual(length(p_opt), 2);
            testCase.verifyGreaterThan(f_mle, -1000);
            testCase.verifyGreaterThan(history.iterations, 0);
        end
    end
end
