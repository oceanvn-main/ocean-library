classdef TestVectorizedEngine < matlab.unittest.TestCase
    % TESTVECTORIZEDENGINE  Unit tests for VectorizedOptimizerEngine & ParameterGroup.

    methods (Test)
        function testParameterGroup(testCase)
            g1 = OceanStats.Optim.ParameterGroup('Backbone', (1:100)', 'LR', 1e-4, 'WeightDecay', 0.01);
            g2 = OceanStats.Optim.ParameterGroup('Head', (101:120)', 'LR', 1e-2, 'WeightDecay', 0.00);

            testCase.verifyEqual(g1.numParams(), 100);
            testCase.verifyEqual(g2.numParams(), 20);
            testCase.verifyEqual(g1.getEffectiveLR(), 1e-4);
            testCase.verifyEqual(g2.getEffectiveLR(), 1e-2);
        end

        function testVectorizedEngineStepping(testCase)
            adamOpt = OceanStats.Optim.AdamOptimizer('LR', 0.01);
            engine  = OceanStats.Optim.VectorizedOptimizerEngine(adamOpt);

            theta0 = [1.0; -0.5; 2.0; 0.3];
            grads  = [0.1; -0.2; 0.05; -0.4];

            [theta1, telemetry] = engine.stepVector(theta0, grads);

            testCase.verifyEqual(numel(theta1), 4);
            testCase.verifyEqual(telemetry.Step, 1);
            testCase.verifyTrue(telemetry.GradNorm > 0);
            testCase.verifyTrue(all(theta1 ~= theta0));
        end

        function testStateSerialization(testCase)
            adamOpt = OceanStats.Optim.AdamOptimizer('LR', 0.01);
            engine  = OceanStats.Optim.VectorizedOptimizerEngine(adamOpt);

            theta = [1.0; -0.5];
            grads = [0.1; -0.2];

            engine.stepVector(theta, grads);
            state = engine.get_state_dict();

            testCase.verifyEqual(state.StepCount, 1);

            engine2 = OceanStats.Optim.VectorizedOptimizerEngine(adamOpt);
            engine2.set_state_dict(state);
            testCase.verifyEqual(engine2.StepCount, 1);
        end
    end
end
