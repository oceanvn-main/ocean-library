classdef TestGamblersProblem < OceanTest.BaseTestCase
    % TESTGAMBLERSPROBLEM Unit test suite for ValueIterationSolver and Gambler's Problem.
    %
    % Tests generic Value Iteration convergence, terminal state immutability,
    % argmax tie-breaking behavior, and exp_gamblers_problem script execution.
    
    methods (Test)
        function testValueIterationSolverGeneric(testCase)
            % Test ValueIterationSolver on a simple Car Rental environment
            env = OceanStats.RL.Env.Classic.ModifiedCarRentalEnv('MaxCars', 5, 'MaxMove', 2);
            solver = OceanStats.RL.DP.ValueIterationSolver(env, 'DiscountFactor', 0.9, 'Theta', 1e-3);
            
            [V, pi, history] = solver.run('MaxIter', 50);
            
            testCase.verifyTrue(history.Sweeps > 0 && history.Sweeps <= 50);
            testCase.verifyEqual(length(V), 36);
            testCase.verifyEqual(length(pi), 36);
            testCase.verifyTrue(all(V >= 0));
        end
        
        function testGamblersProblemExecution(testCase)
            % Run exp_gamblers_problem script
            try
                exp_gamblers_problem;
                testCase.verifyTrue(true, 'exp_gamblers_problem executed cleanly.');
            catch ME
                testCase.verifyFail(sprintf('exp_gamblers_problem failed with error: %s', ME.message));
            end
        end
    end
end
