classdef TestMonteCarloSolverRefactor < matlab.unittest.TestCase
    % TESTMONTECARLOSOLVERREFACTOR Regression tests for MonteCarloSolver refactoring.
    %
    % Verifies that MonteCarloSolver operates correctly using the generic
    % WeightedMeanAccumulator engine.
    %
    % Package: OceanTest.RL

    methods (Test)
        function testFirstVisitPrediction(testCase)
            env = OceanStats.RL.MonteCarlo.BlackjackEnv();
            solver = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
                'Algorithm', 'FirstVisitPrediction', ...
                'NumEpisodes', 100, ...
                'Verbose', false ...
            );

            [V, Q, pi_pol, hist] = solver.run();

            testCase.verifySize(V, [200 1]);
            testCase.verifySize(Q, [200 2]);
            testCase.verifySize(pi_pol, [200 1]);
            testCase.verifyNotEmpty(hist.CheckPoints);
        end

        function testOffPolicyPrediction(testCase)
            env = OceanStats.RL.MonteCarlo.BlackjackEnv();
            target_policy   = @(~, a_idx) double(a_idx == 2); % Always HIT
            behavior_policy = @(~, ~) 0.5;                   % 50/50 random

            solver = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
                'Algorithm', 'OffPolicyPrediction', ...
                'TargetPolicy', target_policy, ...
                'BehaviorPolicy', behavior_policy, ...
                'ISType', 'Weighted', ...
                'NumEpisodes', 100, ...
                'Verbose', false ...
            );

            [V, Q, pi_pol, hist] = solver.run();

            testCase.verifySize(V, [200 1]);
            testCase.verifyNotEmpty(hist.CheckPoints);
        end

        function testMetadataReflection(testCase)
            meta = OceanStats.RL.MonteCarlo.MonteCarloSolver.getMetadata();
            testCase.verifyEqual(meta.ClassName, "OceanStats.RL.MonteCarlo.MonteCarloSolver");
            testCase.verifyNotEmpty(meta.SymbolicLatex);
            testCase.verifyNotEmpty(meta.Symbols);
            testCase.verifyGreaterThanOrEqual(numel(meta.Symbols), 5);
        end

        function testPipelineSpec(testCase)
            spec = OceanStats.RL.MonteCarlo.MonteCarloSolver.getPipelineSpec();
            testCase.verifyEqual(spec.ComponentName, "MonteCarloSolver");
            testCase.verifyEqual(spec.Kind, "NONVIRTUAL");
            testCase.verifyNotEmpty(spec.Inputs);
            testCase.verifyNotEmpty(spec.Outputs);
        end
    end
end


