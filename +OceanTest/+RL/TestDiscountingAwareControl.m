classdef TestDiscountingAwareControl < matlab.unittest.TestCase
    % TESTDISCOUNTINGAWARECONTROL Unit test suite for Exercise 5.14 Discounting-Aware Control.
    %
    % Verifies gamma = 1 equivalence with standard OffPolicyControl, gamma < 1
    % control learning, and 3-Pillar Component Contract compliance.

    methods (Test)
        function testGammaOneEquivalence(testCase)
            % When gamma = 1.0, Discounting-Aware Control must produce equivalent
            % Q-value estimates to standard Off-Policy Control.
            env = OceanStats.RL.MonteCarlo.BlackjackEnv();

            % 1. Standard Off-Policy Control
            solver1 = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
                'Algorithm',        "OffPolicyControl", ...
                'Gamma',            1.0,                ...
                'NumEpisodes',      1000,               ...
                'DiscountingAware', false,              ...
                'Epsilon',          0.1,                ...
                'Verbose',          false               ...
            );
            rng(42);
            [~, Q1, pi1, ~] = solver1.run();

            % 2. Discounting-Aware Control (gamma = 1.0)
            solver2 = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
                'Algorithm',        "DiscountingAwareControl", ...
                'Gamma',            1.0,                       ...
                'NumEpisodes',      1000,                      ...
                'DiscountingAware', true,                      ...
                'Epsilon',          0.1,                       ...
                'Verbose',          false                      ...
            );
            rng(42);
            [~, Q2, pi2, ~] = solver2.run();

            % Verification
            testCase.verifyEqual(Q1, Q2, 'RelTol', 1e-10);
            testCase.verifyEqual(pi1, pi2);
        end

        function testDiscountingAwareControlLearning(testCase)
            % For gamma < 1.0, DiscountingAwareControl must run smoothly and learn
            env = OceanStats.RL.MonteCarlo.BlackjackEnv();
            solver = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
                'Algorithm',        "DiscountingAwareControl", ...
                'Gamma',            0.90,                      ...
                'NumEpisodes',      2000,                      ...
                'Epsilon',          0.1,                       ...
                'Verbose',          false                      ...
            );
            rng(123);
            [V, Q, pi, history] = solver.run();

            testCase.verifyGreaterThan(numel(Q), 0);
            testCase.verifyGreaterThan(numel(pi), 0);
            testCase.verifyNotEmpty(history.AvgReturn);
        end

        function testStrategyMetadataAndSpec(testCase)
            strat = OceanStats.RL.MonteCarlo.Strategies.DiscountingAwareControlStrategy();
            meta  = strat.getMetadata();
            spec  = strat.getPipelineSpec();

            testCase.verifyEqual(meta.ClassName, "OceanStats.RL.MonteCarlo.Strategies.DiscountingAwareControlStrategy");
            testCase.verifyEqual(spec.ComponentName, "DiscountingAwareControlStrategy");
            testCase.verifyEqual(spec.Kind, "VIRTUAL");
        end
    end
end
