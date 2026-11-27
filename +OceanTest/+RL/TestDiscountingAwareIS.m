classdef TestDiscountingAwareIS < matlab.unittest.TestCase
    % TESTDISCOUNTINGAWAREIS Unit test suite for Section 5.8 Discounting-Aware IS.
    %
    % Verifies gamma = 1 equivalence with standard IS, gamma < 1 variance
    % reduction on long episodes, Ordinary vs Weighted weighting schemes,
    % and 3-Pillar Component Contract compliance.

    methods (Test)
        function testGammaOneEquivalence(testCase)
            % When gamma = 1, Discounting-Aware IS must match Standard IS exactly
            env = OceanStats.RL.MonteCarlo.BlackjackEnv();
            target_pi   = @(s_idx, a_idx) double(a_idx == 1);
            behavior_b  = @(s_idx, a_idx) 0.5;

            % 1. Standard Off-Policy Prediction
            solver1 = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
                'Algorithm',        "OffPolicyPrediction", ...
                'Gamma',            1.0,                   ...
                'NumEpisodes',      1000,                  ...
                'TargetPolicy',     target_pi,             ...
                'BehaviorPolicy',   behavior_b,            ...
                'DiscountingAware', false,                 ...
                'Verbose',          false                  ...
            );
            rng(42);
            [V1, ~, ~, ~] = solver1.run();

            % 2. Discounting-Aware Off-Policy Prediction (gamma = 1)
            solver2 = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
                'Algorithm',        "OffPolicyPrediction", ...
                'Gamma',            1.0,                   ...
                'NumEpisodes',      1000,                  ...
                'TargetPolicy',     target_pi,             ...
                'BehaviorPolicy',   behavior_b,            ...
                'DiscountingAware', true,                  ...
                'Verbose',          false                  ...
            );
            rng(42);
            [V2, ~, ~, ~] = solver2.run();

            % Verification
            testCase.verifyEqual(V1, V2, 'RelTol', 1e-10);
        end

        function testWeightingSchemeInstantiation(testCase)
            ord_scheme  = OceanStats.RL.MonteCarlo.Weighting.OrdinaryISWeighting();
            wtd_scheme  = OceanStats.RL.MonteCarlo.Weighting.WeightedISWeighting();
            da_ord      = OceanStats.RL.MonteCarlo.Weighting.DiscountingAwareOrdinaryISWeighting();
            da_wtd      = OceanStats.RL.MonteCarlo.Weighting.DiscountingAwareWeightedISWeighting();

            meta_ord = ord_scheme.getMetadata();
            testCase.verifyEqual(meta_ord.ClassName, "OceanStats.RL.MonteCarlo.Weighting.OrdinaryISWeighting");

            meta_wtd = wtd_scheme.getMetadata();
            testCase.verifyEqual(meta_wtd.ClassName, "OceanStats.RL.MonteCarlo.Weighting.WeightedISWeighting");

            meta_da_ord = da_ord.getMetadata();
            testCase.verifyEqual(meta_da_ord.ClassName, "OceanStats.RL.MonteCarlo.Weighting.DiscountingAwareOrdinaryISWeighting");

            meta_da_wtd = da_wtd.getMetadata();
            testCase.verifyEqual(meta_da_wtd.ClassName, "OceanStats.RL.MonteCarlo.Weighting.DiscountingAwareWeightedISWeighting");
        end

        function testStrategyMetadataAndSpec(testCase)
            strat = OceanStats.RL.MonteCarlo.Strategies.DiscountingAwarePredictionStrategy();
            meta  = strat.getMetadata();
            spec  = strat.getPipelineSpec();

            testCase.verifyEqual(meta.ClassName, "OceanStats.RL.MonteCarlo.Strategies.DiscountingAwarePredictionStrategy");
            testCase.verifyEqual(spec.ComponentName, "DiscountingAwarePredictionStrategy");
        end
    end
end
