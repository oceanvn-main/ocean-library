classdef test_information < matlab.unittest.TestCase
    % TEST_INFORMATION Tests for KellyOptimizer, RaceSimulator, GamblingSimulator, and ShannonFanoCoder.
    %
    % This test suite exercises the core mathematical engine classes for the
    % Information Theory Sandbox, verifying their precision, boundary case
    % behavior, and correct implementation of underlying algorithms.
    %
    % See also: OceanMath.Information.Processes.KellyOptimizer,
    %           OceanMath.Information.Processes.RaceSimulator,
    %           OceanMath.Information.Processes.GamblingSimulator,
    %           OceanMath.Information.Coding.ShannonFanoCoder

    methods (Test)
        function test_entropy(testCase)
            % Verify Shannon entropy calculation and safe zero handling
            p1 = [0.5, 0.5];
            testCase.assertEqual(OceanMath.Information.Processes.KellyOptimizer.entropy(p1), 1.0, 'AbsTol', 1e-12);
            
            p2 = [1, 0, 0];
            testCase.assertEqual(OceanMath.Information.Processes.KellyOptimizer.entropy(p2), 0.0, 'AbsTol', 1e-12);
            
            % Test integration with DiscreteDistribution
            dist = OceanMath.Information.Distributions.DiscreteDistribution.fromPMF([1, 2, 3]', [0.5, 0.25, 0.25]');
            testCase.assertEqual(OceanMath.Information.Processes.KellyOptimizer.entropy(dist), 1.5, 'AbsTol', 1e-12);
        end

        function test_divergence(testCase)
            % Verify Kullback-Leibler divergence and edge cases
            p = [0.5, 0.5];
            testCase.assertEqual(OceanMath.Information.Processes.KellyOptimizer.divergence(p, p), 0.0, 'AbsTol', 1e-12);
            
            p = [0.5, 0.5];
            b = [0.25, 0.75];
            D = 0.5 * log2(0.5 / 0.25) + 0.5 * log2(0.5 / 0.75);
            testCase.assertEqual(OceanMath.Information.Processes.KellyOptimizer.divergence(p, b), D, 'AbsTol', 1e-12);
            
            % Test division by zero
            testCase.assertEqual(OceanMath.Information.Processes.KellyOptimizer.divergence([0.5, 0.5], [1.0, 0.0]), Inf);
            
            % Test zero probability in p (0 * log2(0) = 0)
            testCase.assertEqual(OceanMath.Information.Processes.KellyOptimizer.divergence([1.0, 0.0], [0.5, 0.5]), 1.0, 'AbsTol', 1e-12);
        end

        function test_kelly_optimal_fractions(testCase)
            % Verify optimal fractions computation under fair and unfair odds
            
            % Case 1: Fair odds
            p = [0.6, 0.4];
            odds = [1.0/0.6, 1.0/0.4];
            b_opt = OceanMath.Information.Processes.KellyOptimizer.computeOptimalFractions(p, odds);
            testCase.assertEqual(b_opt, p, 'AbsTol', 1e-6);
            
            % Case 2: Unfair odds with track-take
            p = [0.8, 0.2];
            odds = [1.1, 1.1];
            b_opt = OceanMath.Information.Processes.KellyOptimizer.computeOptimalFractions(p, odds);
            testCase.assertTrue(all(b_opt >= 0));
            testCase.assertTrue(sum(b_opt) <= 1.0);
        end

        function test_expected_growth(testCase)
            % Verify analytical growth calculation
            p = [0.6, 0.4];
            b = p;
            odds = [1/0.6, 1/0.4];
            W = OceanMath.Information.Processes.KellyOptimizer.computeExpectedGrowth(p, b, odds);
            testCase.assertEqual(W, 0.0, 'AbsTol', 1e-12);
        end

        function test_race_simulator(testCase)
            % Verify RaceSimulator samples correctly
            p = [0.7, 0.2, 0.1];
            sim = OceanMath.Information.Processes.RaceSimulator(p);
            
            testCase.assertEqual(sim.NumHorses, 3);
            
            winner = sim.drawWinner();
            testCase.assertTrue(winner >= 1 && winner <= 3);
            
            winners = sim.simulateRaces(100);
            testCase.assertEqual(length(winners), 100);
            testCase.assertTrue(all(winners >= 1 & winners <= 3));
        end

        function test_gambling_simulator(testCase)
            % Verify GamblingSimulator empirical and theoretical integration
            p = [0.6, 0.4];
            odds = [2.0, 2.0];
            b = [0.2, 0.1];
            
            sim = OceanMath.Information.Processes.GamblingSimulator(p, odds, b);
            res = sim.runSimulation(100, 50);
            
            testCase.assertEqual(length(res.WealthHistory), 51);
            testCase.assertEqual(length(res.WinningHistory), 50);
            testCase.assertTrue(isnumeric(res.EmpiricalGrowth));
            testCase.assertTrue(isnumeric(res.TheoreticalGrowth));
        end

        function test_shannon_fano_coder(testCase)
            % Verify Shannon-Fano encoding matches top-down partition rules
            coder = OceanMath.Information.Coding.ShannonFanoCoder();
            testCase.assertEqual(coder.getName(), 'Shannon-Fano (binary)');
            testCase.assertFalse(coder.isOptimal());
            
            symbols = {'a', 'b', 'c'};
            probs = [0.5, 0.3, 0.2];
            table = coder.encode(symbols, probs);
            
            testCase.assertEqual(table.getSize(), 3);
            codewords = table.getCodewords();
            testCase.assertEqual(codewords{1}, '0');
            testCase.assertEqual(codewords{2}, '10');
            testCase.assertEqual(codewords{3}, '11');
            
            eff = table.efficiency();
            testCase.assertTrue(eff >= 0 && eff <= 1.0);
        end
    end
end
