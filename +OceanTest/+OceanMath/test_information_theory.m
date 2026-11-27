classdef test_information_theory < OceanTest.BaseTestCase
    % TEST_INFORMATION_THEORY Tests mathematical gambling, Kelly criterion, and Shannon-Fano source coder.
    
    methods (Test)
        function testEntropy(testCase)
            p = [0.5, 0.25, 0.25];
            H = OceanMath.Information.Processes.KellyOptimizer.entropy(p);
            testCase.verifyEqual(H, 1.5, 'AbsTol', 1e-9);
            
            % Edge case: zero probability
            p_zero = [1.0, 0.0, 0.0];
            H_zero = OceanMath.Information.Processes.KellyOptimizer.entropy(p_zero);
            testCase.verifyEqual(H_zero, 0.0, 'AbsTol', 1e-9);
        end
        
        function testDivergence(testCase)
            p = [0.5, 0.5];
            b = [0.5, 0.5];
            D_equal = OceanMath.Information.Processes.KellyOptimizer.divergence(p, b);
            testCase.verifyEqual(D_equal, 0.0, 'AbsTol', 1e-9);
            
            b_diff = [0.25, 0.75];
            D = OceanMath.Information.Processes.KellyOptimizer.divergence(p, b_diff);
            expected_D = 0.5 * log2(0.5 / 0.25) + 0.5 * log2(0.5 / 0.75);
            testCase.verifyEqual(D, expected_D, 'AbsTol', 1e-9);
            
            % Edge case: infinity
            b_zero = [1.0, 0.0];
            p_pos = [0.5, 0.5];
            D_inf = OceanMath.Information.Processes.KellyOptimizer.divergence(p_pos, b_zero);
            testCase.verifyEqual(D_inf, Inf);
        end
        
        function testKellyOptimization(testCase)
            % Under fair odds, optimal betting equals true probabilities
            p = [0.6, 0.4];
            odds = [1.0 / 0.6, 1.0 / 0.4];
            b_opt = OceanMath.Information.Processes.KellyOptimizer.computeOptimalFractions(p, odds);
            testCase.assertMatrixEquals(b_opt, p, 1e-9);
            
            % Unfair odds with bookmaker cut
            p_unfair = [0.5, 0.5];
            odds_unfair = [1.5, 1.5]; % Implied sum(r) = 2/1.5 = 1.33 > 1
            b_opt_unfair = OceanMath.Information.Processes.KellyOptimizer.computeOptimalFractions(p_unfair, odds_unfair);
            % Expected log-wealth rate: p1 * log2(b1*1.5 + 1 - b1 - b2) + p2 * log2(b2*1.5 + 1 - b1 - b2)
            % Let's make sure it computes non-negative bets
            testCase.verifyTrue(all(b_opt_unfair >= 0));
            testCase.verifyTrue(sum(b_opt_unfair) <= 1.0);
        end
        
        function testShannonFanoCoder(testCase)
            coder = OceanMath.Information.Coding.ShannonFanoCoder();
            symbols = {'a', 'b', 'c', 'd'};
            probs = [0.4, 0.3, 0.2, 0.1];
            
            table = coder.encode(symbols, probs);
            testCase.verifyEqual(table.getSize(), 4);
            testCase.verifyEqual(table.kraftSum(), 1.0, 'AbsTol', 1e-9);
            
            % Check prefix-free property
            codes = table.getCodewords();
            for i = 1:numel(codes)
                for j = 1:numel(codes)
                    if i ~= j
                        testCase.verifyFalse(startsWith(codes{j}, codes{i}), ...
                            sprintf('Code is not prefix-free: %s is prefix of %s', codes{i}, codes{j}));
                    end
                end
            end
            
            % Check bounds
            L = table.expectedLength();
            H = table.entropyBits();
            testCase.verifyTrue(L >= H - 1e-9);
            testCase.verifyTrue(L < H + 1.0 + 1e-9);
        end
    end
end
