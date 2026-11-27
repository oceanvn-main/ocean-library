classdef test_normal_form_game < OceanTest.BaseTestCase
    % TEST_NORMAL_FORM_GAME Unit tests for OceanStats.GameTheory.Core.NormalFormGame.
    %
    % AUTHOR: Ocean Library System
    % DATE: 2026-09-23

    methods (Test)
        function testBimatrixConstruction(testCase)
            % Prisoner's Dilemma: rows/cols = {Cooperate, Defect}
            A = [3 0; 5 1];   % Player 1 payoffs
            B = [3 5; 0 1];   % Player 2 payoffs
            g = OceanStats.GameTheory.Core.NormalFormGame('PayoffA', A, 'PayoffB', B, ...
                'StrategyNames', {{'Cooperate', 'Defect'}, {'Cooperate', 'Defect'}});

            testCase.verifyEqual(g.numPlayers(), 2);
            testCase.verifyEqual(g.numStrategies(1), 2);
            testCase.verifyEqual(g.numStrategies(2), 2);
            testCase.verifyEqual(g.strategyNames(1), {'Cooperate', 'Defect'});

            % Profile (Defect, Cooperate): P1 gets 5, P2 gets 0
            testCase.verifyEqual(g.payoff(1, [2, 1]), 5);
            testCase.verifyEqual(g.payoff(2, [2, 1]), 0);

            % Profile (Defect, Defect): both get 1
            testCase.verifyEqual(g.payoff(1, [2, 2]), 1);
            testCase.verifyEqual(g.payoff(2, [2, 2]), 1);
        end

        function testDefaultStrategyNames(testCase)
            A = [1 2; 3 4];
            B = [4 3; 2 1];
            g = OceanStats.GameTheory.Core.NormalFormGame('PayoffA', A, 'PayoffB', B);
            testCase.verifyEqual(g.strategyNames(1), {'S1', 'S2'});
        end

        function testShapeMismatchErrors(testCase)
            A = [1 2; 3 4];
            B = [1 2 3];
            testCase.verifyError(@() OceanStats.GameTheory.Core.NormalFormGame('PayoffA', A, 'PayoffB', B), ...
                'NormalFormGame:ShapeMismatch');
        end

        function testThreePlayerTensorConstruction(testCase)
            U1 = zeros(2, 2, 2);
            U2 = zeros(2, 2, 2);
            U3 = zeros(2, 2, 2);
            for a = 1:2
                for b = 1:2
                    for c = 1:2
                        U1(a, b, c) = (b == 2) + (c == 2);
                        U2(a, b, c) = (a == 2) + (c == 2);
                        U3(a, b, c) = (a == 2) + (b == 2);
                    end
                end
            end
            g = OceanStats.GameTheory.Core.NormalFormGame('PayoffTensors', {U1, U2, U3});

            testCase.verifyEqual(g.numPlayers(), 3);
            testCase.verifyEqual(g.payoff(1, [1 2 2]), 2);  % both others play strategy 2
            testCase.verifyEqual(g.payoff(2, [2 1 2]), 2);  % excludes own (dim 2) strategy
        end

        function testMetadataHasLatexAndSymbols(testCase)
            A = [1 2; 3 4];
            B = [4 3; 2 1];
            g = OceanStats.GameTheory.Core.NormalFormGame('PayoffA', A, 'PayoffB', B);
            meta = g.getMetadata();
            testCase.verifyTrue(strlength(meta.SymbolicLatex) > 0);
            testCase.verifyGreaterThan(numel(meta.Symbols), 0);
        end
    end
end
