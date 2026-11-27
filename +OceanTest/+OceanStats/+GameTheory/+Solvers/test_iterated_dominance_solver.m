classdef test_iterated_dominance_solver < OceanTest.BaseTestCase
    % TEST_ITERATED_DOMINANCE_SOLVER Unit tests for
    % OceanStats.GameTheory.Solvers.IteratedDominanceSolver.
    %
    % AUTHOR: Ocean Library System
    % DATE: 2026-09-23

    methods (Test)
        function testPrisonersDilemmaSolvesToDefectDefect(testCase)
            A = [3 0; 5 1];   % rows/cols: Cooperate, Defect
            B = [3 5; 0 1];
            g = OceanStats.GameTheory.Core.NormalFormGame('PayoffA', A, 'PayoffB', B, ...
                'StrategyNames', {{'Cooperate', 'Defect'}, {'Cooperate', 'Defect'}});

            solver = OceanStats.GameTheory.Solvers.IteratedDominanceSolver();
            result = solver.solve(g);

            testCase.verifyEqual(result.SurvivingIndices{1}, 2);  % Defect
            testCase.verifyEqual(result.SurvivingIndices{2}, 2);  % Defect
            testCase.verifyEqual(result.ReducedGame.numStrategies(1), 1);
            testCase.verifyEqual(result.ReducedGame.numStrategies(2), 1);
            testCase.verifyEqual(result.ReducedGame.payoff(1, [1 1]), 1);
        end

        function testStrictDominanceCheckDirectly(testCase)
            A = [3 0; 5 1];
            B = [3 5; 0 1];
            g = OceanStats.GameTheory.Core.NormalFormGame('PayoffA', A, 'PayoffB', B);
            solver = OceanStats.GameTheory.Solvers.IteratedDominanceSolver();

            testCase.verifyTrue(solver.isStrictlyDominated(g, 1, 1));   % Cooperate dominated for P1
            testCase.verifyFalse(solver.isStrictlyDominated(g, 1, 2));  % Defect not dominated
        end

        function testMultiRoundCascadeElimination(testCase)
            % Hand-verified 3-round cascade:
            %   Round 1: A3 strictly dominated by A1 (and A2) for player 1.
            %   Round 2: with A3 gone, B2 becomes strictly dominated by B1.
            %   Round 3: with B2 gone, A2 becomes strictly dominated by A1.
            %   Final: (A1, B1) survives.
            U1 = [3 3; 2 4; 1 1];
            U2 = [5 2; 4 1; 0 6];
            g = OceanStats.GameTheory.Core.NormalFormGame('PayoffTensors', {U1, U2});

            solver = OceanStats.GameTheory.Solvers.IteratedDominanceSolver();
            result = solver.solve(g);
            history = solver.getHistory();

            testCase.verifyEqual(result.SurvivingIndices{1}, 1);  % A1
            testCase.verifyEqual(result.SurvivingIndices{2}, 1);  % B1
            testCase.verifyEqual(history.NumRounds, 3);
            testCase.verifyEqual(numel(history.Log), 3);
            testCase.verifyEqual(history.Log(1).Player, 1);
            testCase.verifyEqual(history.Log(1).EliminatedStrategy, 3);
            testCase.verifyEqual(history.Log(2).Player, 2);
            testCase.verifyEqual(history.Log(2).EliminatedStrategy, 2);
            testCase.verifyEqual(history.Log(3).Player, 1);
            testCase.verifyEqual(history.Log(3).EliminatedStrategy, 2);
        end

        function testWeakDominanceCheck(testCase)
            % P1 payoffs: A1=[2,2], A2=[2,1] -> A1 weakly (not strictly) dominates A2
            U1 = [2 2; 2 1];
            U2 = [1 1; 1 1];
            g = OceanStats.GameTheory.Core.NormalFormGame('PayoffTensors', {U1, U2});
            solver = OceanStats.GameTheory.Solvers.IteratedDominanceSolver();

            testCase.verifyTrue(solver.isWeaklyDominated(g, 1, 2));
            testCase.verifyFalse(solver.isStrictlyDominated(g, 1, 2));
        end

        function testMetadataHasLatexAndSymbols(testCase)
            solver = OceanStats.GameTheory.Solvers.IteratedDominanceSolver();
            meta = solver.getMetadata();
            testCase.verifyTrue(strlength(meta.SymbolicLatex) > 0);
            testCase.verifyGreaterThan(numel(meta.Symbols), 0);
        end
    end
end
