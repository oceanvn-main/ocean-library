classdef test_round_robin_tournament < OceanTest.BaseTestCase
    % TEST_ROUND_ROBIN_TOURNAMENT Unit tests for
    % OceanStats.GameTheory.Core.RoundRobinTournament.
    %
    % AUTHOR: Ocean Library System
    % DATE: 2026-09-23

    methods (Test)
        function testSixPlayerOneShotPrisonersDilemma(testCase)
            % Hand-verified: 3 Cooperators + 3 Defectors, one-shot round robin.
            % Each Cooperator: 2 C-C matches (3 each) + 3 C-D matches (0 each) = 6.
            % Each Defector:   2 D-D matches (1 each) + 3 D-C matches (5 each) = 17.
            A = [3 0; 5 1];
            B = [3 5; 0 1];
            base = OceanStats.GameTheory.Core.NormalFormGame('PayoffA', A, 'PayoffB', B, ...
                'StrategyNames', {{'Cooperate', 'Defect'}, {'Cooperate', 'Defect'}});

            playerStrategies = [1 1 1 2 2 2];  % players 1-3 Cooperate, 4-6 Defect
            tournament = OceanStats.GameTheory.Core.RoundRobinTournament(base, playerStrategies);
            [scores, groups] = tournament.run();

            testCase.assertMatrixEquals(scores, [6 6 6 17 17 17], 1e-9, 'Round-robin PD scores');
            testCase.verifyEqual(size(groups, 1), nchoosek(6, 2));
            testCase.verifyGreaterThan(mean(scores(4:6)), mean(scores(1:3)));
        end

        function testScalesToThirtyPlayers(testCase)
            A = [3 0; 5 1];
            B = [3 5; 0 1];
            base = OceanStats.GameTheory.Core.NormalFormGame('PayoffA', A, 'PayoffB', B);

            P = 30;
            playerStrategies = ones(1, P);
            playerStrategies(1:2:end) = 2;  % alternate Cooperate/Defect

            tournament = OceanStats.GameTheory.Core.RoundRobinTournament(base, playerStrategies);
            [scores, groups] = tournament.run();

            testCase.verifyEqual(numel(scores), P);
            testCase.verifyEqual(size(groups, 1), nchoosek(P, 2));
            testCase.verifyEqual(size(groups, 2), 2);
            testCase.verifyTrue(all(isfinite(scores)));
        end

        function testSessionSizeThreeHandVerified(testCase)
            % Symmetric 3-player game: payoff = count of the OTHER two
            % seat-mates playing strategy 2. Hand-verified for P=6 with
            % players [1 2 1 2 1 2] (3 strategy-1, 3 strategy-2):
            %   strategy-1 players score 12 each, strategy-2 players score 8 each.
            U1 = zeros(2, 2, 2);
            U2 = zeros(2, 2, 2);
            U3 = zeros(2, 2, 2);
            for a = 1:2
                for b = 1:2
                    for c = 1:2
                        U1(a, b, c) = (b == 2) + (c == 2);  % excludes own (seat 1) strategy
                        U2(a, b, c) = (a == 2) + (c == 2);  % excludes own (seat 2) strategy
                        U3(a, b, c) = (a == 2) + (b == 2);  % excludes own (seat 3) strategy
                    end
                end
            end
            base = OceanStats.GameTheory.Core.NormalFormGame('PayoffTensors', {U1, U2, U3});

            playerStrategies = [1 2 1 2 1 2];
            tournament = OceanStats.GameTheory.Core.RoundRobinTournament(base, playerStrategies);
            [scores, groups] = tournament.run();

            testCase.verifyEqual(size(groups, 2), 3);
            testCase.verifyEqual(size(groups, 1), nchoosek(6, 3));
            testCase.assertMatrixEquals(scores, [12 8 12 8 12 8], 1e-9, 'k=3 session scores');
        end

        function testSessionSizeThreeScalesToThirtyPlayers(testCase)
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
            base = OceanStats.GameTheory.Core.NormalFormGame('PayoffTensors', {U1, U2, U3});

            P = 30;
            playerStrategies = ones(1, P);
            playerStrategies(1:2:end) = 2;

            tournament = OceanStats.GameTheory.Core.RoundRobinTournament(base, playerStrategies);
            [scores, groups] = tournament.run();

            testCase.verifyEqual(numel(scores), P);
            testCase.verifyEqual(size(groups, 1), nchoosek(P, 3));
            testCase.verifyEqual(size(groups, 2), 3);
        end
    end
end
