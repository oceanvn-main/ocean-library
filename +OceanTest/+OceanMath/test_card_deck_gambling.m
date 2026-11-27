classdef test_card_deck_gambling < OceanTest.BaseTestCase
    % TEST_CARD_DECK_GAMBLING Unit tests for CardDeckSimulator and CardDeckBettingOptimizer.
    %
    % These tests verify that the sequential card deck gambling model behaves
    % in accordance with information-theoretic principles and the Kelly Criterion.
    %
    % In particular, they assert the elegant property that the final wealth of
    % a log-optimal gambler is strictly constant across all shuffles.
    %
    % See also: OceanMath.Information.Processes.CardDeckSimulator,
    %           OceanMath.Information.Processes.CardDeckBettingOptimizer

    methods (Test)
        function test_deck_conservation(testCase)
            % Verify that a full sequence of card draws consumes exactly R red and B black cards,
            % and the sum of conditional probabilities is always 1.0.
            R = 15;
            B = 25;
            sim = OceanMath.Information.Processes.CardDeckSimulator(R, B);

            testCase.verifyEqual(sim.RemainingRed, R);
            testCase.verifyEqual(sim.RemainingBlack, B);

            redCount = 0;
            blackCount = 0;
            totalCards = R + B;

            for i = 1:totalCards
                % Draw card and record probabilities prior to the draw
                [card, pRed, pBlack] = sim.drawNext();

                % Conditional probabilities must sum to 1.0 prior to every draw
                testCase.verifyEqual(pRed + pBlack, 1.0, 'AbsTol', 1e-12, ...
                    sprintf('Sum of conditional probabilities is not 1.0 at draw %d.', i));

                if card == 1
                    redCount = redCount + 1;
                elseif card == 2
                    blackCount = blackCount + 1;
                else
                    testCase.fail('Unexpected card color drawn.');
                end
            end

            % Verify conservation of card counts
            testCase.verifyEqual(redCount, R, 'Total red cards drawn does not match R.');
            testCase.verifyEqual(blackCount, B, 'Total black cards drawn does not match B.');

            % Verify simulator state is empty
            testCase.verifyEqual(sim.RemainingRed, 0);
            testCase.verifyEqual(sim.RemainingBlack, 0);
            testCase.verifyEqual(sim.CurrentIndex, totalCards);

            % Verify drawing from an empty deck throws a clean error
            testCase.verifyError(@() sim.drawNext(), ...
                'OceanMath:Information:CardDeckSimulator:NoCardsRemaining');
        end

        function test_theoretical_wealth(testCase)
            % Assert sequential proportional betting on any shuffled sequence of R red
            % and B black cards under arbitrary payoff odds results in the exact same
            % final wealth: S_final = S_0 * (oRed^R * oBlack^B) / nchoosek(R+B, R).
            R = 10;
            B = 12;
            oRed = 2.5;
            oBlack = 1.8;
            S0 = 100.0;

            % Compute theoretical final wealth using invariant formula
            expectedS_N = OceanMath.Information.Processes.CardDeckBettingOptimizer.computeTheoreticalFinalWealth( ...
                R, B, oRed, oBlack, S0);

            sim = OceanMath.Information.Processes.CardDeckSimulator(R, B);

            % Test over 10 different shuffles/paths
            for run = 1:10
                sim.shuffle();
                wealth = S0;

                while sim.RemainingRed + sim.RemainingBlack > 0
                    % Keep track of state prior to draw
                    r_prior = sim.RemainingRed;
                    b_prior = sim.RemainingBlack;

                    [card, pRed, pBlack] = sim.drawNext();

                    % Proportional betting fractions
                    betFractions = [pRed, pBlack];

                    % Calculate payoff factor
                    payoffFactor = OceanMath.Information.Processes.CardDeckBettingOptimizer.getStepPayoff( ...
                        card, r_prior, b_prior, oRed, oBlack, betFractions);

                    wealth = wealth * payoffFactor;
                end

                % Assert wealth converges to the exact theoretical wealth
                testCase.verifyEqual(wealth, expectedS_N, 'AbsTol', 1e-9, ...
                    sprintf('Final wealth on shuffle %d did not match invariant wealth.', run));
            end
        end

        function test_suboptimal_betting(testCase)
            % Verify that any constant fraction or subjective mismatch strategy yields
            % an expected log-wealth strictly less than the log-optimal proportional Kelly return.
            R = 6;
            B = 6;
            oRed = 2.0;
            oBlack = 2.0;
            S0 = 10.0;

            % Proportional (Kelly) final wealth
            optimalFinalWealth = OceanMath.Information.Processes.CardDeckBettingOptimizer.computeTheoreticalFinalWealth( ...
                R, B, oRed, oBlack, S0);

            % Define a constant fraction strategy: bet fixed 30% Red and 30% Black
            constBetFractions = [0.3, 0.3];

            % Define a subjective belief mismatch strategy: qRed = alpha * pRed + (1-alpha)*0.5
            mismatchAlpha = 0.6;

            sim = OceanMath.Information.Processes.CardDeckSimulator(R, B);
            numShuffles = 100;

            finalWealthsConst = zeros(1, numShuffles);
            finalWealthsMismatch = zeros(1, numShuffles);

            for run = 1:numShuffles
                % 1. Constant fraction run
                sim.shuffle();
                wealthConst = S0;
                while sim.RemainingRed + sim.RemainingBlack > 0
                    r_prior = sim.RemainingRed;
                    b_prior = sim.RemainingBlack;
                    [card, ~, ~] = sim.drawNext();
                    payoffFactor = OceanMath.Information.Processes.CardDeckBettingOptimizer.getStepPayoff( ...
                        card, r_prior, b_prior, oRed, oBlack, constBetFractions);
                    wealthConst = wealthConst * payoffFactor;
                end
                finalWealthsConst(run) = wealthConst;

                % 2. Mismatched belief run
                sim.shuffle();
                wealthMismatch = S0;
                while sim.RemainingRed + sim.RemainingBlack > 0
                    r_prior = sim.RemainingRed;
                    b_prior = sim.RemainingBlack;
                    [card, pRed, pBlack] = sim.drawNext();

                    % Distort beliefs
                    qRed = mismatchAlpha * pRed + (1 - mismatchAlpha) * 0.5;
                    qBlack = mismatchAlpha * pBlack + (1 - mismatchAlpha) * 0.5;
                    
                    % Normalize if sum exceeds 1.0
                    sumQ = qRed + qBlack;
                    if sumQ > 1.0
                        qRed = qRed / sumQ;
                        qBlack = qBlack / sumQ;
                    end

                    payoffFactor = OceanMath.Information.Processes.CardDeckBettingOptimizer.getStepPayoff( ...
                        card, r_prior, b_prior, oRed, oBlack, [qRed, qBlack]);
                    wealthMismatch = wealthMismatch * payoffFactor;
                end
                finalWealthsMismatch(run) = wealthMismatch;
            end

            % Compute average log-wealth for all strategies
            meanLogConst = mean(log(finalWealthsConst));
            meanLogMismatch = mean(log(finalWealthsMismatch));
            optLog = log(optimalFinalWealth);

            % Suboptimal strategies must yield strictly less log-wealth than Kelly proportional
            testCase.verifyLessThan(meanLogConst, optLog, ...
                'Constant fraction strategy log-wealth must be strictly less than Kelly-optimal.');
            testCase.verifyLessThan(meanLogMismatch, optLog, ...
                'Mismatched belief strategy log-wealth must be strictly less than Kelly-optimal.');
        end

        function test_edge_cases(testCase)
            % Verify zero card cases (R=0 or B=0) and large combination cases safely.
            
            % Case 1: Zero red cards (R = 0, B = 5)
            simR0 = OceanMath.Information.Processes.CardDeckSimulator(0, 5);
            testCase.verifyEqual(simR0.RemainingRed, 0);
            testCase.verifyEqual(simR0.RemainingBlack, 5);

            for i = 1:5
                [card, pRed, pBlack] = simR0.drawNext();
                testCase.verifyEqual(card, 2, 'Must draw Black card when R=0.');
                testCase.verifyEqual(pRed, 0.0);
                testCase.verifyEqual(pBlack, 1.0);
            end

            % Case 2: Zero black cards (R = 5, B = 0)
            simB0 = OceanMath.Information.Processes.CardDeckSimulator(5, 0);
            testCase.verifyEqual(simB0.RemainingRed, 5);
            testCase.verifyEqual(simB0.RemainingBlack, 0);

            for i = 1:5
                [card, pRed, pBlack] = simB0.drawNext();
                testCase.verifyEqual(card, 1, 'Must draw Red card when B=0.');
                testCase.verifyEqual(pRed, 1.0);
                testCase.verifyEqual(pBlack, 0.0);
            end

            % Case 3: Empty deck (R = 0, B = 0)
            simEmpty = OceanMath.Information.Processes.CardDeckSimulator(0, 0);
            testCase.verifyEqual(simEmpty.NumRedR, 0);
            testCase.verifyEqual(simEmpty.NumBlackB, 0);
            testCase.verifyEqual(length(simEmpty.DeckSequence), 0);
            testCase.verifyError(@() simEmpty.drawNext(), ...
                'OceanMath:Information:CardDeckSimulator:NoCardsRemaining');

            % Case 4: Large combinations that would overflow standard nchoosek
            % nchoosek(200, 100) handles safely via gammaln calculations.
            R_large = 100;
            B_large = 100;
            oRed = 2.0;
            oBlack = 2.0;
            S0 = 1.0;

            finalWealth = OceanMath.Information.Processes.CardDeckBettingOptimizer.computeTheoreticalFinalWealth( ...
                R_large, B_large, oRed, oBlack, S0);

            testCase.assertTrue(isfinite(finalWealth), 'Final wealth must be a finite double.');
            testCase.assertGreaterThan(finalWealth, 0, 'Final wealth must be strictly positive.');
        end
    end
end
