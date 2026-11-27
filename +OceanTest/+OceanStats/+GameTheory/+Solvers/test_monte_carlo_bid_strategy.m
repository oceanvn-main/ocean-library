classdef test_monte_carlo_bid_strategy < OceanTest.BaseTestCase
    % TEST_MONTE_CARLO_BID_STRATEGY Unit tests for
    % OceanStats.GameTheory.Solvers.MonteCarloBidStrategy.
    %
    % AUTHOR: Ocean Library System
    % DATE: 2026-09-23

    methods (Test)
        function testConvergesToBestArm(testCase)
            rng(123);
            bidGrid = 1:5;
            bestBid = 3;
            trueMeans = -(bidGrid - bestBid) .^ 2;   % [-4 -1 0 -1 -4], max at bid=3

            strategy = OceanStats.GameTheory.Solvers.MonteCarloBidStrategy(bidGrid, 'Epsilon', 0.1);

            numRounds = 3000;
            chosen = zeros(1, numRounds);
            for r = 1:numRounds
                bid = strategy.proposeBid(0, bidGrid);
                idx = find(bidGrid == bid, 1);
                reward = trueMeans(idx) + 0.5 * randn();
                strategy = strategy.observeOutcome(bid, reward);
                chosen(r) = bid;
            end

            [~, bestIdx] = max(strategy.QValues);
            testCase.verifyEqual(bidGrid(bestIdx), bestBid);

            lastChosen = chosen(end - 199:end);
            fracBest = mean(lastChosen == bestBid);
            testCase.verifyGreaterThan(fracBest, 0.7);
        end

        function testIgnoresPassedValueAndBidGrid(testCase)
            strategy = OceanStats.GameTheory.Solvers.MonteCarloBidStrategy(1:3, 'Epsilon', 0);
            bid = strategy.proposeBid(999, [100 200 300]);
            testCase.verifyTrue(ismember(bid, 1:3));
        end
    end
end
