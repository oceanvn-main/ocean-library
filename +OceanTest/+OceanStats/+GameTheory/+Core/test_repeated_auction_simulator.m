classdef test_repeated_auction_simulator < OceanTest.BaseTestCase
    % TEST_REPEATED_AUCTION_SIMULATOR Unit tests for
    % OceanStats.GameTheory.Core.RepeatedAuctionSimulator.
    %
    % AUTHOR: Ocean Library System
    % DATE: 2026-09-23

    methods (Test)
        function testTwoTruthfulBiddersOneRound(testCase)
            rule = OceanStats.GameTheory.Core.SecondPriceRule();
            bidGrid = 0:20;
            values = [8, 5];
            strategies = { ...
                OceanStats.GameTheory.Core.TruthfulBidStrategy(), ...
                OceanStats.GameTheory.Core.TruthfulBidStrategy() ...
                };
            sim = OceanStats.GameTheory.Core.RepeatedAuctionSimulator(rule, bidGrid, values, strategies);
            history = sim.runRounds(1);

            testCase.verifyEqual(history.Bids(1, :), [8, 5]);
            testCase.verifyEqual(history.Payoffs(1, :), [3, 0]);
        end

        function testTruthfulBiddersAreStableAcrossRounds(testCase)
            rule = OceanStats.GameTheory.Core.SecondPriceRule();
            bidGrid = 0:20;
            values = [12, 7, 3];
            strategies = { ...
                OceanStats.GameTheory.Core.TruthfulBidStrategy(), ...
                OceanStats.GameTheory.Core.TruthfulBidStrategy(), ...
                OceanStats.GameTheory.Core.TruthfulBidStrategy() ...
                };
            sim = OceanStats.GameTheory.Core.RepeatedAuctionSimulator(rule, bidGrid, values, strategies);
            history = sim.runRounds(5);

            expectedBids = repmat([12, 7, 3], 5, 1);
            expectedPayoffs = repmat([12 - 7, 0, 0], 5, 1);
            testCase.assertMatrixEquals(history.Bids, expectedBids, 1e-9, 'Truthful bidders bid the same every round');
            testCase.assertMatrixEquals(history.Payoffs, expectedPayoffs, 1e-9, 'Truthful bidders payoffs stable every round');
        end

        function testTooFewBiddersErrors(testCase)
            rule = OceanStats.GameTheory.Core.SecondPriceRule();
            testCase.verifyError(@() OceanStats.GameTheory.Core.RepeatedAuctionSimulator( ...
                rule, 0:10, 5, {OceanStats.GameTheory.Core.TruthfulBidStrategy()}), ...
                'RepeatedAuctionSimulator:TooFewBidders');
        end
    end
end
