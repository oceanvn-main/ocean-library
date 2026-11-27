classdef test_single_item_auction_game < OceanTest.BaseTestCase
    % TEST_SINGLE_ITEM_AUCTION_GAME Unit tests for
    % OceanStats.GameTheory.Core.SingleItemAuctionGame.
    %
    % AUTHOR: Ocean Library System
    % DATE: 2026-09-23

    methods (Test)
        function testBidGridFixedAcrossGenerateCalls(testCase)
            gen = OceanStats.GameTheory.Core.SingleItemAuctionGame( ...
                'NumBidders', 3, 'BidGrid', 0:10);
            [~, values1] = gen.generate();
            [~, values2] = gen.generate();
            testCase.verifyEqual(gen.BidGrid, 0:10);
            testCase.verifyFalse(isequal(values1, values2));
        end

        function testReducedPayoffFormula(testCase)
            bidGrid = 0:10;
            gen = OceanStats.GameTheory.Core.SingleItemAuctionGame( ...
                'NumBidders', 1, 'BidGrid', bidGrid);
            [games, values] = gen.generate([8 8]);   % force true value = 8
            g1 = games{1};
            testCase.verifyEqual(values(1), 8);

            bIdx = find(bidGrid == 7);
            mIdx = find(bidGrid == 3);
            testCase.verifyEqual(g1.payoff(1, [bIdx, mIdx]), 8 - 3);   % 7>=3, wins, pays 3

            bIdx2 = find(bidGrid == 2);
            testCase.verifyEqual(g1.payoff(1, [bIdx2, mIdx]), 0);      % 2<3, loses
        end

        function testTruthfulBiddingIsNeverDominated(testCase)
            % The robust, always-true half of Vickrey's theorem: truthful
            % bidding is never itself weakly (or strictly) dominated by
            % anything, for any private value, any number of bidders.
            bidGrid = 0:20;
            gen = OceanStats.GameTheory.Core.SingleItemAuctionGame( ...
                'NumBidders', 5, 'BidGrid', bidGrid);
            [games, values] = gen.generate();
            solver = OceanStats.GameTheory.Solvers.IteratedDominanceSolver();

            for i = 1:numel(games)
                truthfulIdx = find(bidGrid == values(i));
                testCase.verifyFalse(solver.isStrictlyDominated(games{i}, 1, truthfulIdx));
                testCase.verifyFalse(solver.isWeaklyDominated(games{i}, 1, truthfulIdx));
            end
        end

        function testTruthfulBiddingDominatesFarAlternativesButTiesTheAdjacentUnderbid(testCase)
            % Hand-derived: for true value v, bidding v weakly dominates any
            % bid at least 2 grid-steps away (there's always an opponent
            % value m strictly between them where the gap shows up), but a
            % bid exactly ONE step below v (v-1) is payoff-EQUIVALENT to v
            % against every possible opponent-max — a tie, not a dominance
            % relationship. Both facts are checked here rather than
            % overstating "truthful dominates every alternative".
            bidGrid = 0:20;
            gen = OceanStats.GameTheory.Core.SingleItemAuctionGame( ...
                'NumBidders', 1, 'BidGrid', bidGrid);
            [games, ~] = gen.generate([10 10]);
            g1 = games{1};
            solver = OceanStats.GameTheory.Solvers.IteratedDominanceSolver();
            truthfulIdx = find(bidGrid == 10);

            for offset = [-6 -4 -2 2 4 6]
                idx = truthfulIdx + offset;
                if idx < 1 || idx > numel(bidGrid)
                    continue;
                end
                testCase.verifyTrue(solver.isWeaklyDominated(g1, 1, idx), ...
                    sprintf('bid at offset %d from true value should be weakly dominated', offset));
            end

            adjacentIdx = truthfulIdx - 1;   % bid = 9, one step below value = 10
            testCase.verifyFalse(solver.isWeaklyDominated(g1, 1, adjacentIdx));
            for m = 1:numel(bidGrid)
                testCase.verifyEqual(g1.payoff(1, [truthfulIdx, m]), g1.payoff(1, [adjacentIdx, m]));
            end
        end
    end
end
