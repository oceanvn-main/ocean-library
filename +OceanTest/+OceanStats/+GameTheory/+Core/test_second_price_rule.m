classdef test_second_price_rule < OceanTest.BaseTestCase
    % TEST_SECOND_PRICE_RULE Unit tests for
    % OceanStats.GameTheory.Core.SecondPriceRule.
    %
    % AUTHOR: Ocean Library System
    % DATE: 2026-09-23

    methods (Test)
        function testWinnerPaysSecondPrice(testCase)
            rule = OceanStats.GameTheory.Core.SecondPriceRule();
            testCase.verifyEqual(rule.resolveAgainstMax(10, 6, 15), 9);
        end

        function testLoserGetsZero(testCase)
            rule = OceanStats.GameTheory.Core.SecondPriceRule();
            testCase.verifyEqual(rule.resolveAgainstMax(4, 6, 15), 0);
        end

        function testTieGoesToOwnBidder(testCase)
            rule = OceanStats.GameTheory.Core.SecondPriceRule();
            testCase.verifyEqual(rule.resolveAgainstMax(6, 6, 15), 9);
        end

        function testOverbidCanLosemoney(testCase)
            rule = OceanStats.GameTheory.Core.SecondPriceRule();
            testCase.verifyEqual(rule.resolveAgainstMax(12, 10, 8), -2);
        end
    end
end
