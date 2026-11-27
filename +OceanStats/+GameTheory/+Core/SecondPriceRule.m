classdef SecondPriceRule < OceanStats.GameTheory.Core.IAuctionRule
    % SECONDPRICERULE  Second-price (Vickrey) sealed-bid auction rule.
    %
    %   The highest bidder wins (ties broken in the bidder's own favor:
    %   ownBid >= maxOtherBid) and pays the second-highest bid -- i.e. the
    %   highest bid AMONG THE OTHERS. Winner's payoff = value - price;
    %   every other bidder's payoff = 0.
    %
    %   Author: Ocean Library System
    %   Date: 2026-09-23
    %   Reference: Vickrey (1961)

    methods
        function payoff = resolveAgainstMax(~, ownBid, maxOtherBid, ownValue)
            arguments
                ~
                ownBid (1,1) double
                maxOtherBid (1,1) double
                ownValue (1,1) double
            end
            if ownBid >= maxOtherBid
                payoff = ownValue - maxOtherBid;
            else
                payoff = 0;
            end
        end

        function meta = getMetadata(~)
            meta = struct();
            meta.SymbolicLatex = "u_i(b_i, m) = v_i - m \text{ if } b_i \geq m \text{, else } 0";
            meta.Symbols = [ ...
                struct('Symbol', 'b_i', 'Description', 'Bidder i''s own bid'), ...
                struct('Symbol', 'm',   'Description', 'Max of every other bidder''s bid (the price paid if i wins)'), ...
                struct('Symbol', 'v_i', 'Description', 'Bidder i''s private value') ...
                ];
            meta.Assumptions = {'Ties (ownBid == maxOtherBid) resolved in the bidder''s own favor.'};
            meta.IsSymbolic = true;
            meta.Reference = "Vickrey (1961)";
        end
    end
end
