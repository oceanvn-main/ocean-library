classdef (Abstract) IAuctionRule < handle
    % IAUCTIONRULE  Abstract contract for a single-item auction's allocation
    %   + payment rule, captured as ONE pure function.
    %
    %   Every standard single-item auction (second-price, first-price,
    %   all-pay, ...) is an "aggregative game": a bidder's payoff depends
    %   only on their OWN bid, the MAXIMUM of every other bidder's bid, and
    %   their OWN value -- never on other bidders' bids individually. That
    %   makes the entire swappable "rule" a single pure function, reused
    %   unmodified by both the analytic dominance check
    %   (OceanStats.GameTheory.Core.SingleItemAuctionGame) and the full
    %   N-bidder repeated simulation
    %   (OceanStats.GameTheory.Core.RepeatedAuctionSimulator). Adding a new
    %   auction format later means implementing this one method -- nothing
    %   else changes.
    %
    %   Author: Ocean Library System
    %   Date: 2026-09-23

    methods (Abstract)
        % RESOLVEAGAINSTMAX  A bidder's payoff given their own bid, the max
        %   of every other bidder's bid, and their own private value.
        payoff = resolveAgainstMax(obj, ownBid, maxOtherBid, ownValue)
    end
end
