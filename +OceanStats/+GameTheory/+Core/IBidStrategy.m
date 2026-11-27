classdef (Abstract) IBidStrategy < handle
    % IBIDSTRATEGY  Minimal contract any bidder must satisfy to enter an
    %   auction environment (OceanStats.GameTheory.Core.RepeatedAuctionSimulator).
    %
    %   The environment only ever calls proposeBid -- it never inspects a
    %   strategy's internals. New bidder strategies (learning or not) are
    %   added by implementing this interface; nothing in the environment or
    %   the auction rule changes.
    %
    %   Author: Ocean Library System
    %   Date: 2026-09-23

    methods (Abstract)
        % PROPOSEBID  Choose a bid.
        %   value   — the bidder's own private value for the item
        %   bidGrid — the fixed candidate bid set (row vector)
        bid = proposeBid(obj, value, bidGrid)
    end
end
