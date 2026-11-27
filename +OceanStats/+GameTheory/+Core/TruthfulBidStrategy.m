classdef TruthfulBidStrategy < OceanStats.GameTheory.Core.IBidStrategy
    % TRUTHFULBIDSTRATEGY  Deterministic strategy: always bids own value,
    %   snapped to the nearest point on the fixed BidGrid.
    %
    %   The ground-truth strategy the Vickrey theorem says is weakly
    %   dominant in a second-price auction -- useful as a baseline/oracle
    %   strategy, and as the simplest possible
    %   OceanStats.GameTheory.Core.IBidStrategy example (no learning
    %   capability needed).
    %
    %   Author: Ocean Library System
    %   Date: 2026-09-23

    methods
        function bid = proposeBid(~, value, bidGrid)
            arguments
                ~
                value (1,1) double
                bidGrid (1,:) double
            end
            [~, idx] = min(abs(bidGrid - value));
            bid = bidGrid(idx);
        end

        function meta = getMetadata(~)
            meta = struct();
            meta.SymbolicLatex = "b_i = \arg\min_{b \in \mathrm{BidGrid}} |b - v_i|";
            meta.Symbols = [ ...
                struct('Symbol', 'b_i', 'Description', 'The proposed bid'), ...
                struct('Symbol', 'v_i', 'Description', 'Bidder i''s private value') ...
                ];
            meta.Assumptions = {'Deterministic, non-learning: no state carried between rounds.'};
            meta.IsSymbolic = true;
            meta.Reference = "Vickrey (1961) -- the strategy the theorem says is weakly dominant";
        end
    end
end
