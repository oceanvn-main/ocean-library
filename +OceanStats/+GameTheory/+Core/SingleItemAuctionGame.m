classdef SingleItemAuctionGame < handle
    % SINGLEITEMAUCTIONGAME  Generator: random private values + an injected
    %   IAuctionRule -> one reduced 2-player NormalFormGame per bidder.
    %
    %   For bidder i, the reduced game is "bidder i" (bids from BidGrid) vs.
    %   an abstract "strongest other bidder" (also ranges over BidGrid,
    %   standing in for whatever value max(b_-i) could take). Checking
    %   dominance on this |BidGrid| x |BidGrid| game via
    %   OceanStats.GameTheory.Solvers.IteratedDominanceSolver is EXACTLY
    %   equivalent to checking it on the full N-bidder game -- the map
    %   b_-i -> max(b_-i) is surjective onto BidGrid, so quantifying over
    %   all opponent profiles is equivalent to quantifying over all of
    %   BidGrid -- and it costs the same regardless of NumBidders.
    %
    %   Player 2 of each reduced game is a placeholder (its payoffs are
    %   never meaningful) -- only player 1's (the real bidder's) payoffs
    %   and dominance results should ever be read from the returned games.
    %
    %   Author: Ocean Library System
    %   Date: 2026-09-23
    %   Reference: Vickrey (1961) (default rule); the "aggregative game"
    %   reduction generalizes to any OceanStats.GameTheory.Core.IAuctionRule.

    properties (SetAccess = private)
        NumBidders (1,1) double {mustBePositive, mustBeInteger} = 2
        BidGrid (1,:) double = 0:20
        Rule   % OceanStats.GameTheory.Core.IAuctionRule
        Seed (1,1) double {mustBeNonnegative, mustBeInteger} = 0
    end

    methods
        function obj = SingleItemAuctionGame(options)
            arguments
                options.NumBidders (1,1) double {mustBePositive, mustBeInteger} = 2
                options.BidGrid (1,:) double = 0:20
                options.Rule OceanStats.GameTheory.Core.IAuctionRule = OceanStats.GameTheory.Core.SecondPriceRule()
                options.Seed (1,1) double {mustBeNonnegative, mustBeInteger} = 0
            end
            obj.NumBidders = options.NumBidders;
            obj.BidGrid = options.BidGrid;
            obj.Rule = options.Rule;
            obj.Seed = options.Seed;
        end

        function [games, values] = generate(obj, valueRange)
            % GENERATE  Draw random integer private values and build the
            %   per-bidder reduced 2-player games.
            %   valueRange — [min, max] integer bounds for randi (default: bid grid bounds)
            arguments
                obj
                valueRange (1,2) double = [min(obj.BidGrid), max(obj.BidGrid)]
            end
            if obj.Seed > 0
                rng(obj.Seed);
            end
            values = randi(valueRange, 1, obj.NumBidders);

            M = numel(obj.BidGrid);
            bidNames = arrayfun(@(b) sprintf('bid=%g', b), obj.BidGrid, 'UniformOutput', false);

            games = cell(1, obj.NumBidders);
            for i = 1:obj.NumBidders
                PayoffA = zeros(M, M);
                for bi = 1:M
                    for mi = 1:M
                        PayoffA(bi, mi) = obj.Rule.resolveAgainstMax( ...
                            obj.BidGrid(bi), obj.BidGrid(mi), values(i));
                    end
                end
                PayoffB = zeros(M, M);   % placeholder — never meaningful
                games{i} = OceanStats.GameTheory.Core.NormalFormGame( ...
                    'PayoffA', PayoffA, 'PayoffB', PayoffB, ...
                    'StrategyNames', {bidNames, bidNames});
            end
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = "u_i(b_i, m) = \mathrm{Rule.resolveAgainstMax}(b_i, m, v_i), \quad m = \max_{j \neq i} b_j";
            meta.Symbols = [ ...
                struct('Symbol', 'b_i', 'Description', 'Bidder i''s own bid'), ...
                struct('Symbol', 'm',   'Description', 'Max of every other bidder''s bid'), ...
                struct('Symbol', 'v_i', 'Description', 'Bidder i''s private value') ...
                ];
            meta.Assumptions = {sprintf('%d bidders, bid grid [%g..%g] (%d levels), rule = %s', ...
                obj.NumBidders, min(obj.BidGrid), max(obj.BidGrid), numel(obj.BidGrid), class(obj.Rule))};
            meta.IsSymbolic = false;
            meta.Reference = "Vickrey (1961); aggregative-game reduction";
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.ComponentName = "SingleItemAuctionGame";
            spec.Inputs = struct('Name', "valueRange", 'Domain', "GT-ValueRange", 'Dtype', "double", 'Shape', "1x2");
            spec.Outputs = struct('Name', "games,values", 'Domain', "GT-AuctionInstance", 'Dtype', "object", ...
                'Shape', string(mat2str([1, obj.NumBidders])));
            spec.Kind = "VIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "singleitemauctiongame");
            spec.ComposesFrom = {"OceanStats.GameTheory.Core.NormalFormGame", "OceanStats.GameTheory.Core.IAuctionRule"};
        end
    end
end
