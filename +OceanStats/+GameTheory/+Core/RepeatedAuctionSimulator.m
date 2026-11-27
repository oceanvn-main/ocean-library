classdef RepeatedAuctionSimulator < handle
    % REPEATEDAUCTIONSIMULATOR  The auction "environment": executes ONLY
    %   the fixed rules (via an injected IAuctionRule), completely decoupled
    %   from HOW each bidder decides what to bid (via injected IBidStrategy
    %   objects). Swapping the rule, or swapping/adding bidder strategies,
    %   never requires touching this class.
    %
    %   Each round: every strategy proposes a bid (proposeBid); every
    %   bidder's payoff is resolved via
    %   Rule.resolveAgainstMax(ownBid, max-of-others, ownValue); any
    %   strategy implementing ILearnableBidStrategy is then given the
    %   chance to adapt (observeOutcome). Full history of bids/payoffs is
    %   recorded for convergence analysis.
    %
    %   Author: Ocean Library System
    %   Date: 2026-09-23

    properties (SetAccess = private)
        Rule   % OceanStats.GameTheory.Core.IAuctionRule
        BidGrid (1,:) double
        Values (1,:) double
        Strategies cell   % 1xNumBidders cell of IBidStrategy
    end

    methods
        function obj = RepeatedAuctionSimulator(rule, bidGrid, values, strategies)
            arguments
                rule (1,1) OceanStats.GameTheory.Core.IAuctionRule
                bidGrid (1,:) double
                values (1,:) double
                strategies cell
            end
            if numel(values) ~= numel(strategies)
                error('RepeatedAuctionSimulator:CountMismatch', ...
                    'values (%d) and strategies (%d) must have the same length.', ...
                    numel(values), numel(strategies));
            end
            if numel(values) < 2
                error('RepeatedAuctionSimulator:TooFewBidders', ...
                    'An auction needs at least 2 bidders, got %d.', numel(values));
            end
            obj.Rule = rule;
            obj.BidGrid = bidGrid;
            obj.Values = values;
            obj.Strategies = strategies(:)';
        end

        function history = runRounds(obj, numRounds)
            % RUNROUNDS  Play numRounds independent rounds.
            %   history.Bids    — (numRounds x NumBidders) double
            %   history.Payoffs — (numRounds x NumBidders) double
            arguments
                obj
                numRounds (1,1) double {mustBePositive, mustBeInteger}
            end
            n = numel(obj.Strategies);
            bidsHist = zeros(numRounds, n);
            payoffsHist = zeros(numRounds, n);

            for r = 1:numRounds
                bids = zeros(1, n);
                for i = 1:n
                    bids(i) = obj.Strategies{i}.proposeBid(obj.Values(i), obj.BidGrid);
                end

                payoffs = zeros(1, n);
                for i = 1:n
                    others = bids([1:i - 1, i + 1:n]);
                    m = max(others);
                    payoffs(i) = obj.Rule.resolveAgainstMax(bids(i), m, obj.Values(i));
                    if isa(obj.Strategies{i}, 'OceanStats.GameTheory.Core.ILearnableBidStrategy')
                        obj.Strategies{i} = obj.Strategies{i}.observeOutcome(bids(i), payoffs(i));
                    end
                end

                bidsHist(r, :) = bids;
                payoffsHist(r, :) = payoffs;
            end

            history = struct('Bids', bidsHist, 'Payoffs', payoffsHist);
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = "\mathrm{payoffs}(i) = \mathrm{Rule.resolveAgainstMax}(b_i, \max_{j \neq i} b_j, v_i)";
            meta.Symbols = [ ...
                struct('Symbol', 'b_i', 'Description', 'Bidder i''s proposed bid this round'), ...
                struct('Symbol', 'v_i', 'Description', 'Bidder i''s private value') ...
                ];
            meta.Assumptions = {sprintf('%d bidders, rule = %s', numel(obj.Strategies), class(obj.Rule))};
            meta.IsSymbolic = false;
            meta.Reference = "Independent-learners repeated-game simulation";
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.ComponentName = "RepeatedAuctionSimulator";
            spec.Inputs = struct('Name', "numRounds", 'Domain', "GT-Rounds", 'Dtype', "double", 'Shape', "1x1");
            spec.Outputs = struct('Name', "history", 'Domain', "GT-History", 'Dtype', "struct", ...
                'Shape', string(mat2str([numel(obj.Strategies), 2])));
            spec.Kind = "VIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "repeatedauctionsimulator");
            spec.ComposesFrom = {"OceanStats.GameTheory.Core.IAuctionRule", "OceanStats.GameTheory.Core.IBidStrategy"};
        end
    end
end
