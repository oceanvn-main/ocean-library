classdef MonteCarloBidStrategy < OceanStats.GameTheory.Core.ILearnableBidStrategy
    % MONTECARLOBIDSTRATEGY  Epsilon-greedy K-armed-bandit bid strategy.
    %
    %   Maintains a running sample-average payoff estimate (QValues) per
    %   candidate bid in a FIXED BidGrid, chosen at construction time to
    %   match the auction it will play in. Deliberately does NOT use its
    %   own private value to pick a bid -- it discovers the best bid purely
    %   empirically from realized payoffs, the way a K-armed bandit learns
    %   which arm pays best without being told the true means.
    %
    %   Action selection reuses OceanStats.RL.TD.Functional.epsilonGreedyProbs
    %   (a stateless numeric utility, not an RL environment/agent
    %   abstraction) -- a deliberate, narrow reuse across the GameTheory/RL
    %   boundary.
    %
    %   Author: Ocean Library System
    %   Date: 2026-09-23
    %   Reference: Sutton & Barto (2018) sec. 2.2, 2.4 -- sample-average action-value method

    properties (SetAccess = private)
        BidGrid (1,:) double
        Epsilon (1,1) double {mustBeNonnegative, mustBeLessThanOrEqual(Epsilon, 1)} = 0.1
        QValues (1,:) double
        Counts (1,:) double
        History (1,:) double = []   % chosen bid index per round
    end

    methods
        function obj = MonteCarloBidStrategy(bidGrid, options)
            arguments
                bidGrid (1,:) double
                options.Epsilon (1,1) double {mustBeNonnegative, mustBeLessThanOrEqual(options.Epsilon, 1)} = 0.1
                options.Seed (1,1) double {mustBeNonnegative, mustBeInteger} = 0
            end
            obj.BidGrid = bidGrid;
            obj.Epsilon = options.Epsilon;
            obj.QValues = zeros(1, numel(bidGrid));
            obj.Counts = zeros(1, numel(bidGrid));
            if options.Seed > 0
                rng(options.Seed);
            end
        end

        function bid = proposeBid(obj, ~, ~)
            % PROPOSEBID  Epsilon-greedy draw over the learner's own BidGrid.
            %   The value/bidGrid inputs (the IBidStrategy contract) are
            %   ignored here — this strategy tracks its own fixed grid from
            %   construction, so its Q-estimates stay meaningful across rounds.
            valid = true(size(obj.QValues));
            p = OceanStats.RL.TD.Functional.epsilonGreedyProbs(obj.QValues, valid, obj.Epsilon);
            idx = obj.sampleFromProbs_(p);
            bid = obj.BidGrid(idx);
            obj.History(end + 1) = idx;
        end

        function obj = observeOutcome(obj, bid, payoff)
            idx = find(obj.BidGrid == bid, 1);
            obj.Counts(idx) = obj.Counts(idx) + 1;
            obj.QValues(idx) = obj.QValues(idx) + (payoff - obj.QValues(idx)) / obj.Counts(idx);
        end

        function meta = getMetadata(~)
            meta = struct();
            meta.SymbolicLatex = "Q_{n+1}(a) = Q_n(a) + \dfrac{1}{N_n(a)}\left[R_n - Q_n(a)\right]";
            meta.Symbols = [ ...
                struct('Symbol', 'Q_n(a)', 'Description', 'Sample-average payoff estimate for bid a after n picks'), ...
                struct('Symbol', 'R_n',    'Description', 'Realized payoff from the most recent round') ...
                ];
            meta.Assumptions = {'Epsilon-greedy exploration; ties for the greedy bid split evenly (epsilonGreedyProbs).'};
            meta.IsSymbolic = true;
            meta.Reference = "Sutton & Barto (2018) sec. 2.2, 2.4 -- sample-average action-value method";
        end

        function spec = getPipelineSpec(~)
            spec = struct();
            spec.ComponentName = "MonteCarloBidStrategy";
            spec.Inputs = struct('Name', "value,bidGrid", 'Domain', "GT-BidContext", 'Dtype', "double", 'Shape', "1x1");
            spec.Outputs = struct('Name', "bid", 'Domain', "GT-Bid", 'Dtype', "double", 'Shape', "1x1");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "montecarlobidstrategy");
            spec.ComposesFrom = {};
        end
    end

    methods (Access = private)
        function idx = sampleFromProbs_(~, p)
            edges = cumsum(p);
            edges(end) = 1;   % guard against floating-point drift
            r = rand();
            idx = find(r < edges, 1);
            if isempty(idx)
                idx = numel(p);
            end
        end
    end
end
