classdef (Abstract) ILearnableBidStrategy < OceanStats.GameTheory.Core.IBidStrategy
    % ILEARNABLEBIDSTRATEGY  Optional capability: a bid strategy that adapts
    %   after observing the outcome of a round. Mirrors
    %   OceanStats.RL.Env.IModelEnvironment's pattern of layering an opt-in
    %   capability on top of a minimal base contract -- the environment
    %   calls observeOutcome only on strategies that implement this
    %   (checked via isa(...)), never on plain IBidStrategy implementations.
    %
    %   Author: Ocean Library System
    %   Date: 2026-09-23

    methods (Abstract)
        % OBSERVEOUTCOME  Update internal state from the realized outcome
        %   of the round this strategy's last proposeBid() call played in.
        obj = observeOutcome(obj, bid, payoff)
    end
end
