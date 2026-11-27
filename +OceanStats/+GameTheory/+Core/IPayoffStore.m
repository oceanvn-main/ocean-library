classdef (Abstract) IPayoffStore < handle
    % IPAYOFFSTORE  Abstract interface for normal-form game payoff storage.
    %
    %   Decouples HOW payoffs are stored (dense tensor, sparse/Heuristic
    %   Payoff Table, symmetric count-vector, ...) from the NormalFormGame
    %   facade and every solver that reads payoffs, so a bigger backend can
    %   be dropped in later without touching NormalFormGame or any solver
    %   (Open/Closed Principle). Only DensePayoffTensor is implemented
    %   today; sparse/symmetric backends are future implementations of this
    %   same contract.
    %
    %   Author: Ocean Library System
    %   Date: 2026-09-23

    methods (Abstract)
        % NUMPLAYERS  Number of players N.
        n = numPlayers(obj)

        % NUMSTRATEGIES  Number of pure strategies available to a player.
        m = numStrategies(obj, player)

        % GETPAYOFF  Player i's payoff for a joint strategy profile.
        %   profile — 1xN vector of strategy indices (1-based), one per player.
        v = getPayoff(obj, player, profile)

        % GETPLAYERTENSOR  Player i's full payoff array (for vectorized
        %   dominance checks). Backends that cannot cheaply materialize a
        %   dense array may still implement this on demand.
        tensor = getPlayerTensor(obj, player)
    end
end
