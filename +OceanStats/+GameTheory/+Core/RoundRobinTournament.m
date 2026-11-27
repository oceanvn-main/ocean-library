classdef RoundRobinTournament < handle
    % ROUNDROBINTOURNAMENT  Decomposed many-player tournament.
    %
    %   Scores P players by playing every k-player group through one shared
    %   BASE game, instead of building a single P-player payoff tensor
    %   (which would need O(M^P) entries — intractable past a handful of
    %   players; even P=30, M=2 is already ~1e9 entries per player).
    %
    %   Architecture: the same decomposition Robert Axelrod used for his
    %   1980/1984 Iterated Prisoner's Dilemma tournaments and that the
    %   modern Axelrod-Python library (github.com/Axelrod-Python/Axelrod)
    %   still uses — split an N-player population into small, bounded-size
    %   groups (Axelrod used pairs, k=2), score each group through the SAME
    %   small base game, and sum. It also mirrors how shipping game engines
    %   (Rockstar's RAGE / GTA Online) scale many-agent worlds: bounded-size
    %   sessions instead of one shared joint simulation, so per-group cost
    %   never grows with total population size P.
    %
    %   The session/group size k is fixed by BaseGame.numPlayers() — pass a
    %   2-player BaseGame for classic pairwise (Axelrod-style) tournaments,
    %   or a k-player BaseGame for richer k-way group interactions.
    %
    %   ASSUMES BaseGame is symmetric across seats (players are
    %   interchangeable — the common case for population tournaments: every
    %   seat's payoff tensor depends only on "my strategy" and "the other
    %   seats' strategies", not on seat identity). Groups are formed by
    %   exhaustive combination (nchoosek); for very large P and/or k this
    %   should switch to random sampling of groups — not needed at the
    %   textbook/thesis scale this class targets (documented, not built).
    %
    %   ONE-SHOT ONLY: each group plays BaseGame exactly once. History-
    %   dependent repeated play (Tit-for-Tat etc., Axelrod's actual 200-round
    %   protocol) needs a policy abstraction and is out of scope here.
    %
    %   Author: Ocean Library System
    %   Date: 2026-09-23
    %
    %   Reference: Axelrod (1980, 1984); Axelrod-Python (github.com/Axelrod-Python/Axelrod).

    properties (SetAccess = private)
        BaseGame   % OceanStats.GameTheory.Core.NormalFormGame
        NumPlayers (1,1) double {mustBePositive, mustBeInteger} = 1
        PlayerStrategies (1,:) double {mustBePositive, mustBeInteger} = 1
        SessionSize (1,1) double {mustBePositive, mustBeInteger} = 1
    end

    methods
        function obj = RoundRobinTournament(baseGame, playerStrategies)
            arguments
                baseGame (1,1) OceanStats.GameTheory.Core.NormalFormGame
                playerStrategies (1,:) double {mustBePositive, mustBeInteger}
            end
            obj.BaseGame = baseGame;
            obj.SessionSize = baseGame.numPlayers();
            obj.NumPlayers = numel(playerStrategies);

            if obj.NumPlayers < obj.SessionSize
                error('RoundRobinTournament:TooFewPlayers', ...
                    'Need at least %d players for a %d-player base game, got %d.', ...
                    obj.SessionSize, obj.SessionSize, obj.NumPlayers);
            end
            maxStrategy = baseGame.numStrategies(1);
            if any(playerStrategies > maxStrategy)
                error('RoundRobinTournament:InvalidStrategy', ...
                    'PlayerStrategies must index into BaseGame''s strategy set (1..%d).', maxStrategy);
            end
            obj.PlayerStrategies = playerStrategies;
        end

        function [scores, groups] = run(obj)
            % RUN  Play every k-player group exactly once; return total score per player.
            %
            %   scores — 1 x NumPlayers double, total accumulated payoff.
            %   groups — (numGroups x SessionSize) double, the player-index
            %            groups played (row order from nchoosek).
            k = obj.SessionSize;
            P = obj.NumPlayers;
            groups = nchoosek(1:P, k);
            scores = zeros(1, P);

            for g = 1:size(groups, 1)
                players = groups(g, :);
                profile = obj.PlayerStrategies(players);
                for seat = 1:k
                    scores(players(seat)) = scores(players(seat)) + obj.BaseGame.payoff(seat, profile);
                end
            end
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = "\mathrm{Score}(i) = \sum_{g \,:\, i \in g} u_{\mathrm{seat}(i,g)}(s_g), \quad g \in \binom{[P]}{k}";
            meta.Symbols = [ ...
                struct('Symbol', 'P',   'Description', 'Total population size'), ...
                struct('Symbol', 'k',   'Description', 'Session/group size (BaseGame.numPlayers())'), ...
                struct('Symbol', 'g',   'Description', 'A k-player group drawn from the population'), ...
                struct('Symbol', 's_g', 'Description', 'The strategy profile of group g''s members') ...
                ];
            meta.Assumptions = {sprintf('%d players, session size %d, base game with %d strategies/player', ...
                obj.NumPlayers, obj.SessionSize, obj.BaseGame.numStrategies(1))};
            meta.IsSymbolic = false;
            meta.Reference = "Axelrod (1980, 1984); Axelrod-Python (github.com/Axelrod-Python/Axelrod)";
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.ComponentName = "RoundRobinTournament";
            spec.Inputs = struct('Name', "playerStrategies", 'Domain', "GT-Population", 'Dtype', "double", ...
                'Shape', string(mat2str([1, obj.NumPlayers])));
            spec.Outputs = struct('Name', "scores", 'Domain', "GT-Scores", 'Dtype', "double", ...
                'Shape', string(mat2str([1, obj.NumPlayers])));
            spec.Kind = "VIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "roundrobintournament");
            spec.ComposesFrom = {"OceanStats.GameTheory.Core.NormalFormGame"};
        end
    end
end
