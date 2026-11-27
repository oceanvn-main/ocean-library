classdef IteratedDominanceSolver < OceanStats.GameTheory.Solvers.ISolver
    % ITERATEDDOMINANCESOLVER  Strict/weak dominance checks and Iterated
    % Elimination of Strictly Dominated Strategies (IESD).
    %
    %   Strategy s_i' strictly dominates s_i for player i iff
    %     u_i(s_i', s_-i) > u_i(s_i, s_-i)   for every s_-i in S_-i.
    %   (weak dominance: >= everywhere, > somewhere).
    %
    %   solve(game) repeatedly removes every strictly dominated pure
    %   strategy (checked against the CURRENT, shrinking strategy sets)
    %   until no more can be removed — order-independent for strict
    %   dominance. Weak-dominance ELIMINATION is not implemented (the
    %   outcome depends on elimination order and is typically introduced
    %   later); only the weak-dominance CHECK is exposed.
    %
    %   Domination checks are vectorized: both candidate strategies' payoff
    %   slices across every surviving opponent profile are extracted as
    %   N-D sub-arrays in one indexing operation and compared with
    %   all(...)/any(...), rather than looping over opponent profiles.
    %
    %   Author: Ocean Library System
    %   Date: 2026-09-23
    %
    %   Reference: Bonanno, Game Theory, Ch.1 Sec 1.1.2 — Dominated Strategies.

    methods
        function tf = isStrictlyDominated(obj, game, player, strategy)
            arguments
                obj
                game (1,1) OceanStats.GameTheory.Core.NormalFormGame
                player (1,1) double {mustBePositive, mustBeInteger}
                strategy (1,1) double {mustBePositive, mustBeInteger}
            end
            [survivors, candidates, tensor] = obj.fullState_(game, player);
            tf = obj.findDominator_(tensor, player, survivors, strategy, candidates);
        end

        function tf = isWeaklyDominated(obj, game, player, strategy)
            arguments
                obj
                game (1,1) OceanStats.GameTheory.Core.NormalFormGame
                player (1,1) double {mustBePositive, mustBeInteger}
                strategy (1,1) double {mustBePositive, mustBeInteger}
            end
            [survivors, candidates, tensor] = obj.fullState_(game, player);

            idxA = survivors;
            idxA{player} = strategy;
            sliceA = tensor(idxA{:});

            tf = false;
            for aj = 1:numel(candidates)
                aPrime = candidates(aj);
                if aPrime == strategy
                    continue;
                end
                idxAp = survivors;
                idxAp{player} = aPrime;
                sliceAp = tensor(idxAp{:});
                if all(sliceAp(:) >= sliceA(:)) && any(sliceAp(:) > sliceA(:))
                    tf = true;
                    return;
                end
            end
        end

        function result = solve(obj, game)
            % SOLVE  Run Iterated Elimination of Strictly Dominated Strategies.
            %
            %   result.ReducedGame       — NormalFormGame over surviving strategies only
            %   result.SurvivingIndices  — 1xN cell of original-index vectors, one per player
            arguments
                obj
                game (1,1) OceanStats.GameTheory.Core.NormalFormGame
            end
            n = game.numPlayers();
            survivors = cell(1, n);
            tensors = cell(1, n);
            for i = 1:n
                survivors{i} = 1:game.numStrategies(i);
                tensors{i} = game.playerTensor(i);
            end

            log = struct('Round', {}, 'Player', {}, 'EliminatedStrategy', {}, 'DominatingStrategy', {});
            round = 0;

            while true
                round = round + 1;
                toEliminate = cell(1, n);
                anyEliminated = false;

                for i = 1:n
                    candidates = survivors{i};
                    eliminateMask = false(1, numel(candidates));
                    for ai = 1:numel(candidates)
                        a = candidates(ai);
                        [dominated, dominator] = obj.findDominator_(tensors{i}, i, survivors, a, candidates);
                        if dominated
                            eliminateMask(ai) = true;
                            log(end + 1) = struct('Round', round, 'Player', i, ...
                                'EliminatedStrategy', a, 'DominatingStrategy', dominator); %#ok<AGROW>
                            anyEliminated = true;
                        end
                    end
                    toEliminate{i} = candidates(eliminateMask);
                end

                if ~anyEliminated
                    round = round - 1;
                    break;
                end

                for i = 1:n
                    survivors{i} = setdiff(survivors{i}, toEliminate{i}, 'stable');
                end
                if any(cellfun(@isempty, survivors))
                    break;
                end
            end

            obj.History_ = struct('Log', log, 'NumRounds', round, 'SurvivingIndices', {survivors});

            reducedTensors = cell(1, n);
            reducedNames = cell(1, n);
            for i = 1:n
                reducedTensors{i} = tensors{i}(survivors{:});
                allNames = game.strategyNames(i);
                reducedNames{i} = allNames(survivors{i});
            end

            result = struct();
            result.ReducedGame = OceanStats.GameTheory.Core.NormalFormGame( ...
                'PayoffTensors', reducedTensors, 'StrategyNames', reducedNames);
            result.SurvivingIndices = survivors;
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = "s_i^\prime \text{ strictly dominates } s_i \iff u_i(s_i^\prime, s_{-i}) > u_i(s_i, s_{-i}) \; \forall s_{-i} \in S_{-i}";
            meta.Symbols = [ ...
                struct('Symbol', 's_i',       'Description', 'A pure strategy of player i'), ...
                struct('Symbol', 's_i^\prime', 'Description', 'An alternative pure strategy of player i'), ...
                struct('Symbol', 's_{-i}',    'Description', 'A strategy profile of player i''s opponents'), ...
                struct('Symbol', 'u_i',       'Description', 'Player i''s payoff function') ...
                ];
            meta.Assumptions = {['solve() removes strictly dominated pure strategies only, ' ...
                'round by round against the current (shrinking) strategy sets, until none remain.']};
            meta.IsSymbolic = true;
            meta.Reference = "Bonanno, Game Theory, Ch.1 Sec 1.1.2 — Dominated Strategies";
        end

        function spec = getPipelineSpec(~)
            spec = struct();
            spec.ComponentName = "IteratedDominanceSolver";
            spec.Inputs = struct('Name', "game", 'Domain', "GT-NormalFormGame", 'Dtype', "object", 'Shape', "1x1");
            spec.Outputs = struct('Name', "reducedGame", 'Domain', "GT-NormalFormGame", 'Dtype', "object", 'Shape', "1x1");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "iterateddominancesolver");
            spec.ComposesFrom = {};
        end
    end

    methods (Access = private)
        function [survivors, candidates, tensor] = fullState_(~, game, player)
            % FULLSTATE_  Full (unreduced) strategy sets, for standalone checks.
            n = game.numPlayers();
            survivors = arrayfun(@(i) 1:game.numStrategies(i), 1:n, 'UniformOutput', false);
            candidates = survivors{player};
            tensor = game.playerTensor(player);
        end

        function [dominated, dominator] = findDominator_(~, tensor, player, survivors, a, candidates)
            % FINDDOMINATOR_  True iff some other candidate a' strictly
            %   dominates a, checked over the opponents' CURRENT surviving
            %   profiles only (vectorized per candidate pair).
            dominated = false;
            dominator = [];

            idxA = survivors;
            idxA{player} = a;
            sliceA = tensor(idxA{:});

            for aj = 1:numel(candidates)
                aPrime = candidates(aj);
                if aPrime == a
                    continue;
                end
                idxAp = survivors;
                idxAp{player} = aPrime;
                sliceAp = tensor(idxAp{:});
                if all(sliceAp(:) > sliceA(:))
                    dominated = true;
                    dominator = aPrime;
                    return;
                end
            end
        end
    end
end
