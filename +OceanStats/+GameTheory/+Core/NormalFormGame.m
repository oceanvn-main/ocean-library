classdef NormalFormGame < handle
    % NORMALFORMGAME  Strategic-form game G = (N, {S_i}, {u_i}).
    %
    %   A finite set of N players, each with a finite pure-strategy set S_i,
    %   and a payoff function u_i : S_1 x ... x S_N -> R for each player.
    %   This is a data container (a thin facade over an IPayoffStore), not a
    %   stateful process — compare OceanStats.RL.Env.IEnvironment, which
    %   plays the same role for sequential environments.
    %
    %   USAGE (2-player bimatrix shortcut):
    %     g = OceanStats.GameTheory.Core.NormalFormGame( ...
    %             'PayoffA', [3 0; 5 1], 'PayoffB', [3 5; 0 1], ...
    %             'StrategyNames', {{'Cooperate','Defect'}, {'Cooperate','Defect'}});
    %
    %   USAGE (general N-player):
    %     g = OceanStats.GameTheory.Core.NormalFormGame('PayoffTensors', {U1, U2, U3});
    %
    %   Author: Ocean Library System
    %   Date: 2026-09-23
    %
    %   Reference: Bonanno, Game Theory, Ch.1 Sec 1.1.1 — Strategic-Form Games.

    properties (SetAccess = private)
        Store   % OceanStats.GameTheory.Core.IPayoffStore -- payoff storage backend
        Names cell   % 1xN cell of cellstr: Names{i} = strategy labels for player i
    end

    methods
        function obj = NormalFormGame(options)
            arguments
                options.PayoffA double = []
                options.PayoffB double = []
                options.PayoffTensors cell = {}
                options.StrategyNames cell = {}
            end

            if ~isempty(options.PayoffTensors)
                tensors = options.PayoffTensors;
            elseif ~isempty(options.PayoffA) && ~isempty(options.PayoffB)
                if ~isequal(size(options.PayoffA), size(options.PayoffB))
                    error('NormalFormGame:ShapeMismatch', ...
                        'PayoffA and PayoffB must have the same shape.');
                end
                tensors = {options.PayoffA, options.PayoffB};
            else
                error('NormalFormGame:MissingPayoffs', ...
                    ['Provide either PayoffTensors (N-player) or both ' ...
                     'PayoffA/PayoffB (2-player bimatrix).']);
            end

            obj.Store = OceanStats.GameTheory.Core.DensePayoffTensor(tensors);

            n = obj.Store.numPlayers();
            if isempty(options.StrategyNames)
                names = cell(1, n);
                for i = 1:n
                    m = obj.Store.numStrategies(i);
                    names{i} = arrayfun(@(k) sprintf('S%d', k), 1:m, 'UniformOutput', false);
                end
                obj.Names = names;
            else
                if numel(options.StrategyNames) ~= n
                    error('NormalFormGame:NameCountMismatch', ...
                        'StrategyNames must have one cell per player (%d), got %d.', ...
                        n, numel(options.StrategyNames));
                end
                obj.Names = options.StrategyNames(:)';
            end
        end

        function n = numPlayers(obj)
            n = obj.Store.numPlayers();
        end

        function m = numStrategies(obj, player)
            arguments
                obj
                player (1,1) double {mustBePositive, mustBeInteger}
            end
            m = obj.Store.numStrategies(player);
        end

        function names = strategyNames(obj, player)
            arguments
                obj
                player (1,1) double {mustBePositive, mustBeInteger}
            end
            names = obj.Names{player};
        end

        function v = payoff(obj, player, profile)
            % PAYOFF  u_i(s) for player i under joint strategy profile s.
            %   profile — 1xN vector of strategy indices, one per player.
            arguments
                obj
                player (1,1) double {mustBePositive, mustBeInteger}
                profile (1,:) double {mustBePositive, mustBeInteger}
            end
            v = obj.Store.getPayoff(player, profile);
        end

        function tensor = playerTensor(obj, player)
            % PLAYERTENSOR  Player i's full payoff array (for vectorized solvers).
            arguments
                obj
                player (1,1) double {mustBePositive, mustBeInteger}
            end
            tensor = obj.Store.getPlayerTensor(player);
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = "G = (N, \{S_i\}_{i \in N}, \{u_i\}_{i \in N}), \quad u_i : S_1 \times \cdots \times S_N \to \mathbb{R}";
            meta.Symbols = [ ...
                struct('Symbol', 'N',   'Description', 'Finite set of players'), ...
                struct('Symbol', 'S_i', 'Description', 'Player i''s finite pure-strategy set'), ...
                struct('Symbol', 'u_i', 'Description', 'Player i''s payoff function over joint strategy profiles') ...
                ];
            counts = arrayfun(@(i) num2str(obj.numStrategies(i)), 1:obj.numPlayers(), 'UniformOutput', false);
            meta.Assumptions = {sprintf('%d players, strategy counts: [%s]', ...
                obj.numPlayers(), strjoin(counts, ', '))};
            meta.IsSymbolic = false;
            meta.Reference = "Bonanno, Game Theory, Ch.1 Sec 1.1.1 — Strategic-Form Games";
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.ComponentName = "NormalFormGame";
            spec.Inputs = struct('Name', "profile", 'Domain', "GT-JointAction", 'Dtype', "double", ...
                'Shape', string(mat2str([1, obj.numPlayers()])));
            spec.Outputs = struct('Name', "payoff", 'Domain', "GT-Payoff", 'Dtype', "double", 'Shape', "1x1");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "normalformgame");
            spec.ComposesFrom = {};
        end
    end
end
