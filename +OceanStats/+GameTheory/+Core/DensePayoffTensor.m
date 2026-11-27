classdef DensePayoffTensor < OceanStats.GameTheory.Core.IPayoffStore
    % DENSEPAYOFFTENSOR  Payoff storage backed by one N-D array per player.
    %
    %   The default (and, for now, only) IPayoffStore implementation.
    %   Mirrors OpenSpiel's matrix_game/tensor_game: for N players each with
    %   up to M strategies, player i's tensor has shape [M_1,...,M_N] and
    %   PayoffTensors{i}(s_1,...,s_N) = u_i(s).
    %
    %   Size is O(N * M^N) — intractable for large N (e.g. N=30 strategic
    %   players would need billions of entries even at M=2). This is
    %   deliberately the SMALL-GAME backend; large populations should be
    %   composed from many small games (see
    %   OceanStats.GameTheory.Core.RoundRobinTournament) rather than grown
    %   into one big tensor. Sparse (Heuristic Payoff Table) and symmetric
    %   (count-vector) backends are natural future siblings of this class.
    %
    %   Author: Ocean Library System
    %   Date: 2026-09-23

    properties (SetAccess = private)
        PayoffTensors cell   % 1xN cell; PayoffTensors{i} is player i's N-D double array
    end

    methods
        function obj = DensePayoffTensor(payoffTensors)
            arguments
                payoffTensors cell
            end
            n = numel(payoffTensors);
            if n == 0
                error('DensePayoffTensor:EmptyPayoffs', ...
                    'At least one player payoff tensor is required.');
            end
            sz1 = size(payoffTensors{1});
            for i = 2:n
                szi = size(payoffTensors{i});
                if ~isequal(sz1, szi)
                    error('DensePayoffTensor:ShapeMismatch', ...
                        'Player %d payoff tensor shape [%s] does not match player 1''s [%s].', ...
                        i, num2str(szi), num2str(sz1));
                end
            end
            obj.PayoffTensors = payoffTensors(:)';
        end

        function n = numPlayers(obj)
            n = numel(obj.PayoffTensors);
        end

        function m = numStrategies(obj, player)
            arguments
                obj
                player (1,1) double {mustBePositive, mustBeInteger}
            end
            % size(A, dim) returns 1 for dim > ndims(A), so this is safe for
            % trailing-singleton players without manual padding.
            m = size(obj.PayoffTensors{1}, player);
        end

        function v = getPayoff(obj, player, profile)
            arguments
                obj
                player (1,1) double {mustBePositive, mustBeInteger}
                profile (1,:) double {mustBePositive, mustBeInteger}
            end
            idx = num2cell(profile);
            v = obj.PayoffTensors{player}(idx{:});
        end

        function tensor = getPlayerTensor(obj, player)
            arguments
                obj
                player (1,1) double {mustBePositive, mustBeInteger}
            end
            tensor = obj.PayoffTensors{player};
        end
    end
end
