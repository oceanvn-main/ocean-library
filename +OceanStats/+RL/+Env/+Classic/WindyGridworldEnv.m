classdef WindyGridworldEnv < OceanStats.RL.Env.Classic.GridTabularEnv
    % WINDYGRIDWORLDENV  Windy Gridworld (Sutton & Barto Example 6.5): 7 x 10 grid, a crosswind pushes the agent
    % UP by the column's strength (0 0 0 1 1 1 2 2 1 0) applied from the column it is in; start (4,1), goal (4,8);
    % reward -1 per step. Actions: 1 up, 2 right, 3 down, 4 left; KingMoves adds the four diagonals (5-8);
    % StayAction adds a 'stay' action (9 with king moves, else 5). StochasticWind (Ex. 6.10): in windy columns
    % the wind varies by -1, 0, +1 with equal probability. The optimal path takes 15 steps (7 with king moves,
    % 7 with stay too), which the tests confirm by solving the exact model.
    methods
        function obj = WindyGridworldEnv(options)
            arguments
                options.Wind (1,:) double = [0 0 0 1 1 1 2 2 1 0]
                options.Rows (1,1) double {mustBeInteger, mustBePositive} = 7
                options.Start (1,2) double = [4 1]
                options.Goal (1,2) double = [4 8]
                options.KingMoves (1,1) logical = false
                options.StayAction (1,1) logical = false
                options.StochasticWind (1,1) logical = false
            end
            R = options.Rows;  C = numel(options.Wind);
            obj.Rows = R;  obj.Cols = C;
            S = R * C;
            moves = [-1 0; 0 1; 1 0; 0 -1];                                % up right down left
            if options.KingMoves, moves = [moves; -1 1; 1 1; 1 -1; -1 -1]; end   % + diagonals
            if options.StayAction, moves = [moves; 0 0]; end
            A = size(moves, 1);
            goal = obj.idx(options.Goal(1), options.Goal(2));
            model = cell(S, A);
            terminal = false(S, 1);  terminal(goal) = true;
            for s = 1:S
                if terminal(s), continue; end
                [r, c] = obj.rc(s);
                w0 = options.Wind(c);
                if options.StochasticWind && w0 > 0, winds = w0 + [-1 0 1]; probs = [1 1 1] / 3; else, winds = w0; probs = 1; end
                for a = 1:A
                    nxt = zeros(numel(winds), 1);
                    for k = 1:numel(winds)
                        r2 = min(max(r + moves(a, 1) - winds(k), 1), R);
                        c2 = min(max(c + moves(a, 2), 1), C);
                        nxt(k) = obj.idx(r2, c2);
                    end
                    model{s, a} = struct('Next', nxt, 'Prob', probs(:), 'Reward', -ones(numel(winds), 1));
                end
            end
            st = zeros(S, 1);  st(obj.idx(options.Start(1), options.Start(2))) = 1;
            obj.setModel_(model, terminal, st);
        end
    end
end
