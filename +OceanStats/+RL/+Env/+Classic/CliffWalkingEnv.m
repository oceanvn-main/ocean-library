classdef CliffWalkingEnv < OceanStats.RL.Env.Classic.GridTabularEnv
    % CLIFFWALKINGENV  Cliff Walking (Sutton & Barto Example 6.6): 4 x 12 grid, start bottom-left, goal
    % bottom-right, the cells between them are the cliff. Every step costs -1; stepping into the cliff costs
    % -100 and sends the agent back to the start (the episode continues). Actions: 1 up, 2 right, 3 down, 4 left.
    % Same dynamics as Gymnasium's CliffWalking-v1 (there 0-based, index 12*row+col, actions 0-3 = up, right, down, left).
    methods
        function obj = CliffWalkingEnv(options)
            arguments
                options.Rows (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(options.Rows, 2)} = 4
                options.Cols (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(options.Cols, 3)} = 12
            end
            R = options.Rows;  C = options.Cols;
            obj.Rows = R;  obj.Cols = C;
            S = R * C;
            start = obj.idx(R, 1);  goal = obj.idx(R, C);
            cliff = false(S, 1);
            for c = 2:C-1, cliff(obj.idx(R, c)) = true; end
            moves = [-1 0; 0 1; 1 0; 0 -1];
            model = cell(S, 4);
            terminal = false(S, 1);  terminal(goal) = true;
            for s = 1:S
                if terminal(s) || cliff(s), continue; end        % cliff cells are never occupied
                [r, c] = obj.rc(s);
                for a = 1:4
                    r2 = min(max(r + moves(a, 1), 1), R);
                    c2 = min(max(c + moves(a, 2), 1), C);
                    s2 = obj.idx(r2, c2);
                    rew = -1;
                    if cliff(s2), s2 = start; rew = -100; end
                    model{s, a} = struct('Next', s2, 'Prob', 1, 'Reward', rew);
                end
            end
            st = zeros(S, 1);  st(start) = 1;
            % cliff cells are unreachable: declare them terminal-like (no actions) so nothing treats them as states to visit
            obj.setModel_(model, terminal | cliff, st);
        end
    end
end
