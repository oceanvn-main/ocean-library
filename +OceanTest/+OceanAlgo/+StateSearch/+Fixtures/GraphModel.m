classdef GraphModel < OceanAlgo.StateSearch.IDistributionModel
    % GRAPHMODEL  Deterministic weighted digraph as a planning model. State = node id; action label = target
    % node id; reward = -W(s, a) (cost W); isGoal(goal) = true. W(i,j) > 0 is an edge, 0 means none (the
    % convention of OceanMath.Algorithms.Graph.Dijkstra).
    properties (SetAccess = immutable)
        W double
        Goal (1,1) double
    end
    methods
        function obj = GraphModel(W, goal)
            obj.W = W; obj.Goal = goal;
        end
        function a = actions(obj, s), a = find(obj.W(s, :) > 0); end
        function k = key(~, s), k = s; end
        function tf = isGoal(obj, s), tf = s == obj.Goal; end
        function o = outcomes(obj, s, a)
            o = struct('Next', a, 'Reward', -obj.W(s, a), 'Terminated', false, 'Probability', 1);
        end
    end
    methods (Static)
        function W = randomGraph(n, density, stream)
            % Random sparse digraph with integer weights 1..9 and a guaranteed path 1 -> 2 -> ... -> n.
            W = (rand(stream, n) < density) .* randi(stream, 9, n);
            W(1:n+1:end) = 0;
            for i = 1:n-1
                if W(i, i+1) == 0, W(i, i+1) = randi(stream, 9); end
            end
        end
        function h = gridHeuristic(rows, cols, goal)
            % Manhattan distance on a 4-connected unit grid, node = (r-1)*cols + c (consistent for unit costs).
            gc = mod(goal - 1, cols) + 1; gr = floor((goal - 1) / cols) + 1;
            h = @(s) abs(mod(s - 1, cols) + 1 - gc) + abs(floor((s - 1) / cols) + 1 - gr);
            assert(rows >= gr);
        end
        function W = unitGrid(rows, cols, blocked)
            % 4-connected grid with unit costs; blocked cells have no edges.
            n = rows * cols; W = zeros(n);
            for s = 1:n
                if any(blocked == s), continue; end
                r = floor((s - 1) / cols) + 1; c = mod(s - 1, cols) + 1;
                nb = [r-1 c; r+1 c; r c-1; r c+1];
                for k = 1:4
                    if nb(k,1) >= 1 && nb(k,1) <= rows && nb(k,2) >= 1 && nb(k,2) <= cols
                        t = (nb(k,1) - 1) * cols + nb(k,2);
                        if ~any(blocked == t), W(s, t) = 1; end
                    end
                end
            end
        end
    end
end
