classdef Dijkstra
    % DIJKSTRA  Shortest-path solver using Dijkstra's algorithm.
    %
    % Pure math implementation — no MATLAB built-in shortestpath().
    % Uses a simple priority queue (linear scan for min) which gives
    % O(V^2) complexity. Sufficient for transit networks (V < 1000).
    %
    % For sparse graphs needing O((V+E) log V), a binary heap can be
    % substituted later.
    %
    % Usage:
    %   [path, dist] = OceanMath.Algorithms.Graph.Dijkstra.solve(W, 1, 5)
    %
    % See also: OceanMath.Graph.DigraphCus

    methods (Static)
        function [path, dist] = solve(W, src, tgt)
            % SOLVE  Find shortest path from src to tgt.
            %
            %   [path, dist] = Dijkstra.solve(W, srcId, tgtId)
            %
            %   Inputs:
            %     W   - (V x V) weight matrix. W(i,j) > 0 means edge with
            %           that weight. W(i,j) = 0 means no edge.
            %     src - source node ID (1-indexed)
            %     tgt - target node ID (1-indexed)
            %
            %   Outputs:
            %     path - [1 x N] vector of node IDs from src to tgt
            %            Empty [] if no path exists.
            %     dist - total distance. Inf if no path exists.
            arguments
                W   (:,:) double
                src (1,1) double {mustBePositive, mustBeInteger}
                tgt (1,1) double {mustBePositive, mustBeInteger}
            end

            V = size(W, 1);

            % Distance array: dist(i) = shortest known distance from src to i
            distArr = inf(1, V);
            distArr(src) = 0;

            % Previous node array for path reconstruction
            prev = zeros(1, V);

            % Visited flag
            visited = false(1, V);

            for iter = 1:V
                % --- Find unvisited node with smallest distance ---
                % (simple linear scan — O(V) per iteration)
                minDist = inf;
                u = -1;
                for i = 1:V
                    if ~visited(i) && distArr(i) < minDist
                        minDist = distArr(i); %Update new shortest distance
                        u = i;
                    end
                end

                % No reachable unvisited nodes left
                if u == -1
                    break;
                end

                % Early exit if we reached the target
                if u == tgt
                    break;
                end

                visited(u) = true;

                % --- Relax all edges from u ---
                for v = 1:V
                    if W(u, v) > 0 && ~visited(v)
                        alt = distArr(u) + W(u, v);
                        if alt < distArr(v)
                            distArr(v) = alt;
                            prev(v) = u;
                        end
                    end
                end
            end

            % --- Reconstruct path ---
            dist = distArr(tgt);
            if isinf(dist)
                path = [];
                return;
            end

            % Backtrack from tgt to src via prev[]
            path = tgt;
            current = tgt;
            while current ~= src
                current = prev(current);
                if current == 0
                    % Should not happen if dist is finite
                    path = [];
                    dist = inf;
                    return;
                end
                path = [current, path]; %#ok<AGROW>
            end
        end
    end
end
