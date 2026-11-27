classdef NetworkAnalytics
    % NETWORKANALYTICS  Static utility class for graph-level metrics.
    %
    % All methods take a DigraphCus as input — no domain-specific logic.
    %
    % Usage:
    %   L = OceanMath.Algorithms.Graph.NetworkAnalytics.totalLength(g);
    %   tf = OceanMath.Algorithms.Graph.NetworkAnalytics.isConnected(g);
    %
    % See also: OceanMath.Graph.DigraphCus

    methods (Static)

        function L = totalLength(graph)
            % TOTALLENGTH  Sum of all unique edge weights.
            W = graph.getAdjacencyMatrix();
            if graph.isDirected()
                L = sum(W(:));
            else
                L = sum(W(:)) / 2;
            end
        end

        function avg = averageEdgeLength(graph)
            % AVERAGEEDGELENGTH  Mean edge weight.
            W = graph.getAdjacencyMatrix();
            if graph.isDirected()
                nEdges = nnz(W);
            else
                nEdges = nnz(W) / 2;
            end
            if nEdges == 0
                avg = 0;
                return;
            end
            avg = OceanMath.Algorithms.Graph.NetworkAnalytics.totalLength(graph) / nEdges;
        end

        function dist = degreeDistribution(graph)
            % DEGREEDISTRIBUTION  Return [1 x V] vector of node degrees.
            V = graph.getNodeCount();
            dist = zeros(1, V);
            for i = 1:V
                dist(i) = graph.degree(i);
            end
        end

        function tf = isConnected(graph)
            % ISCONNECTED  Check if all nodes are reachable from node 1.
            %   Uses BFS traversal.
            V = graph.getNodeCount();
            if V == 0, tf = true; return; end

            W = graph.getAdjacencyMatrix();
            visited = false(1, V);
            queue = 1;
            visited(1) = true;

            while ~isempty(queue)
                u = queue(1);
                queue(1) = [];

                % Find neighbors
                if graph.isDirected()
                    nbrs = find(W(u,:) > 0 | W(:,u)' > 0);
                else
                    nbrs = find(W(u,:) > 0);
                end

                for v = nbrs
                    if ~visited(v)
                        visited(v) = true;
                        queue(end+1) = v; %#ok<AGROW>
                    end
                end
            end
            tf = all(visited);
        end

        function labels = connectedComponents(graph)
            % CONNECTEDCOMPONENTS  Return [1 x V] component labels.
            %   nodes in the same connected component share the same label.
            V = graph.getNodeCount();
            labels = zeros(1, V);
            W = graph.getAdjacencyMatrix();
            componentId = 0;

            for start = 1:V
                if labels(start) > 0, continue; end
                componentId = componentId + 1;

                % BFS from start
                queue = start;
                labels(start) = componentId;
                while ~isempty(queue)
                    u = queue(1);
                    queue(1) = [];
                    if graph.isDirected()
                        nbrs = find(W(u,:) > 0 | W(:,u)' > 0);
                    else
                        nbrs = find(W(u,:) > 0);
                    end
                    for v = nbrs
                        if labels(v) == 0
                            labels(v) = componentId;
                            queue(end+1) = v; %#ok<AGROW>
                        end
                    end
                end
            end
        end

        function n = numComponents(graph)
            % NUMCOMPONENTS  Number of connected components.
            labels = OceanMath.Algorithms.Graph.NetworkAnalytics.connectedComponents(graph);
            n = max(labels);
        end

        function d = density(graph)
            % DENSITY  Edge density of the graph.
            %   0 = no edges, 1 = complete graph.
            V = graph.getNodeCount();
            if V < 2, d = 0; return; end
            maxEdges = V * (V - 1);
            W = graph.getAdjacencyMatrix();
            if ~graph.isDirected(), maxEdges = maxEdges / 2; end
            nEdges = nnz(W);
            if ~graph.isDirected(), nEdges = nEdges / 2; end
            d = nEdges / maxEdges;
        end
    end
end
