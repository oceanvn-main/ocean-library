classdef test_graph < OceanTest.BaseTestCase
    % TEST_GRAPH  Unit tests for +OceanMath/+Graph/ package.

    methods (Test)

        % ============================================================== %
        %  Node
        % ============================================================== %

        function test_node_construction(testCase)
            n = OceanMath.Graph.Node(1, 'Orchard', [3, 7]);
            testCase.assertEqual(n.Id, 1);
            testCase.assertEqual(n.Name, 'Orchard');
            testCase.assertEqual(n.Position, [3, 7]);
        end

        function test_node_metadata(testCase)
            meta = struct('zone', 1, 'interchange', true);
            n = OceanMath.Graph.Node(2, 'Dhoby Ghaut', [4, 6], meta);
            testCase.assertTrue(n.Metadata.interchange);
            testCase.assertEqual(n.Metadata.zone, 1);
        end

        % ============================================================== %
        %  Edge
        % ============================================================== %

        function test_edge_explicit_weight(testCase)
            e = OceanMath.Graph.Edge(1, 2, 5.3);
            testCase.assertEqual(e.Source, 1);
            testCase.assertEqual(e.Target, 2);
            testCase.assertEqual(e.Weight, 5.3);
        end

        function test_edge_with_label(testCase)
            e = OceanMath.Graph.Edge(1, 2, 5.3, 'Label', 'NSL');
            testCase.assertEqual(e.Source, 1);
            testCase.assertEqual(e.Target, 2);
            testCase.assertEqual(e.Weight, 5.3);
            testCase.assertEqual(e.Label, 'NSL');
        end

        % ============================================================== %
        %  NetworkGraph — Building
        % ============================================================== %

        function test_add_nodes_and_edges(testCase)
            g = OceanMath.Graph.DigraphCus();
            g.addNode('A', [0, 0]);
            g.addNode('B', [3, 0]);
            g.addNode('C', [3, 4]);
            g.addEdge(1, 2, 3);
            g.addEdge(2, 3, 4);
            g.addEdge(1, 3, 5);

            testCase.assertEqual(g.getNodeCount(), 3);
            testCase.assertEqual(g.getEdgeCount(), 3);
        end

        function test_adjacency_matrix(testCase)
            g = OceanMath.Graph.DigraphCus();
            g.addNode('A', [0, 0]);
            g.addNode('B', [3, 0]);
            g.addNode('C', [1, 1]);
            g.addEdge(1, 2, 3);
            g.addEdge(2, 3, 4);

            A = g.getAdjacencyMatrix();
            % Undirected: symmetric
            testCase.assertEqual(A(1,2), 3);
            testCase.assertEqual(A(2,1), 3);
            testCase.assertEqual(A(2,3), 4);
            testCase.assertEqual(A(1,3), 0, ...
                'No direct edge from A to C.');
        end

        function test_directed_graph(testCase)
            g = OceanMath.Graph.DigraphCus('Directed', true);
            g.addNode('X', [0,0]);
            g.addNode('Y', [1,0]);
            g.addEdge(1, 2, 7);

            A = g.getAdjacencyMatrix();
            testCase.assertEqual(A(1,2), 7);
            testCase.assertEqual(A(2,1), 0, ...
                'Directed: reverse edge should not exist.');
        end

        function test_neighbors(testCase)
            g = OceanMath.Graph.DigraphCus();
            g.addNode('A', [0,0]);
            g.addNode('B', [1,0]);
            g.addNode('C', [2,0]);
            g.addNode('D', [3,0]);
            g.addEdge(1, 2, 1);
            g.addEdge(1, 3, 2);

            nbrs = g.neighbors(1);
            testCase.assertEqual(sort(nbrs), [2, 3]);
            testCase.assertEqual(g.degree(1), 2);
            testCase.assertEqual(g.degree(4), 0);
        end

        function test_has_edge(testCase)
            g = OceanMath.Graph.DigraphCus();
            g.addNode('A', [0,0]);
            g.addNode('B', [1,0]);
            g.addEdge(1, 2, 5);

            testCase.assertTrue(g.hasEdge(1, 2));
            testCase.assertTrue(g.hasEdge(2, 1));  % undirected
        end

        function test_get_node_by_name(testCase)
            g = OceanMath.Graph.DigraphCus();
            g.addNode('Orchard', [3, 7]);
            g.addNode('Dhoby Ghaut', [4, 6]);

            n = g.getNodeByName('Dhoby Ghaut');
            testCase.assertEqual(n.Id, 2);
            testCase.assertEqual(n.Position, [4, 6]);

            missing = g.getNodeByName('Nonexistent');
            testCase.assertEmpty(missing);
        end

        % ============================================================== %
        %  Dijkstra — Shortest Path
        % ============================================================== %

        function test_dijkstra_triangle(testCase)
            %  A --3-- B
            %  |       |
            %  5       4
            %  |       |
            %  +--C----+
            % Shortest A->C: A->B->C = 7 (not A->C = 5... wait, 3+4=7 > 5)
            % Actually shortest A->C = 5 (direct)
            g = OceanMath.Graph.DigraphCus();
            g.addNode('A', [0,0]);
            g.addNode('B', [3,0]);
            g.addNode('C', [3,4]);
            g.addEdge(1, 2, 3);  % A-B
            g.addEdge(2, 3, 4);  % B-C
            g.addEdge(1, 3, 5);  % A-C

            [path, dist] = OceanMath.Algorithms.Graph.Dijkstra.solve(g.getAdjacencyMatrix(), 1, 3);
            testCase.assertEqual(dist, 5);
            testCase.assertEqual(path, [1, 3]);
        end

        function test_dijkstra_longer_path(testCase)
            %  A --1-- B --1-- C --1-- D
            %  |                       |
            %  +---------- 10 ---------+
            g = OceanMath.Graph.DigraphCus();
            g.addNode('A', [0,0]);
            g.addNode('B', [1,0]);
            g.addNode('C', [2,0]);
            g.addNode('D', [3,0]);
            g.addEdge(1, 2, 1);
            g.addEdge(2, 3, 1);
            g.addEdge(3, 4, 1);
            g.addEdge(1, 4, 10);

            [path, dist] = OceanMath.Algorithms.Graph.Dijkstra.solve(g.getAdjacencyMatrix(), 1, 4);
            testCase.assertEqual(dist, 3);
            testCase.assertEqual(path, [1, 2, 3, 4]);
        end

        function test_dijkstra_no_path(testCase)
            g = OceanMath.Graph.DigraphCus();
            g.addNode('A', [0,0]);
            g.addNode('B', [1,0]);
            % No edges

            [path, dist] = OceanMath.Algorithms.Graph.Dijkstra.solve(g.getAdjacencyMatrix(), 1, 2);
            testCase.assertEmpty(path);
            testCase.assertEqual(dist, inf);
        end

        function test_dijkstra_same_node(testCase)
            g = OceanMath.Graph.DigraphCus();
            g.addNode('A', [0,0]);

            [path, dist] = OceanMath.Algorithms.Graph.Dijkstra.solve(g.getAdjacencyMatrix(), 1, 1);
            testCase.assertEqual(dist, 0);
            testCase.assertEqual(path, 1);
        end

        % ============================================================== %
        %  Network Statistics
        % ============================================================== %

        function test_analytics_total_length(testCase)
            g = OceanMath.Graph.DigraphCus();
            g.addNode('A', [0,0]);
            g.addNode('B', [1,0]);
            g.addNode('C', [2,0]);
            g.addEdge(1, 2, 3);
            g.addEdge(2, 3, 4);

            L = OceanMath.Algorithms.Graph.NetworkAnalytics.totalLength(g);
            testCase.assertEqual(L, 7);

            avg = OceanMath.Algorithms.Graph.NetworkAnalytics.averageEdgeLength(g);
            testCase.assertEqual(avg, 3.5);
        end

        function test_density(testCase)
            g = OceanMath.Graph.DigraphCus();
            g.addNode('A', [0,0]);
            g.addNode('B', [1,0]);
            g.addNode('C', [2,0]);
            % Complete undirected: 3 edges possible
            g.addEdge(1, 2, 1);
            g.addEdge(2, 3, 1);
            g.addEdge(1, 3, 1);

            testCase.assertEqual(OceanMath.Algorithms.Graph.NetworkAnalytics.density(g), 1, 'AbsTol', 1e-10, ...
                'Complete graph should have density 1.');
        end

        % ============================================================== %
        %  Serialization
        % ============================================================== %

        function test_to_struct(testCase)
            g = OceanMath.Graph.DigraphCus();
            g.addNode('A', [0,0]);
            g.addNode('B', [1,0]);
            g.addEdge(1, 2, 5);

            s = g.toStruct();
            testCase.assertEqual(numel(s.Nodes), 2);
            testCase.assertEqual(numel(s.Edges), 1);
            testCase.assertEqual(s.Edges{1}.Weight, 5);
            testCase.assertFalse(s.Directed);
        end

        % ============================================================== %
        %  Integration: Spline edge + graph
        % ============================================================== %

        function test_transit_spline_integration(testCase)
            tn = OceanMath.Transit.TransitNetwork();
            tn.addStation('Start', [0, 0]);
            tn.addStation('End', [10, 0]);

            pts = [0 0; 5 3; 10 0];
            curve = OceanMath.Geometry.CatmullRomSpline(pts);
            tn.addRoute(1, 2, curve, 'Label', 'Route-A');

            g = tn.getGraph();
            w = g.getEdgeWeight(1, 2);
            testCase.assertEqual(w, curve.totalLength(), 'AbsTol', 1e-6);

            A = g.getAdjacencyMatrix();
            testCase.assertGreaterThan(A(1,2), 0);
        end

        % ============================================================== %
        %  Integration: GraphMapper bridge
        % ============================================================== %

        function test_graphmapper_bridge(testCase)
            g = OceanMath.Graph.DigraphCus();
            g.addNode('A', [0, 0]);
            g.addNode('B', [3, 0]);
            g.addNode('C', [3, 4]);
            g.addEdge(1, 2, 3);
            g.addEdge(2, 3, 4);

            positions = g.getNodePositions();
            A = g.getAdjacencyMatrix();
            mapper = OceanViz.Adapters.GraphMapper(positions, 'Adjacency', A);

            coords = mapper.mapState(2);
            testCase.assertEqual(coords, [3, 0]);
        end

        % ============================================================== %
        %  Remove Operations
        % ============================================================== %

        function test_remove_edge(testCase)
            g = OceanMath.Graph.DigraphCus();
            g.addNode('A', [0,0]);
            g.addNode('B', [1,0]);
            g.addEdge(1, 2, 5);
            testCase.assertTrue(g.hasEdge(1, 2));

            g.removeEdge(1, 2);
            testCase.assertFalse(g.hasEdge(1, 2));
            testCase.assertFalse(g.hasEdge(2, 1));
        end

        function test_remove_node(testCase)
            g = OceanMath.Graph.DigraphCus();
            g.addNode('A', [0,0]);
            g.addNode('B', [1,0]);
            g.addNode('C', [2,0]);
            g.addEdge(1, 2, 3);
            g.addEdge(2, 3, 4);

            g.removeNode(2);
            testCase.assertFalse(g.hasEdge(1, 2));
            testCase.assertFalse(g.hasEdge(2, 3));
        end

        % ============================================================== %
        %  NetworkAnalytics
        % ============================================================== %

        function test_is_connected(testCase)
            g = OceanMath.Graph.DigraphCus();
            g.addNode('A', [0,0]);
            g.addNode('B', [1,0]);
            g.addNode('C', [2,0]);
            g.addEdge(1, 2, 1);
            g.addEdge(2, 3, 1);

            testCase.assertTrue(...
                OceanMath.Algorithms.Graph.NetworkAnalytics.isConnected(g));
        end

        function test_not_connected(testCase)
            g = OceanMath.Graph.DigraphCus();
            g.addNode('A', [0,0]);
            g.addNode('B', [1,0]);
            % No edge
            testCase.assertFalse(...
                OceanMath.Algorithms.Graph.NetworkAnalytics.isConnected(g));
        end

        function test_connected_components(testCase)
            g = OceanMath.Graph.DigraphCus();
            g.addNode('A', [0,0]);
            g.addNode('B', [1,0]);
            g.addNode('C', [2,0]);
            g.addNode('D', [3,0]);
            g.addEdge(1, 2, 1);  % component 1: A-B
            g.addEdge(3, 4, 1);  % component 2: C-D

            labels = OceanMath.Algorithms.Graph.NetworkAnalytics.connectedComponents(g);
            testCase.assertEqual(labels(1), labels(2));
            testCase.assertEqual(labels(3), labels(4));
            testCase.assertNotEqual(labels(1), labels(3));
        end
    end
end
