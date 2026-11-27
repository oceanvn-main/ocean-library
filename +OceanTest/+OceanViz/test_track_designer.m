classdef test_track_designer < OceanTest.BaseTestCase
    % TEST_TRACK_DESIGNER  Unit tests for TrackDesigner (programmatic API, no GUI).

    methods (Test)

        % ============================================================== %
        %  Basic Station Management
        % ============================================================== %

        function test_add_stations(testCase)
            td = OceanViz.Designer.TrackDesigner();
            id1 = td.addStation('Orchard', [3, 7]);
            id2 = td.addStation('Dhoby Ghaut', [4, 6]);

            g = td.exportGraph();
            testCase.assertEqual(g.getNodeCount(), 2);
            testCase.assertEqual(id1, 1);
            testCase.assertEqual(id2, 2);
        end

        function test_station_positions_in_graph(testCase)
            td = OceanViz.Designer.TrackDesigner();
            td.addStation('A', [0, 0]);
            td.addStation('B', [10, 0]);

            g = td.exportGraph();
            pos = g.getNodePositions();
            testCase.assertEqual(pos, [0 0; 10 0]);
        end

        % ============================================================== %
        %  Connecting Stations
        % ============================================================== %

        function test_connect_creates_edge(testCase)
            td = OceanViz.Designer.TrackDesigner();
            td.addStation('A', [0, 0]);
            td.addStation('B', [10, 0]);
            td.connectStations(1, 2);

            g = td.exportGraph();
            testCase.assertEqual(g.getEdgeCount(), 1);
            testCase.assertTrue(g.hasEdge(1, 2));
        end

        function test_edge_weight_is_spline_length(testCase)
            td = OceanViz.Designer.TrackDesigner();
            td.addStation('A', [0, 0]);
            td.addStation('B', [10, 0]);
            td.connectStations(1, 2);

            g = td.exportGraph();
            w = g.getEdgeWeight(1, 2);
            testCase.assertGreaterThan(w, 0, ...
                'Edge weight should be positive (from spline arc length).');
        end

        function test_connect_with_label(testCase)
            td = OceanViz.Designer.TrackDesigner();
            td.addStation('A', [0, 0]);
            td.addStation('B', [5, 5]);
            td.connectStations(1, 2, 'Label', 'North-South Line');

            g = td.exportGraph();
            e = g.getEdge(1);
            testCase.assertEqual(e.Label, 'North-South Line');
        end

        % ============================================================== %
        %  Multi-Edge Network
        % ============================================================== %

        function test_triangle_network(testCase)
            td = OceanViz.Designer.TrackDesigner();
            td.addStation('A', [0, 0]);
            td.addStation('B', [10, 0]);
            td.addStation('C', [5, 8]);
            td.connectStations(1, 2);
            td.connectStations(2, 3);
            td.connectStations(1, 3);

            g = td.exportGraph();
            testCase.assertEqual(g.getEdgeCount(), 3);

            % Shortest path should work
            [path, dist] = OceanMath.Algorithms.Graph.Dijkstra.solve(g.getAdjacencyMatrix(), 1, 3);
            testCase.assertFalse(isempty(path));
            testCase.assertGreaterThan(dist, 0);
        end

        % ============================================================== %
        %  Shortest Path with Spline Weights
        % ============================================================== %

        function test_shortest_path_uses_arc_length(testCase)
            %  A ---straight--- B ---straight--- C
            %  |                                  |
            %  +----------- curved ---------------+
            td = OceanViz.Designer.TrackDesigner();
            td.addStation('A', [0, 0]);
            td.addStation('B', [5, 0]);
            td.addStation('C', [10, 0]);

            % A-B and B-C are short straight edges
            td.connectStations(1, 2, 'NumMidpoints', 0);
            td.connectStations(2, 3, 'NumMidpoints', 0);

            % A-C is a long curved detour
            td.connectStations(1, 3, 'NumMidpoints', 3);

            g = td.exportGraph();
            [path, ~] = OceanMath.Algorithms.Graph.Dijkstra.solve(g.getAdjacencyMatrix(), 1, 3);

            % Via B should be shorter than the curved A-C
            testCase.assertEqual(path, [1, 2, 3], ...
                'Shortest path should go via B, not the curved A-C.');
        end

        % ============================================================== %
        %  GraphMapper Bridge
        % ============================================================== %

        function test_graph_mapper_bridge(testCase)
            td = OceanViz.Designer.TrackDesigner();
            td.addStation('A', [0, 0]);
            td.addStation('B', [5, 3]);
            td.connectStations(1, 2);

            g = td.exportGraph();
            mapper = OceanViz.Adapters.GraphMapper( ...
                g.getNodePositions(), 'Adjacency', g.getAdjacencyMatrix());

            coords = mapper.mapState(2);
            testCase.assertEqual(coords, [5, 3]);
        end

        % ============================================================== %
        %  Save / Load
        % ============================================================== %

        function test_save_and_load(testCase)
            td = OceanViz.Designer.TrackDesigner();
            td.addStation('X', [1, 2]);
            td.addStation('Y', [3, 4]);
            td.connectStations(1, 2);

            tmpFile = fullfile(tempdir, 'test_track_save.mat');
            td.saveDesign(tmpFile);

            % Load into a new designer
            td2 = OceanViz.Designer.TrackDesigner();
            td2.loadDesign(tmpFile);

            g2 = td2.exportGraph();
            testCase.assertEqual(g2.getNodeCount(), 2);
            testCase.assertEqual(g2.getEdgeCount(), 1);
            testCase.assertTrue(g2.hasEdge(1, 2));

            % Clean up
            delete(tmpFile);
        end
    end
end
