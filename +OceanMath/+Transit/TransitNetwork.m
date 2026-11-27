classdef TransitNetwork < handle
    % TRANSITNETWORK  Domain wrapper around DigraphCus for transit systems.
    %
    % Adds transit vocabulary (stations, routes, splines, lines) on top of
    % a generic graph. The inner DigraphCus handles all graph algorithms.
    %
    % Usage:
    %   tn = OceanMath.Transit.TransitNetwork();
    %
    %   % Register lines
    %   tn.addLine('NSL', 'North-South Line', [0.85 0.15 0.15]);
    %   tn.addLine('EWL', 'East-West Line',   [0.15 0.65 0.15]);
    %
    %   % Add stations (shared, line-agnostic)
    %   tn.addStation('Orchard',     [3, 7]);
    %   tn.addStation('Dhoby Ghaut', [4, 6]);
    %   tn.addStation('City Hall',   [5, 5]);
    %
    %   % Build routes on lines
    %   tn.addRouteToLine('NSL', 1, 2, spline1);
    %   tn.addRouteToLine('EWL', 2, 3, spline2);
    %
    %   % Query
    %   tn.getInterchanges()          % → [2] (Dhoby Ghaut)
    %   tn.getStationLines(2)         % → {'NSL', 'EWL'}
    %   tn.getLine('NSL').getRoute()  % → [1, 2]
    %
    % See also: OceanMath.Transit.TransitLine, OceanMath.Graph.DigraphCus

    % ------------------------------------------------------------------ %
    properties (Access = private)
        Graph_          % OceanMath.Graph.DigraphCus
        Lines_          % containers.Map of code -> TransitLine
        Splines_        % cell array of geometry objects aligned with Edge IDs
        
        StationLineCache_ % cell array: cache{stationId} = {'L1', 'L2'}
        CacheIsValid_     (1,1) logical = false
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = TransitNetwork(options)
            % TRANSITNETWORK  Construct an empty transit network.
            arguments
                options.Directed (1,1) logical = false
            end
            obj.Graph_ = OceanMath.Graph.DigraphCus('Directed', options.Directed);
            obj.Splines_ = {};
            obj.Lines_ = containers.Map();
        end

        % ============================================================== %
        %  Station Operations (transit vocabulary)
        % ============================================================== %

        function reserveStations(obj, N)
            % RESERVESTATIONS  Pre-allocate storage for bulk-loading stations.
            obj.Graph_.reserveNodes(N);
        end

        function id = addStation(obj, name, position, metadata)
            % ADDSTATION  Add a station to the transit network.
            arguments
                obj
                name      (1,:) char
                position  (1,:) double = [0, 0]
                metadata  (1,1) struct = struct()
            end
            id = obj.Graph_.addNode(name, position, metadata);
        end

        function ids = addStations(obj, stationList)
            % ADDSTATIONS  Bulk-add stations from a cell array.
            %
            %   ids = tn.addStations({
            %       'Newton',      [5, 9];
            %       'Orchard',     [5, 7];
            %       'Dhoby Ghaut', [5, 5];
            %       'Marina Bay',  [5, 3]
            %   });
            %
            %   Each row: {name, [x, y]}. Returns vector of assigned IDs.
            arguments
                obj
                stationList (:,2) cell
            end
            nStations = size(stationList, 1);
            ids = zeros(1, nStations);
            for i = 1:nStations
                ids(i) = obj.addStation(stationList{i, 1}, stationList{i, 2});
            end
        end

        function station = getStation(obj, id)
            % GETSTATION  Retrieve a station (Node) by ID.
            station = obj.Graph_.getNode(id);
        end

        function station = getStationByName(obj, name)
            % GETSTATIONBYNAME  Find a station by name.
            station = obj.Graph_.getNodeByName(name);
        end

        function N = getStationCount(obj)
            N = obj.Graph_.getNodeCount();
        end

        function names = getStationNames(obj)
            names = obj.Graph_.getNodeNames();
        end

        function updateStationName(obj, id, newName)
            % UPDATESTATIONNAME  Rename a station.
            arguments
                obj
                id       (1,1) double {mustBePositive, mustBeInteger}
                newName  (1,:) char
            end
            obj.Graph_.updateNodeName(id, newName);
        end

        function updateStationPosition(obj, id, newPosition)
            % UPDATESTATIONPOSITION  Move a station to a new position.
            arguments
                obj
                id          (1,1) double {mustBePositive, mustBeInteger}
                newPosition (1,:) double
            end
            obj.Graph_.updateNodePosition(id, newPosition);
        end

        % ============================================================== %
        %  Line Management
        % ============================================================== %

        function line = addLine(obj, code, name, color)
            % ADDLINE  Register a new transit line.
            %
            %   tn.addLine('NSL', 'North-South Line', [0.85 0.15 0.15])
            arguments
                obj
                code  (1,:) char
                name  (1,:) char
                color (1,3) double
            end
            line = OceanMath.Transit.TransitLine(code, name, color);
            obj.Lines_(code) = line;
        end

        function line = getLine(obj, code)
            % GETLINE  Retrieve a TransitLine by its code.
            arguments
                obj
                code (1,:) char
            end
            if obj.Lines_.isKey(code)
                line = obj.Lines_(code);
            else
                error('TransitNetwork:NoSuchLine', ...
                    'Line "%s" not registered.', code);
            end
            obj.invalidateLineCache();
        end

        function codes = getLineNames(obj)
            % GETLINENAMES  Return all registered line codes.
            codes = obj.Lines_.keys();
        end

        function list = getLinesForStation(obj, stationId)
            % GETLINESFORSTATION  Return cell array of line codes serving this station.
            % Uses an O(1) internal cache to completely avoid O(L*S) loop bottlenecks.
            if ~obj.CacheIsValid_
                obj.rebuildLineCache_();
            end
            if stationId <= numel(obj.StationLineCache_)
                list = obj.StationLineCache_{stationId};
            else
                list = {};
            end
        end

        function invalidateLineCache(obj)
            % INVALIDATELINECACHE  Mark the internal station-to-line map as dirty.
            % Must be called anytime a station is added/removed from a TransitLine.
            obj.CacheIsValid_ = false;
        end

        function N = getLineCount(obj)
            % GETLINECOUNT  Number of registered lines.
            N = obj.Lines_.Count;
        end

        function line = buildLine(obj, code, name, color, stationIds, options)
            % BUILDLINE  One-call line builder: create + register + connect.
            %
            %   tn.buildLine('NSL', 'North-South Line', [0.85 0.15 0.15], [1 2 3 4])
            %   tn.buildLine('NSL', 'North-South Line', [.85 .15 .15], [1 2 3 4], ...
            %                'Splines', {s1, s2, s3})
            %
            %   Creates the line, registers all stations in order, and connects
            %   consecutive stations with routes. If 'Splines' not provided,
            %   auto-generates straight Bezier curves between stations.
            %
            %   Requires: stations must already exist (via addStation).
            arguments
                obj
                code       (1,:) char
                name       (1,:) char
                color      (1,3) double
                stationIds (1,:) double {mustBePositive, mustBeInteger}
                options.Splines cell = {}
            end

            if numel(stationIds) < 2
                error('TransitNetwork:TooFewStations', ...
                    'buildLine requires at least 2 stations.');
            end

            % Create line
            line = obj.addLine(code, name, color);

            nSegs = numel(stationIds) - 1;
            hasCustomSplines = ~isempty(options.Splines);
            if hasCustomSplines && numel(options.Splines) ~= nSegs
                error('TransitNetwork:SplineMismatch', ...
                    'Expected %d splines, got %d.', nSegs, numel(options.Splines));
            end

            % Connect consecutive stations
            for i = 1:nSegs
                src = stationIds(i);
                tgt = stationIds(i+1);

                if hasCustomSplines
                    spl = options.Splines{i};
                else
                    % Auto-generate straight Bezier between station positions
                    pSrc = obj.getStation(src).Position;
                    pTgt = obj.getStation(tgt).Position;
                    spl = OceanMath.Geometry.BezierCurve([ ...
                        pSrc; ...
                        pSrc + (pTgt - pSrc) * 1/3; ...
                        pSrc + (pTgt - pSrc) * 2/3; ...
                        pTgt]);
                end

                obj.addRouteToLine(code, src, tgt, spl);
            end
        end

        function insertStationOnLine(obj, lineCode, newStationId, afterStationId)
            % INSERTSTATIONONLINE  Insert a station between two consecutive stops.
            %
            %   tn.insertStationOnLine('NSL', 99, 2)
            %     Inserts station 99 AFTER station 2 on the NSL line.
            %
            %   Atomically:
            %     1. Removes old edge (afterStation -> nextStation)
            %     2. Inserts new station into route
            %     3. Creates two new edges with auto-generated splines
            arguments
                obj
                lineCode       (1,:) char
                newStationId   (1,1) double {mustBePositive, mustBeInteger}
                afterStationId (1,1) double {mustBePositive, mustBeInteger}
            end

            line = obj.getLine(lineCode);
            route = line.getRoute();

            % Find afterStationId in the route
            idx = find(route == afterStationId, 1);
            if isempty(idx)
                error('TransitNetwork:StationNotOnLine', ...
                    'Station %d is not on line "%s".', afterStationId, lineCode);
            end
            if idx >= numel(route)
                error('TransitNetwork:NoNextStation', ...
                    'Station %d is the last stop. Use registerStation instead.', ...
                    afterStationId);
            end

            nextStationId = route(idx + 1);

            % Remove old edge and its spline
            obj.removeRoute_(afterStationId, nextStationId);

            % Insert into route ordering (after idx)
            line.insertStation(idx + 1, newStationId);

            % Create two new edges with auto-generated splines
            pAfter = obj.getStation(afterStationId).Position;
            pNew   = obj.getStation(newStationId).Position;
            pNext  = obj.getStation(nextStationId).Position;

            spl1 = obj.makeAutoSpline_(pAfter, pNew);
            spl2 = obj.makeAutoSpline_(pNew, pNext);

            obj.addRouteToLine(lineCode, afterStationId, newStationId, spl1);
            obj.addRouteToLine(lineCode, newStationId, nextStationId, spl2);
        end

        % ============================================================== %
        %  Route Operations (transit vocabulary + spline awareness)
        % ============================================================== %

        function edgeIdx = addRoute(obj, srcId, tgtId, spline, options)
            % ADDROUTE  Connect two stations with a spline-shaped route.
            %   Weight is auto-computed from the spline's total arc length.
            %
            %   tn.addRoute(1, 2, mySpline)
            %   tn.addRoute(1, 2, mySpline, 'Label', 'NSL')
            arguments
                obj
                srcId   (1,1) double {mustBePositive, mustBeInteger}
                tgtId   (1,1) double {mustBePositive, mustBeInteger}
                spline
                options.Label (1,:) char = ''
            end

            % Compute weight from spline arc length
            weight = spline.totalLength();

            % Add edge to the generic graph
            edgeIdx = obj.Graph_.addEdge(srcId, tgtId, weight, ...
                'Label', options.Label);

            % Store spline
            obj.Splines_{edgeIdx} = spline;
        end

        function edgeIdx = addRouteToLine(obj, lineCode, srcId, tgtId, spline)
            % ADDROUTETOLINE  Add a spline route AND register it on a line.
            %
            %   tn.addRouteToLine('NSL', 1, 2, mySpline)
            %
            %   This does three things:
            %     1. Adds the edge to DigraphCus (weight from arc length)
            %     2. Stores the spline for rendering
            %     3. Registers both stations on the TransitLine
            arguments
                obj
                lineCode (1,:) char
                srcId    (1,1) double {mustBePositive, mustBeInteger}
                tgtId    (1,1) double {mustBePositive, mustBeInteger}
                spline
            end

            % Add route with line code as label
            edgeIdx = obj.addRoute(srcId, tgtId, spline, 'Label', lineCode);

            % Register stations on the line (in order)
            line = obj.getLine(lineCode);
            line.registerStation(srcId);
            line.registerStation(tgtId);
        end

        function edgeIdx = addSimpleRoute(obj, srcId, tgtId, weight, options)
            % ADDSIMPLEROUTE  Connect two stations without a spline.
            arguments
                obj
                srcId   (1,1) double {mustBePositive, mustBeInteger}
                tgtId   (1,1) double {mustBePositive, mustBeInteger}
                weight  (1,1) double {mustBePositive}
                options.Label (1,:) char = ''
            end
            edgeIdx = obj.Graph_.addEdge(srcId, tgtId, weight, ...
                'Label', options.Label);
        end

        function spline = getSpline(obj, edgeIdx)
            % GETSPLINE  Return the IParametricCurve for an edge.
            if edgeIdx <= numel(obj.Splines_) && ~isempty(obj.Splines_{edgeIdx})
                spline = obj.Splines_{edgeIdx};
            else
                spline = [];
            end
        end

        function updateRouteSpline(obj, edgeIdx, spline)
            % UPDATEROUTESPLINE  Replace a route's spline and update weight.
            obj.Splines_{edgeIdx} = spline;
            newWeight = spline.totalLength();
            obj.Graph_.updateEdgeWeight(edgeIdx, newWeight);
        end

        function E = getRouteCount(obj)
            E = obj.Graph_.getEdgeCount();
        end

        function edge = getRouteEdge(obj, edgeIdx)
            % GETROUTEEDGE  Retrieve the Edge object for a route by index.
            %   Returns empty [] if the edge has been removed.
            edge = obj.Graph_.getEdge(edgeIdx);
        end

        function removeRoute(obj, srcId, tgtId)
            % REMOVEROUTE  Remove a route (edge + spline) between two stations.
            %
            %   tn.removeRoute(1, 2)
            arguments
                obj
                srcId (1,1) double {mustBePositive, mustBeInteger}
                tgtId (1,1) double {mustBePositive, mustBeInteger}
            end
            obj.removeRoute_(srcId, tgtId);
        end

        function idx = findEdgeIndex(obj, srcId, tgtId)
            % FINDEDGEINDEX  Return the edge index for the (src, tgt) pair.
            %   O(1) dictionary lookup via DigraphCus. Returns 0 if none exists.
            arguments
                obj
                srcId (1,1) double {mustBePositive, mustBeInteger}
                tgtId (1,1) double {mustBePositive, mustBeInteger}
            end
            idx = obj.Graph_.getEdgeId(srcId, tgtId);
        end

        % ============================================================== %
        %  Passthrough to Generic Graph DigraphCus
        % ============================================================== %

        function g = getGraph(obj)
            % GETGRAPH  Return the inner DigraphCus for algorithms.
            g = obj.Graph_;
        end

        function [path, dist] = shortestPath(obj, srcId, tgtId)
            % SHORTESTPATH  Shortest path via Dijkstra.
            [path, dist] = OceanMath.Algorithms.Graph.Dijkstra.solve( ...
                obj.Graph_.getAdjacencyMatrix(), srcId, tgtId);
        end

        function A = getAdjacencyMatrix(obj)
            A = obj.Graph_.getAdjacencyMatrix();
        end

        function positions = getStationPositions(obj)
            positions = obj.Graph_.getNodePositions();
        end

        % ============================================================== %
        %  Interchange & Line Queries
        % ============================================================== %

        function ids = getInterchanges(obj)
            % GETINTERCHANGES  Station IDs that belong to 2+ lines.
            %
            %   ids = tn.getInterchanges()
            stationLineCounts = zeros(1, obj.getStationCount());
            codes = obj.Lines_.keys();
            for i = 1:numel(codes)
                line = obj.Lines_(codes{i});
                route = line.getRoute();
                for j = 1:numel(route)
                    stationLineCounts(route(j)) = stationLineCounts(route(j)) + 1;
                end
            end
            ids = find(stationLineCounts >= 2);
        end

        function codes = getStationLines(obj, stationId)
            % GETSTATIONLINES  Line codes passing through a station.
            %
            %   codes = tn.getStationLines(2)  % → {'NSL', 'EWL'}
            arguments
                obj
                stationId (1,1) double {mustBePositive, mustBeInteger}
            end
            allCodes = obj.Lines_.keys();
            codes = {};
            for i = 1:numel(allCodes)
                line = obj.Lines_(allCodes{i});
                if line.hasStation(stationId)
                    codes{end+1} = allCodes{i}; %#ok<AGROW>
                end
            end
        end

        function tf = isInterchange(obj, stationId)
            % ISINTERCHANGE  True if station belongs to 2+ lines.
            tf = numel(obj.getStationLines(stationId)) >= 2;
        end

        % ============================================================== %
        %  Transit-Specific Analytics
        % ============================================================== %

        function L = totalNetworkLength(obj)
            L = OceanMath.Algorithms.Graph.NetworkAnalytics.totalLength(obj.Graph_);
        end

        function avg = averageRouteLength(obj)
            avg = OceanMath.Algorithms.Graph.NetworkAnalytics.averageEdgeLength(obj.Graph_);
        end

        % ============================================================== %
        %  Route Reconciliation
        % ============================================================== %

        function reconcileLineRoutes(obj)
            % RECONCILELINEROUTES  Reorder each line's route to match actual edge topology.
            %   The user may have registered stations in arbitrary click order.
            %   This method walks the actual graph edges to produce the correct
            %   physical traversal sequence for each line.
            %
            %   tn.reconcileLineRoutes()
            %
            %   Algorithm:
            %     1. For each line, build local adjacency among registered stations
            %     2. Find degree-1 nodes (line terminals) in the local subgraph
            %     3. Walk from one terminal, collecting stations in connected order
            %     4. Circular lines (no degree-1): start from arbitrary station
            %
            %   Called automatically by DesignerSerializer.exportJSON().

            codes = obj.Lines_.keys();
            g = obj.Graph_;

            for i = 1:numel(codes)
                line = obj.Lines_(codes{i});
                S = line.getRoute();  % current (possibly unordered) station set
                %S =[38, 154, 50];

                n = numel(S);
                if n < 2, continue; end

                % Build local adjacency within S
                % adj{k} = [indices into S that are neighbors of S(k)]
                sSet = false(1, max(S));        % fast lookup: is station in S?
                sSet(S) = true;

                localAdj = cell(1, n);
                for a = 1:n
                    localAdj{a} = []; % {[], [], ...}
                end

                for a = 1:n
                    graphNeighbors = g.neighbors(S(a)); %Get the arary of stations(Maybe belongs to different lines) that connect to this node
                    for nb = 1:numel(graphNeighbors) %Loop through the list of neighbors
                        nId = graphNeighbors(nb); %Get the actual id of this neighbor
                        if nId <= numel(sSet) && sSet(nId) %Check if this neighbor is in the current line and valid
                            % Find index of nId in S
                            idx = find(S == nId, 1); %Find the index of this neighbor in S
                            if ~isempty(idx)
                                localAdj{a}(end+1) = idx; %fill the local adjacency list
                            end
                        end
                    end
                end

                % Find degree-1 stations within the local subgraph (terminals)
                localDeg = cellfun(@numel, localAdj);
                terminals = find(localDeg == 1);

                if isempty(terminals)
                    % Circular line: start from first registered station
                    startIdx = 1;
                else
                    startIdx = terminals(1);
                end

                % Walk the chain
                visited = false(1, n);
                ordered = zeros(1, n);
                current = startIdx; %Control by the Graph (Actual Id)
                for step = 1:n
                    ordered(step) = S(current); %current = position in S, S(current): actual Id
                    visited(current) = true; %Mark as visited
                    nextIdx = 0;
                    for nb = localAdj{current}
                        if ~visited(nb)
                            nextIdx = nb;
                            break; %Save time. 
                        end
                    end
                    if nextIdx == 0, break; end
                    current = nextIdx;
                end

                % Use the connected chain, warn about orphaned stations
                reachable = ordered(1:step);
                if step < n %visit fewer stations than registered
                    orphanIds = S(~visited); %Get the stations that were not visited
                    orphanNames = cell(1, numel(orphanIds));
                    for oi = 1:numel(orphanIds)
                        st = obj.getStation(orphanIds(oi));
                        if ~isempty(st)
                            orphanNames{oi} = st.Name;
                        else
                            orphanNames{oi} = sprintf('#%d', orphanIds(oi));
                        end
                    end
                    fprintf('[Reconcile] Line "%s": %d/%d stations connected. Orphaned: %s\n', ...
                        codes{i}, step, n, strjoin(orphanNames, ', '));
                end
                line.setRoute(reachable);
            end
        end

        % ============================================================== %
        %  Visualization Export
        % ============================================================== %

        function data = toVizData(obj)
            % TOVIZDATA  Export struct for TransitGraphStrategy.
            %
            %   data = tn.toVizData();
            data.Positions  = obj.getStationPositions();
            data.NodeNames  = obj.getStationNames();
            data.Splines    = obj.Splines_;
            data.EdgePairs  = obj.getEdgePairs_();
            data.EdgeLabels = obj.getEdgeLabels_();

            % Line information
            codes = obj.Lines_.keys();
            lineInfo = struct();
            for i = 1:numel(codes)
                line = obj.Lines_(codes{i});
                s = line.toStruct();
                % Use valid fieldname (replace spaces/dashes)
                fieldName = matlab.lang.makeValidName(codes{i});
                lineInfo.(fieldName) = s;
            end
            data.Lines = lineInfo;
            data.InterchangeIds = obj.getInterchanges();
            
            % Terminal stations: degree-1 nodes (leaf nodes in the graph)
            W = obj.Graph_.getAdjacencyMatrix();
            V = obj.Graph_.getNodeCount();
            degrees = sum(W(1:V, 1:V) > 0, 2);  % row sums = degree per node
            data.TerminalIds = find(degrees == 1)';
        end

        % ============================================================== %
        %  Serialization
        % ============================================================== %

        function s = toStruct(obj)
            % TOSTRUCT  Serialize the network state to a struct.
            arguments
                obj
            end
            s = obj.Graph_.toStruct(); %Call DigraphCus toStruct method, return the full graph topology and weights.
            
            % Compact SplineData strictly matching the compacted Edges array
            splineData = {};
            for i = 1:obj.Graph_.getEdgeCount()
                if ~isempty(obj.Graph_.getEdge(i))
                    if i <= numel(obj.Splines_) && ~isempty(obj.Splines_{i})
                        splineData{end+1} = obj.Splines_{i}.getControlPoints(); %#ok<AGROW>
                    else
                        splineData{end+1} = []; %#ok<AGROW>
                    end
                end
            end
            s.SplineData = splineData;

            % Serialize lines
            codes = obj.Lines_.keys();
            lineData = cell(1, numel(codes));
            for i = 1:numel(codes)
                lineData{i} = obj.Lines_(codes{i}).toStruct(); %Everything needed to reconstruct the line
            end
            s.LineData = lineData;
        end

        % ============================================================== %
        %  Display
        % ============================================================== %

        function describe(obj)
            % DESCRIBE Prints summary of TransitNetwork
            arguments
                obj
            end
            fprintf('TransitNetwork:\n');
            fprintf('  Stations: %d\n', obj.getStationCount());
            fprintf('  Routes:   %d\n', obj.getRouteCount());
            fprintf('  Lines:    %d\n', obj.getLineCount());
            if obj.getRouteCount() > 0
                fprintf('  Total Length: %.4f\n', obj.totalNetworkLength());
                fprintf('  Avg Route:   %.4f\n', obj.averageRouteLength());
            end

            % List lines with station names
            allNames = obj.getStationNames();
            codes = obj.Lines_.keys();
            for i = 1:numel(codes)
                line = obj.Lines_(codes{i});
                route = line.getRoute();
                routeNames = allNames(route);
                fprintf('  [%s] %s (%d stations)\n', line.Code, line.Name, ...
                    line.getStationCount());
                fprintf('    Route: %s\n', strjoin(routeNames, ' -> '));
                % Policies
                p = line.getAllPolicies();
                fields = fieldnames(p);
                for k = 1:numel(fields)
                    val = p.(fields{k});
                    if isnumeric(val)
                        fprintf('    %s: %g\n', fields{k}, val);
                    else
                        fprintf('    %s: %s\n', fields{k}, string(val));
                    end
                end
            end

            % Interchanges
            ixIds = obj.getInterchanges();
            if ~isempty(ixIds)
                ixNames = allNames(ixIds);
                fprintf('  Interchanges: %s\n', strjoin(ixNames, ', '));
            end
        end
    end

    % ================================================================== %
    methods (Access = private)
        
        function rebuildLineCache_(obj)
            % REBUILDLINECACHE_  One-time O(L*S) scan to power O(1) lookups.
            maxId = obj.Graph_.getNodeCount();
            
            cache = cell(1, maxId);
            for i = 1:maxId
                cache{i} = {};
            end
            
            codes = obj.getLineNames();
            for i = 1:numel(codes)
                c = codes{i};
                route = obj.Lines_(c).getRoute();
                for j = 1:numel(route)
                    sId = route(j);
                    if sId <= numel(cache)
                        % Append carefully to avoid duplicates
                        if ~any(strcmp(cache{sId}, c))
                            cache{sId}{end+1} = c;
                        end
                    end
                end
            end
            obj.StationLineCache_ = cache;
            obj.CacheIsValid_ = true;
        end

        function pairs = getEdgePairs_(obj)
            % GETEDGEPAIRS_  Return [E x 2] matrix of [source, target].
            E = obj.Graph_.getEdgeCount();
            pairs = zeros(E, 2);
            for i = 1:E
                e = obj.Graph_.getEdge(i);
                if ~isempty(e)
                    pairs(i, :) = [e.Source, e.Target];
                end
            end
        end

        function labels = getEdgeLabels_(obj)
            % GETEDGELABELS_  Return {1 x E} cell of edge labels.
            E = obj.Graph_.getEdgeCount();
            labels = cell(1, E);
            for i = 1:E
                e = obj.Graph_.getEdge(i);
                if ~isempty(e)
                    labels{i} = e.Label;
                else
                    labels{i} = '';
                end
            end
        end

        function removeRoute_(obj, srcId, tgtId)
            % REMOVEROUTE_  Remove an edge and its associated spline.
            % Find and clear the spline for this edge
            for i = 1:numel(obj.Splines_)
                if i <= obj.Graph_.getEdgeCount()
                    e = obj.Graph_.getEdge(i);
                    if ~isempty(e) && e.Source == srcId && e.Target == tgtId
                        obj.Splines_{i} = [];
                        break;
                    end
                end
            end
            % Remove from graph
            obj.Graph_.removeEdge(srcId, tgtId);
        end

        function spl = makeAutoSpline_(~, pSrc, pTgt)
            % MAKEAUTOSPLINE_  Generate a straight Bezier between two positions.
            spl = OceanMath.Geometry.BezierCurve([ ...
                pSrc; ...
                pSrc + (pTgt - pSrc) * 1/3; ...
                pSrc + (pTgt - pSrc) * 2/3; ...
                pTgt]);
        end
    end
end
