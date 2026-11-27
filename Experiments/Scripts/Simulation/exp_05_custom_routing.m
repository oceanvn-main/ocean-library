% EXP_05_CUSTOM_ROUTING  Simulate Shortest Path Agent on a Custom Interactive Map.
%
%   Loads a custom JSON map built by the TrackDesigner GUI, runs Dijkstra's 
%   Shortest Path Algorithm across the mathematically strict splines, and 
%   renders the agent's decision using the OceanViz engine.
%
%   Prerequisite: Must have exported a map from TrackDesigner to:
%   Experiments/Scripts/Test/testinteractive_01.json

fprintf('\n=== CUSTOM MAP DIJKSTRA SIMULATION ===\n\n');

% Prevent base workspace pollution from shadowing built-in functions
clear text data jsonText pathIndices;

%% 1. Load the Custom Map mathematically
baseDir = fileparts(fileparts(mfilename('fullpath'))); % Goes up from \Simulation to \Scripts
jsonFile = fullfile(baseDir, 'Test', 'test_case_1.json');
if exist(jsonFile, 'file') ~= 2
    error('Could not find %s. Please launch TrackDesigner, draw a map, and hit Export!', jsonFile);
end

% Read JSON directly — we need both the TransitNetwork math AND the raw JSON for rendering
jsonText = fileread(jsonFile);
data = jsondecode(jsonText);

% Build the TransitNetwork for Dijkstra from the GUI map exporter
tn = OceanViz.Designer.TrackDesigner.loadNetwork(jsonFile);
nNodes = tn.getStationCount();
fprintf('Loaded Custom Map: %d stations, %d physical routes.\n', ...
    nNodes, tn.getRouteCount());

if nNodes < 2
    error('Your custom map needs at least 2 stations to run a routing agent!');
end

%% 2. Run Dijkstra's Shortest Path Algorithm
srcId = 1;
tgtId = nNodes;

fprintf('Agent calculating shortest path from Station %d to Station %d...\n', srcId, tgtId);
[pathIndices, totalLength] = tn.shortestPath(srcId, tgtId);

if isempty(pathIndices)
    fprintf('No path found! The network might be disconnected.\n');
    return;
end
fprintf('Success! Path found traversing %d stations with a total physical length of %.2f units.\n', ...
    numel(pathIndices), totalLength);

%% 3. Render the Map Directly from JSON (bypassing TransitGraphStrategy)
viz = OceanViz.Core('Theme', 'Dark');
fig = figure('Name', 'Custom Map Routing Agent', 'Color', viz.Colors.Background, ...
    'Position', [100 100 1000 700]);
ax = axes('Parent', fig);
hold(ax, 'on');
title(ax, 'Dijkstra Shortest Path on Custom Map', ...
    'Color', viz.Colors.Text, 'FontSize', 14, 'FontWeight', 'bold');

% --- Draw ALL curves straight from the Math Engine ---
for i = 1:tn.getRouteCount()
    e = tn.getRouteEdge(i);
    if isempty(e), continue; end
    
    spline = tn.getSpline(i);
    if ~isempty(spline)
        tVals = spline.parameterize(100);
        pts = spline.evaluate(tVals);
        plot(ax, pts(:,1), pts(:,2), '-', 'Color', [0.3 0.6 0.8 0.4], ...
            'LineWidth', 2.5, 'HandleVisibility', 'off');
    else
        p1 = tn.getStation(e.Source).Position;
        p2 = tn.getStation(e.Target).Position;
        plot(ax, [p1(1), p2(1)], [p1(2), p2(2)], '-', 'Color', [0.3 0.6 0.8 0.4], ...
            'LineWidth', 2.5, 'HandleVisibility', 'off');
    end
end

% --- Draw station markers ---
nStats = tn.getStationCount();
stationNames = cell(1, nStats);
for i = 1:nStats
    st = tn.getStation(i);
    if isempty(st), continue; end
    pos = st.Position(:)';
    stationNames{i} = st.Name;
    
    plot(ax, pos(1), pos(2), 'o', ...
        'MarkerSize', 10, ...
        'MarkerFaceColor', viz.Colors.Text, ...
        'MarkerEdgeColor', viz.Colors.Background, ...
        'LineWidth', 1.5, 'HandleVisibility', 'off');
    
    text(ax, pos(1), pos(2) + 1.5, st.Name, ...
        'Color', viz.Colors.Text, 'FontSize', 9, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', 'Interpreter', 'none');
end

% --- Draw the Agent's Shortest Path Decision with THICK RED curves ---
for i = 1:numel(pathIndices)-1
    u = pathIndices(i);
    v = pathIndices(i+1);
    
    % Find the matching edge directly from the mathematically verified Math Engine
    found = false;
    for j = 1:tn.getRouteCount()
        e = tn.getRouteEdge(j);
        if isempty(e), continue; end
        
        if (e.Source == u && e.Target == v) || (e.Source == v && e.Target == u)
            spline = tn.getSpline(j);
            if ~isempty(spline)
                tVals = spline.parameterize(100);
                pts = spline.evaluate(tVals);
                
                % Thick red overlay for the shortest path
                plot(ax, pts(:,1), pts(:,2), '-', ...
                    'Color', [1.0, 0.2, 0.2, 0.8], 'LineWidth', 5, ...
                    'HandleVisibility', 'off');
                found = true;
                break;
            end
        end
    end
    
    if ~found
        % Fallback: straight line
        p1 = tn.getStation(u).Position;
        p2 = tn.getStation(v).Position;
        plot(ax, [p1(1), p2(1)], [p1(2), p2(2)], '-', ...
            'Color', [1.0, 0.2, 0.2, 0.8], 'LineWidth', 5, ...
            'HandleVisibility', 'off');
    end
end

% Mark the shortest path stops with Star markers
for i = 1:numel(pathIndices)
    sid = pathIndices(i);
    st = tn.getStation(sid);
    if isempty(st), continue; end
    pos = st.Position(:)';
    
    plot(ax, pos(1), pos(2), 'p', ...
        'Color', [1.0, 0.8, 0.1], ...
        'MarkerFaceColor', [1.0, 0.2, 0.2], ...
        'MarkerSize', 18, 'HandleVisibility', 'off');
end

% --- Axes limits ---
xlim(ax, [-50, 50]);
ylim(ax, [-50, 50]);
axis(ax, 'equal');
ax.XColor = 'none'; ax.YColor = 'none';

% Legend
legend(ax, 'off');
text(ax, 0.02, 0.97, sprintf('Path: %s', strjoin(stationNames(pathIndices), ' → ')), ...
    'Units', 'normalized', 'Color', [1.0, 0.5, 0.5], ...
    'FontSize', 10, 'FontWeight', 'bold', 'VerticalAlignment', 'top');
text(ax, 0.02, 0.93, sprintf('Total Length: %.2f units', totalLength), ...
    'Units', 'normalized', 'Color', viz.Colors.Text, ...
    'FontSize', 9, 'VerticalAlignment', 'top');

fprintf('\n=== DEMO COMPLETE ===\n');
