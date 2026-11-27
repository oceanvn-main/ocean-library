% EXP_MRT_DEMO  Mini MRT network visualization demo.
%
%   Demonstrates the enriched TransitNetwork with TransitLine managers:
%     - 3 transit lines (NSL, EWL, DTL) with distinct colors
%     - 7 stations with Bezier curve routes
%     - Automatic interchange detection
%     - Shortest path highlighted
%     - Line policies (speed, frequency)
%
%   Uses: TransitNetwork, TransitLine, BezierCurve, GraphStyle, TransitGraphStrategy

fprintf('\n=== MRT NETWORK VISUALIZATION DEMO ===\n\n');

%% 1. Build the transit network
tn = OceanMath.Transit.TransitNetwork();

% Register lines (each line is a manager with its own color + policies)
tn.addLine('NSL', 'North-South Line', [0.85, 0.15, 0.15]);
tn.addLine('EWL', 'East-West Line',   [0.15, 0.65, 0.15]);
tn.addLine('DTL', 'Downtown Line',    [0.15, 0.45, 0.85]);

% Set policies
tn.getLine('NSL').setPolicy('SpeedLimit', 80);
tn.getLine('NSL').setPolicy('Frequency', 3);
tn.getLine('EWL').setPolicy('SpeedLimit', 90);
tn.getLine('EWL').setPolicy('Frequency', 4);
tn.getLine('DTL').setPolicy('SpeedLimit', 100);
tn.getLine('DTL').setPolicy('Frequency', 2);

% Schematic MRT layout — each line has 4 stations:
%
%   y=9:                    Newton [5,9]
%   y=8:  Botanic [1,8]
%   y=7:                    Orchard [5,7]               Bugis [9,7]
%   y=5:  Jurong [1,5] = Clementi [3,5] = Dhoby Ghaut [5,5] ====== City Hall [9,5]
%   y=3:                    Marina Bay [5,3]
%   y=1:                                                Bayfront [9,1]

tn.addStation('Jurong East',     [1, 5]);      % id=1   EWL
tn.addStation('Orchard',         [5, 7]);      % id=2   NSL
tn.addStation('Dhoby Ghaut',     [5, 5]);      % id=3   interchange hub
tn.addStation('City Hall',       [9, 5]);      % id=4   EWL
tn.addStation('Marina Bay',      [5, 3]);      % id=5   NSL
tn.addStation('Bugis',           [9, 7]);      % id=6   DTL
tn.addStation('Botanic Gardens', [1, 8]);      % id=7   DTL
tn.addStation('Newton',          [5, 9]);      % id=8   NSL (north end)
tn.addStation('Clementi',        [3, 5]);      % id=9   EWL (between Jurong & DG)
tn.addStation('Bayfront',        [9, 1]);      % id=10  DTL (south-east end)

%% 2. Create curved routes — 4 stations per line

% NSL (red): Newton → Orchard → Dhoby Ghaut → Marina Bay
%   Vertical corridor at x=5
r1 = OceanMath.Geometry.BezierCurve([5 9; 5 8.3; 5 7.7; 5 7]);
tn.addRouteToLine('NSL', 8, 2, r1);

r2 = OceanMath.Geometry.BezierCurve([5 7; 5 6.3; 5 5.7; 5 5]);
tn.addRouteToLine('NSL', 2, 3, r2);

r3 = OceanMath.Geometry.BezierCurve([5 5; 5 4.3; 5 3.7; 5 3]);
tn.addRouteToLine('NSL', 3, 5, r3);

% EWL (green): Jurong East → Clementi → Dhoby Ghaut → City Hall
%   Horizontal corridor at y=5
r4 = OceanMath.Geometry.BezierCurve([1 5; 1.7 5; 2.3 5; 3 5]);
tn.addRouteToLine('EWL', 1, 9, r4);

r5 = OceanMath.Geometry.BezierCurve([3 5; 3.7 5; 4.3 5; 5 5]);
tn.addRouteToLine('EWL', 9, 3, r5);

r6 = OceanMath.Geometry.BezierCurve([5 5; 6.3 5; 7.7 5; 9 5]);
tn.addRouteToLine('EWL', 3, 4, r6);

% DTL (blue): Botanic Gardens → Dhoby Ghaut → Bugis → Bayfront
%   Diagonal NW → center → NE, then drops south to Bayfront
r7 = OceanMath.Geometry.BezierCurve([1 8; 2 7.2; 3.5 6; 5 5]);
tn.addRouteToLine('DTL', 7, 3, r7);

r8 = OceanMath.Geometry.BezierCurve([5 5; 6.5 5.5; 7.5 6.2; 9 7]);
tn.addRouteToLine('DTL', 3, 6, r8);

r9 = OceanMath.Geometry.BezierCurve([9 7; 9 5.5; 9 3; 9 1]);
tn.addRouteToLine('DTL', 6, 10, r9);

%% 3. Query the network
fprintf('--- Network Info ---\n');
tn.describe();
fprintf('\n');

% Interchanges
ixIds = tn.getInterchanges();
names = tn.getStationNames();
fprintf('--- Interchanges ---\n');
for i = 1:numel(ixIds)
    linesList = tn.getStationLines(ixIds(i));
    fprintf('  %s: %s\n', names{ixIds(i)}, strjoin(linesList, ', '));
end
fprintf('\n');

% Lines and policies
fprintf('--- Line Details ---\n');
lineNames = tn.getLineNames();
for i = 1:numel(lineNames)
    line = tn.getLine(lineNames{i});
    L = line.getLength(tn.getGraph());
    fprintf('  [%s] Length: %.2f | Speed: %d km/h | Freq: %d min\n', ...
        lineNames{i}, L, ...
        line.getPolicy('SpeedLimit'), line.getPolicy('Frequency'));
end
fprintf('\n');

% Shortest path
[path, dist] = tn.shortestPath(7, 5);  % Botanic Gardens → Marina Bay
fprintf('--- Shortest Path: Botanic Gardens -> Marina Bay ---\n');
pathNames = names(path);
fprintf('  %s\n', strjoin(pathNames, ' -> '));
fprintf('  Distance: %.2f\n\n', dist);

%% 4. Visualize with auto-coloring from lines
data = tn.toVizData();

% Build GraphStyle from line data
nNodes = tn.getStationCount();
nEdges = tn.getRouteCount();
style = OceanViz.Adapters.GraphStyle(nNodes, nEdges);

% Auto-color edges by line
codes = tn.getLineNames();
for i = 1:numel(codes)
    line = tn.getLine(codes{i});
    style.setLineColor(codes{i}, line.Color, data.EdgeLabels);
end

% Set edge widths
for i = 1:nEdges
    style.setEdgeWidth(i, 3.5);
end

% Mark interchanges
for i = 1:numel(ixIds)
    style.setNodeIcon(ixIds(i), 'interchange');
    style.setNodeSize(ixIds(i), 10);
end

% Regular station sizes
for i = 1:nNodes
    if ~ismember(i, ixIds)
        style.setNodeSize(i, 7);
    end
end

% Highlight shortest path
style.highlightPath(path, [0.2, 1.0, 0.8], 2.0);

%% 5. Build legend from network data
lm = OceanViz.Adapters.LegendManager();
lm.fromTransitNetwork(tn);                                     % auto-adds lines + interchange icon
lm.addSymbol('path', 'Shortest Path', [0.2 1.0 0.8], '--');   % custom entry

%% 6. Render
viz = OceanViz.Core('Theme', 'Dark');
fig = figure('Name', 'MRT Network Demo', 'Color', viz.Colors.Background, ...
    'Position', [100 100 1000 700]);
ax = axes('Parent', fig);

data.Style = style;
strat = OceanViz.Strategies.TransitGraphStrategy();
strat.render(ax, data, viz.Colors);
title(ax, 'Mini Singapore MRT', 'Color', viz.Colors.Text, ...
    'FontSize', 14, 'FontWeight', 'bold');

% Render legend on top
lm.render(ax, 'Position', 'northeast');

fprintf('=== DEMO COMPLETE ===\n');

