%% exp_mrt_real.m — Real Singapore MRT Network from CSV
%
% Supports two layout modes:
%   'geographic' — real SVY21 projected coordinates
%   'custom'     — loads a manually designed layout from .mat file
%                   (created via TrackDesigner GUI, see Section 7)
%
% Change LAYOUT below to switch.

clear; clc; close all;
fprintf('\n=== REAL SINGAPORE MRT NETWORK ===\n\n');

%% ====== USER CONFIG ======
LAYOUT = 'geographic';    % 'geographic' or 'custom'
CUSTOM_FILE = fullfile('Data', 'mrt_schematic.mat');  % saved via TrackDesigner
% ==========================

%% 1. Load CSV
projectRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
csvPath = fullfile(projectRoot, 'Data', 'MRT & LRT stations in Singapore (Jan 2025)', ...
    'mrt_lrt_stations.csv');
T = readtable(csvPath, 'VariableNamingRule', 'preserve');

isMRT = strcmp(T.TRANSPORT_TYPE, 'MRT');
T = T(isMRT, :);
T.LINE_ENGLISH = strtrim(T.LINE_ENGLISH);
fprintf('Loaded %d MRT station entries\n', height(T));

%% 2. Line definitions
linesDef = {
    'NSL', {'North-South Line'},                              [0.85, 0.12, 0.12]
    'EWL', {'East-West Line', 'Changi Airport Branch Line'},  [0.13, 0.70, 0.20]
    'NEL', {'North East Line'},                               [0.55, 0.20, 0.70]
    'CCL', {'Circle Line', 'Circle Line Extension'},          [0.95, 0.60, 0.10]
    'DTL', {'Downtown Line'},                                 [0.15, 0.40, 0.85]
    'TEL', {'Thomson-East Coast Line'},                       [0.60, 0.40, 0.20]
};

%% 3. Build TransitNetwork from CSV (geographic coordinates)
tn = OceanMath.Transit.TransitNetwork();
stationMap = containers.Map('KeyType', 'char', 'ValueType', 'double');

for li = 1:size(linesDef, 1)
    lineCode  = linesDef{li, 1};
    lineNames = linesDef{li, 2};
    lineColor = linesDef{li, 3};

    tn.addLine(lineCode, lineNames{1}, lineColor);

    mask = false(height(T), 1);
    for ni = 1:numel(lineNames)
        mask = mask | strcmp(T.LINE_ENGLISH, lineNames{ni});
    end
    lineRows = find(mask);
    if isempty(lineRows), continue; end

    [~, si] = sort(T.ALPHANUMERIC_CODE(lineRows));
    lineRows = lineRows(si);

    stationIds = zeros(numel(lineRows), 1);
    for si = 1:numel(lineRows)
        row = lineRows(si);
        name = strtrim(T.STATION_NAME_ENGLISH{row});
        xPos = T.X(row) / 1000;
        yPos = T.Y(row) / 1000;

        if stationMap.isKey(name)
            stationIds(si) = stationMap(name);
        else
            id = tn.addStation(name, [xPos, yPos]);
            stationMap(name) = id;
            stationIds(si) = id;
        end
    end

    % Connect consecutive stations with straight Bezier curves
    N = numel(stationIds);
    for si = 1:N - 1
        srcPos = tn.getStation(stationIds(si)).Position;
        tgtPos = tn.getStation(stationIds(si+1)).Position;
        mid1 = srcPos + (tgtPos - srcPos) * 1/3;
        mid2 = srcPos + (tgtPos - srcPos) * 2/3;
        spline = OceanMath.Geometry.BezierCurve([srcPos; mid1; mid2; tgtPos]);
        tn.addRouteToLine(lineCode, stationIds(si), stationIds(si+1), spline);
    end

    fprintf('  %s: %d stations\n', lineCode, numel(stationIds));
end

fprintf('\nNetwork: %d stations, %d routes, %d lines\n', ...
    tn.getStationCount(), tn.getRouteCount(), tn.getLineCount());

%% 4. If custom layout requested, load saved design from TrackDesigner
if strcmp(LAYOUT, 'custom')
    matPath = fullfile(projectRoot, CUSTOM_FILE);
    if ~isfile(matPath)
        warning(['Custom layout file not found: %s\n' ...
                 'Use TrackDesigner to create one (see Section 7 below).\n' ...
                 'Falling back to geographic layout.'], matPath);
        LAYOUT = 'geographic';
    else
        fprintf('\nLoading custom layout from: %s\n', CUSTOM_FILE);
        td = OceanViz.Designer.TrackDesigner();
        td.loadDesign(matPath);
        tn = td.exportTransitNetwork();
        % Re-register line colors (the saved design preserves structure)
        fprintf('Custom layout loaded: %d stations\n', tn.getStationCount());
    end
end

%% 5. Visualize
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('Singapore MRT Network');
set(fig, 'Position', [50, 50, 1400, 900]);
ax  = axes('Parent', fig);
viz.styleAxes(ax);
hold(ax, 'on');

fprintf('  Preparing viz data...\n');
data = tn.toVizData();
nEdges = size(data.EdgePairs, 1);
nNodes = size(data.Positions, 1);
style = OceanViz.Adapters.GraphStyle(nNodes, nEdges);

% Color edges
for li = 1:size(linesDef, 1)
    style.setLineColor(linesDef{li, 1}, linesDef{li, 3}, data.EdgeLabels);
end
for ei = 1:nEdges
    style.setEdgeWidth(ei, 2.0);
end

% Nodes
interchanges = data.InterchangeIds;
isInterchange = false(nNodes, 1);
if ~isempty(interchanges)
    isInterchange(interchanges) = true;
end

showLabels = strcmp(LAYOUT, 'custom');
for ni = 1:nNodes
    if ~showLabels
        style.setNodeLabelVisible(ni, false);
    end
    if isInterchange(ni)
        style.setNodeIcon(ni, 'interchange');
        style.setNodeSize(ni, 7);
    else
        style.setNodeSize(ni, 4);
    end
end

data.Style = style;

fprintf('  Rendering map (%d edges, %d nodes)...\n', nEdges, nNodes);
strat = OceanViz.Strategies.TransitGraphStrategy();
strat.render(ax, data, viz.Colors);
fprintf('  Map rendered.\n');

if strcmp(LAYOUT, 'geographic')
    axis(ax, 'normal');
end
pos = data.Positions;
xPad = range(pos(:,1)) * 0.08;
yPad = range(pos(:,2)) * 0.08;
xlim(ax, [min(pos(:,1)) - xPad, max(pos(:,1)) + xPad]);
ylim(ax, [min(pos(:,2)) - yPad, max(pos(:,2)) + yPad]);

title(ax, sprintf('Singapore MRT (%s)', upper(LAYOUT)), ...
    'Color', viz.Colors.Text, 'FontSize', 16);

%% 6. Legend
for li = 1:size(linesDef, 1)
    plot(ax, NaN, NaN, '-', 'Color', linesDef{li, 3}, ...
        'LineWidth', 3, 'DisplayName', linesDef{li, 2}{1});
end
plot(ax, NaN, NaN, 's', 'MarkerFaceColor', [1.0 0.85 0.3], ...
    'MarkerEdgeColor', [1 1 1], 'MarkerSize', 8, ...
    'DisplayName', 'Interchange', 'LineStyle', 'none');
legend(ax, 'Location', 'best', 'TextColor', viz.Colors.Text, ...
    'Color', [0.15 0.15 0.18], 'EdgeColor', [0.3 0.3 0.3], ...
    'FontSize', 9);

fprintf('\nMap rendered. Starting simulation...\n');

%% 7. Animate ONE agent with REAL-TIME SCALING + TRAPEZOIDAL PROFILE
SIMULATE_LINE = 'NSL';  % which line to simulate

% --- Real MRT parameters ---
SPEED_KMH   = 45;                         % real avg cruise speed: 45 km/h
DWELL_SEC   = 30;                         % real dwell: 30 seconds per station
ACCEL_MS2   = 1.0;                        % real acceleration: 1.0 m/s^2
BRAKE_MS2   = 1.2;                        % real braking: 1.2 m/s^2
TIME_SCALE  = 200;                        % 200x fast-forward

% Convert to simulation units (coordinates are in km)
realSpeed  = SPEED_KMH / 3600;             % km/s = 0.0125
simSpeed   = realSpeed * TIME_SCALE;        % 2.5 km/s (scaled)
simDwell   = DWELL_SEC / TIME_SCALE;        % 0.15 s (scaled)
simDt      = 0.02;                          % sim time step

% Acceleration scales by TIME_SCALE^2 (a = distance/time^2)
%   real: m/s^2 -> km/s^2 (divide by 1000)
%   then multiply by TIME_SCALE^2
simAccel   = (ACCEL_MS2 / 1000) * TIME_SCALE^2;
simBrake   = (BRAKE_MS2 / 1000) * TIME_SCALE^2;

fprintf('  Real: %d km/h | accel=%.1f m/s^2 | brake=%.1f m/s^2 | dwell=%ds | %dx\n', ...
    SPEED_KMH, ACCEL_MS2, BRAKE_MS2, DWELL_SEC, TIME_SCALE);

sim = OceanMath.Transit.TransitSimulation(tn, 'Dt', simDt);
sim.addAgent(SIMULATE_LINE, 'Speed', simSpeed, 'DwellTime', simDwell, ...
    'AccelRate', simAccel, 'BrakeRate', simBrake);
nLines = 1;
lineCodes = {SIMULATE_LINE};
fprintf('  Created 1 agent on %s (trapezoidal profile)\n', SIMULATE_LINE);

% Create one plot handle per train (each with its own color)
trainHandles = gobjects(nLines, 1);
for li = 1:nLines
    agentColor = linesDef{li, 3};
    trainHandles(li) = plot(ax, NaN, NaN, 'o', ...
        'MarkerSize', 10, ...
        'MarkerFaceColor', agentColor, ...
        'MarkerEdgeColor', [1 1 1], ...
        'LineWidth', 1.5, ...
        'HandleVisibility', 'off');
end

% Clock text
clockText = text(ax, 0.02, 0.02, '', ...
    'Units', 'normalized', ...
    'Color', viz.Colors.Text, ...
    'FontSize', 11, 'FontWeight', 'bold', ...
    'BackgroundColor', [0.1 0.1 0.12 0.8], ...
    'Margin', 4);

% Status text
statusText = text(ax, 0.02, 0.07, '', ...
    'Units', 'normalized', ...
    'Color', [0.6 0.6 0.6], ...
    'FontSize', 9, ...
    'BackgroundColor', [0.1 0.1 0.12 0.8], ...
    'Margin', 3);

% --- TNB Frame visualization ---
FOCUS_AGENT = 1;

% Arrow scale = 5% of map diagonal (ratio-based, not hardcoded)
posAll = data.Positions;
mapDiag = sqrt(range(posAll(:,1))^2 + range(posAll(:,2))^2);
arrowLen = mapDiag * 0.05;  % arrows are 5% of map size

% Quiver handles for T (tangent) and N (normal) vectors
hT = quiver(ax, NaN, NaN, 0, 0, 0, ...
    'Color', [0.3 1.0 0.3], 'LineWidth', 2, ...
    'MaxHeadSize', 0.8, 'HandleVisibility', 'off');
hN = quiver(ax, NaN, NaN, 0, 0, 0, ...
    'Color', [1.0 0.3 0.3], 'LineWidth', 2, ...
    'MaxHeadSize', 0.8, 'HandleVisibility', 'off');

% Legend entries for T and N
plot(ax, NaN, NaN, '-', 'Color', [0.3 1.0 0.3], ...
    'LineWidth', 2, 'DisplayName', 'T (tangent)');
plot(ax, NaN, NaN, '-', 'Color', [1.0 0.3 0.3], ...
    'LineWidth', 2, 'DisplayName', 'N (normal)');
legend(ax, 'Location', 'best', 'TextColor', viz.Colors.Text, ...
    'Color', [0.15 0.15 0.18], 'EdgeColor', [0.3 0.3 0.3], ...
    'FontSize', 9);

% Info panel (bottom-right)
infoText = text(ax, 0.98, 0.02, '', ...
    'Units', 'normalized', ...
    'HorizontalAlignment', 'right', ...
    'Color', [0.9 0.9 0.9], ...
    'FontSize', 9, 'FontName', 'Consolas', ...
    'BackgroundColor', [0.08 0.08 0.10 0.9], ...
    'EdgeColor', [0.3 0.3 0.35], ...
    'Margin', 6, ...
    'VerticalAlignment', 'bottom');

% Animation loop
fprintf('  Animating...\n');
frameSkip = 1;     % every frame (only 1 agent, so it's fast)
frameCount = 0;
maxFrames  = 50000;

while ~sim.isAllArrived() && frameCount < maxFrames
    sim.step();
    frameCount = frameCount + 1;

    if mod(frameCount, frameSkip) ~= 0
        continue;
    end

    % Update train dot
    agentPos = sim.getPositions();
    set(trainHandles(1), 'XData', agentPos(1,1), 'YData', agentPos(1,2));

    % --- TNB vectors for the focused agent ---
    focusAgent = sim.getAgent(FOCUS_AGENT);
    if ~focusAgent.isArrived() && strcmp(focusAgent.getAgentState(), 'running')
        s = focusAgent.getState();
        agentPos = s.Position;
        vel = s.Velocity;
        tDir = s.Tangent;  % unit tangent from KinematicState

        % Normal: rotate tangent 90 deg CCW in 2D
        nDir = [-tDir(2), tDir(1)];

        % Scale arrows: unit vectors * fixed ratio of map size
        scaledT = tDir * arrowLen;
        scaledN = nDir * arrowLen * 0.6;

        % Update quiver arrows
        set(hT, 'XData', agentPos(1), 'YData', agentPos(2), ...
                'UData', scaledT(1),  'VData', scaledT(2));
        set(hN, 'XData', agentPos(1), 'YData', agentPos(2), ...
                'UData', scaledN(1),  'VData', scaledN(2));

        % Compute curvature from the active spline via FrenetFrame
        kappa = 0;
        segIdx = focusAgent.getCurrentSegment();
        nSeg = focusAgent.getSegmentCount();
        try
            activeSpline = focusAgent.getActiveSpline();
            frenet = OceanMath.Geometry.FrenetFrame(activeSpline);
            kappa = frenet.curvature(s.Parameter);
        catch
            % spline may not support curvature — graceful fallback
        end

        % Update info panel with REAL units
        realVel_kmh  = vel / TIME_SCALE * 3600;   % sim km/s → real km/h
        realTime_min = s.Time * TIME_SCALE / 60;   % sim s → real minutes
        realDist_km  = s.ArcDistance;               % already in km
        infoStr = sprintf([ ...
            ' Agent: %s (%dx fast-forward) \n', ...
            ' v = %.1f km/h \n', ...
            ' a = %.2f km/s^2 \n', ...
            ' t = %.4f (param) \n', ...
            ' real time = %.1f min \n', ...
            ' distance = %.1f km \n', ...
            ' k = %.4f \n', ...
            ' segment: %d / %d '], ...
            SIMULATE_LINE, TIME_SCALE, ...
            realVel_kmh, s.Acceleration, ...
            s.Parameter, realTime_min, ...
            realDist_km, kappa, segIdx, nSeg);
        set(infoText, 'String', infoStr);
    else
        % Agent dwelling or arrived — hide vectors
        set(hT, 'UData', 0, 'VData', 0);
        set(hN, 'UData', 0, 'VData', 0);
    end

    % Update clock (show real time)
    realClock_min = sim.getClock() * TIME_SCALE / 60;
    set(clockText, 'String', sprintf(' Real time: %.1f min ', realClock_min));

    % Update status
    nRunning = 0;
    for li = 1:nLines
        if ~sim.getAgent(li).isArrived()
            nRunning = nRunning + 1;
        end
    end
    set(statusText, 'String', sprintf(' Trains running: %d / %d ', nRunning, nLines));

    drawnow;
    pause(0.01);  % smooth 100fps cap
end

% Final state
set(trainHandles(1), 'MarkerSize', 12, 'MarkerFaceColor', [1.0 0.85 0.3]);
set(hT, 'UData', 0, 'VData', 0);
set(hN, 'UData', 0, 'VData', 0);
set(clockText, 'String', sprintf(' FINISHED  t = %.1f s ', sim.getClock()));
set(statusText, 'String', ' All trains arrived ');
drawnow;

fprintf('\n=== SIMULATION COMPLETE (%.1f s) ===\n', sim.getClock());

%% 8. CREATE A CUSTOM LAYOUT
%
% Step 1: Open TrackDesigner
%   td = OceanViz.Designer.TrackDesigner();
%   td.launch();
%
% Step 2: Place stations and connect them (click, drag)
%
% Step 3: Save:
%   td.saveDesign('Data/mrt_schematic.mat')
%
% Step 4: Change LAYOUT = 'custom' above and run again.
