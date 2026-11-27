% EXP_07_3D_ROLLERCOASTER  Animated 3D physics on a custom map.
%
%   Loads a custom JSON map built by TrackDesigner with Z-Elevations,
%   initializes the Newtonian Gravity physics engine, and visually animates
%   a coaster car moving along 6-DOF splines using T, N, B components.

fprintf('\n=== CUSTOM 3D ROLLERCOASTER SIMULATION ===\n\n');

% Prevent base workspace pollution
clear data jsonText pathIndices sim tn viz ax fig;

%% 1. Load the Custom Map mathematically
baseDir = fileparts(fileparts(mfilename('fullpath'))); % Goes up from \Simulation to \Scripts
jsonFile = fullfile(baseDir, 'Test', 'test_case_1.json');
if exist(jsonFile, 'file') ~= 2
    error('Could not find %s. Please launch TrackDesigner, draw a 3D map, and hit Export!', jsonFile);
end

tn = OceanViz.Designer.TrackDesigner.loadNetwork(jsonFile);
nNodes = tn.getStationCount();
nLines = tn.getLineCount();

if nNodes < 2
    error('Your custom map needs at least 2 stations to run a rollercoaster!');
end

fprintf('Loaded 3D Custom Map: %d stations, %d lines.\n', nNodes, nLines);

%% 2. Initialize Routing
% Find a guaranteed mathematically valid route using Dijkstra
[pathIndices, ~] = tn.shortestPath(1, nNodes);
if numel(pathIndices) < 2
    error('Cannot run simulation: No continuous track exists between Station 1 and Station %d.', nNodes);
end

% Create a bespoke simulation line dynamically that perfectly traces the Dijkstra path
tn.addLine('SIM1', 'Newtonian Gravity Agent', [1.0, 0.4, 0.4]);
lineObj = tn.getLine('SIM1');
stationNames = cell(1, numel(pathIndices));
for i = 1:numel(pathIndices)
    lineObj.registerStation(pathIndices(i));
    stationNames{i} = tn.getStation(pathIndices(i)).Name;
end

% Spawn the Kinematic Coaster on the guaranteed route. Give it a gentle initial push!
ag = OceanMath.Transit.KinematicAgent(tn, 'SIM1', 'Speed', 5.0, 'Dt', 0.008);

fprintf('Spawned Newtonian Coaster tracking Shortest Path: %s\n\n', strjoin(stationNames, ' → '));

%% 3. Setup OceanViz Graphics
viz = OceanViz.Core('Theme', 'Dark');
fig = figure('Name', '3D Rollercoaster Newtonian Simulator', 'Color', viz.Colors.Background, ...
    'Position', [100 100 1200 800]);
ax = axes('Parent', fig);
hold(ax, 'on');

% --- Generate the Base Map ---
data = tn.toVizData();
nEdges = tn.getRouteCount();
style = OceanViz.Adapters.GraphStyle(nNodes, nEdges);

codes = tn.getLineNames();
for i = 1:numel(codes)
    line = tn.getLine(codes{i});
    style.setLineColor(codes{i}, line.Color, data.EdgeLabels);
end
for i = 1:nEdges
    style.setEdgeWidth(i, 3.0); % THICK TRACKS
end
for i = 1:nNodes
    style.setNodeSize(i, 8);
    style.setNodeIcon(i, 'square'); % Match TrackDesigner
end

data.Style = style;
strat = OceanViz.Strategies.TransitGraphStrategy();
strat.render(ax, data, viz.Colors);

% Activate 3D Camera globally
xlim(ax, [-50, 50]);
ylim(ax, [-50, 50]);
zlim(ax, [-50, 50]);
axis(ax, 'equal');
ax.XColor = 'none'; ax.YColor = 'none'; ax.ZColor = 'none';
view(ax, 3);
rotate3d(ax, 'on');

% Status Text Overlay
telemetryText = text(ax, -48, 48, 50, 'Agent Telemetry Initializing...', ...
    'Color', [0.7 1.0 0.7], 'FontSize', 10, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');

%% 4. Animation Loop
fprintf('Starting real-time 3D physics simulation...\n');
fprintf('DRAG TO ROTATE CAMERA FREE-LOOK!\n');

% Setup Agent Markers
agentColor = tn.getLine('SIM1').Color;

% Draw train marker (Now 3D!)
hAgent = plot3(ax, NaN, NaN, NaN, 'o', ...
    'MarkerSize', 16, ...
    'MarkerFaceColor', [1 1 1], ...
    'MarkerEdgeColor', agentColor, ...
    'LineWidth', 4, ...
    'HandleVisibility', 'off');

% Draw TNB Quivers (Tangent = Green, Normal = Red, Binormal = Blue)
hTangent  = quiver3(ax, NaN, NaN, NaN, NaN, NaN, NaN, 0, 'Color', [0.3 1.0 0.3], 'LineWidth', 2.5, 'MaxHeadSize', 0.5, 'HandleVisibility', 'off');
hNormal   = quiver3(ax, NaN, NaN, NaN, NaN, NaN, NaN, 0, 'Color', [1.0 0.3 0.3], 'LineWidth', 2.5, 'MaxHeadSize', 0.5, 'HandleVisibility', 'off');
hBinormal = quiver3(ax, NaN, NaN, NaN, NaN, NaN, NaN, 0, 'Color', [0.3 0.3 1.0], 'LineWidth', 2.5, 'MaxHeadSize', 0.5, 'HandleVisibility', 'off');

% Real-time physics engine loop
simTimer = tic;
frameCount = 0;

while isvalid(fig)
    % Step the mathematics engine
    ag.step();
    frameCount = frameCount + 1;
    
    % Sync graphics every N steps to save CPU
    if mod(frameCount, 2) == 0
        % Update visual position
        st = ag.getState();
        pos = st.Position;
        if length(pos) < 3
            pos = [pos(1), pos(2), 0];
        end
        if ~isempty(pos)
            set(hAgent, 'XData', pos(1), 'YData', pos(2), 'ZData', pos(3));
            
            % Check if stalled or chain-lifting
            stateName = ag.getAgentState();
            if strcmpi(stateName, 'dwelling')
                set(hAgent, 'MarkerEdgeColor', [1.0 0.8 0.2]); % Orange glow
            else
                set(hAgent, 'MarkerEdgeColor', agentColor);
            end
            
            % Update Telemetry
            if isvalid(telemetryText)
                v = st.Velocity;
                a = st.Acceleration;
                
                % Get edge length and compute local Frenet Frame
                splineObj = ag.getActiveSpline();
                distText = 'N/A';
                if ~isempty(splineObj)
                    edgeLength = splineObj.totalLength();
                    distText = sprintf('%.1f units', edgeLength);
                    
                    % Query Tangent (T), Normal (N), and Binormal (B) vividly from Math Framework
                    frenet = OceanMath.Geometry.FrenetFrame(splineObj);
                    [T_vec, N_vec, B_vec] = frenet.frame(st.Parameter);
                    
                    % Ensure vectors have Z components for plotting
                    if length(T_vec) < 3, T_vec = [T_vec(1), T_vec(2), 0]; end
                    if length(N_vec) < 3, N_vec = [N_vec(1), N_vec(2), 0]; end
                    if length(B_vec) < 3, B_vec = [B_vec(1), B_vec(2), 0]; end
                    
                    vecScale = 6.0;
                    set(hTangent,  'XData', pos(1), 'YData', pos(2), 'ZData', pos(3), 'UData', T_vec(1)*vecScale, 'VData', T_vec(2)*vecScale, 'WData', T_vec(3)*vecScale);
                    set(hNormal,   'XData', pos(1), 'YData', pos(2), 'ZData', pos(3), 'UData', N_vec(1)*vecScale, 'VData', N_vec(2)*vecScale, 'WData', N_vec(3)*vecScale);
                    set(hBinormal, 'XData', pos(1), 'YData', pos(2), 'ZData', pos(3), 'UData', B_vec(1)*vecScale, 'VData', B_vec(2)*vecScale, 'WData', B_vec(3)*vecScale);
                else
                    % Hide vectors if no active spline
                    set(hTangent,  'XData', NaN, 'YData', NaN, 'ZData', NaN, 'UData', NaN, 'VData', NaN, 'WData', NaN);
                    set(hNormal,   'XData', NaN, 'YData', NaN, 'ZData', NaN, 'UData', NaN, 'VData', NaN, 'WData', NaN);
                    set(hBinormal, 'XData', NaN, 'YData', NaN, 'ZData', NaN, 'UData', NaN, 'VData', NaN, 'WData', NaN);
                end
                
                progress = ag.getProgress() * 100;
                
                msg = sprintf('Status: %s\nVelocity: %.1f units/s\nGravity Accel: %.1f units/s^2\nEdge Distance: %s\nProgress: %.2f%%', ...
                    upper(stateName), v, a, distText, progress);
                set(telemetryText, 'String', msg);
            end
        end
        drawnow limitrate;
    end
    
    % Cap simulation speed (roughly 60 FPS)
    dt_sleep = 0.016 - toc(simTimer);
    if dt_sleep > 0
        pause(dt_sleep);
    else
        pause(0.001); % Yield
    end
    simTimer = tic;
end
