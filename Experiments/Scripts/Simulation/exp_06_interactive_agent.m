% EXP_06_INTERACTIVE_AGENT  Animated MRT agent on a custom map.
%
%   Loads a custom JSON map built by TrackDesigner, initializes the physics
%   TransitSimulation engine, and visually animates a train moving along
%   the user's customized splines in real-time.

fprintf('\n=== CUSTOM MAP ANIMATED SIMULATION ===\n\n');

% Prevent base workspace pollution
clear data jsonText pathIndices sim tn viz ax fig;

%% 1. Load the Custom Map mathematically
baseDir = fileparts(fileparts(mfilename('fullpath'))); % Goes up from \Simulation to \Scripts
jsonFile = fullfile(baseDir, 'Test', 'test_case_1.json');
if exist(jsonFile, 'file') ~= 2
    error('Could not find %s. Please launch TrackDesigner, draw a map, and hit Export!', jsonFile);
end

tn = OceanViz.Designer.TrackDesigner.loadNetwork(jsonFile);
nNodes = tn.getStationCount();
nLines = tn.getLineCount();

if nNodes < 2 || nLines == 0
    error('Your custom map needs at least 2 stations and 1 completed Line to run an agent!');
end

fprintf('Loaded Custom Map: %d stations, %d lines.\n', nNodes, nLines);

%% 2. Initialize Simulation
sim = OceanMath.Transit.TransitSimulation(tn, 'Dt', 0.008);

% Find a guaranteed mathematically valid route using Dijkstra
[pathIndices, ~] = tn.shortestPath(1, nNodes);
if numel(pathIndices) < 2
    error('Cannot run simulation: No continuous physical track exists between Station 1 and Station %d.', nNodes);
end

% Create a bespoke simulation line dynamically that perfectly traces the Dijkstra path
tn.addLine('SIM1', 'Smart Routing Agent', [1.0, 0.4, 0.4]);
lineObj = tn.getLine('SIM1');
stationNames = cell(1, numel(pathIndices));
for i = 1:numel(pathIndices)
    lineObj.registerStation(pathIndices(i));
    stationNames{i} = tn.getStation(pathIndices(i)).Name;
end

% Spawn the Agent on the guaranteed route. Give it blistering physics constants so it moves fast!
sim.addAgent('SIM1', 'Speed', 35.0, 'DwellTime', 0.8, 'AccelRate', 20.0, 'BrakeRate', 25.0);

fprintf('Spawned Animated Agent tracking Shortest Path: %s\n\n', strjoin(stationNames, ' → '));

%% 3. Setup OceanViz Graphics
viz = OceanViz.Core('Theme', 'Dark');
fig = figure('Name', 'Custom Map Animated Agent', 'Color', viz.Colors.Background, ...
    'Position', [100 100 1000 700]);
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

% Center the camera globally like TrackDesigner
xlim(ax, [-50, 50]);
ylim(ax, [-50, 50]);
axis(ax, 'equal');
ax.XColor = 'none'; ax.YColor = 'none';

% Status Text Overlay
telemetryText = text(ax, -48, 48, 'Agent Telemetry Initializing...', ...
    'Color', [0.7 1.0 0.7], 'FontSize', 10, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');

%% 4. Animation Loop
fprintf('Starting real-time physics simulation...\n');

% Setup Agent Markers
hAgents = gobjects(1, sim.getAgentCount());
hTangent = gobjects(1, sim.getAgentCount());
hNormal  = gobjects(1, sim.getAgentCount());
agentColors = cell(1, sim.getAgentCount());
for i = 1:sim.getAgentCount()
    ag = sim.getAgent(i);
    cl = tn.getLine(ag.getLineCode()).Color;
    agentColors{i} = cl;
    
    % Draw train marker
    hAgents(i) = plot(ax, NaN, NaN, 'o', ...
        'MarkerSize', 16, ...
        'MarkerFaceColor', [1 1 1], ...
        'MarkerEdgeColor', cl, ...
        'LineWidth', 4, ...
        'HandleVisibility', 'off');
        
    % Draw TNB Quivers (Tangent = Green, Normal = Red)
    hTangent(i) = quiver(ax, NaN, NaN, NaN, NaN, 0, 'Color', [0.3 1.0 0.3], 'LineWidth', 2.5, 'MaxHeadSize', 0.5, 'HandleVisibility', 'off');
    hNormal(i)  = quiver(ax, NaN, NaN, NaN, NaN, 0, 'Color', [1.0 0.3 0.3], 'LineWidth', 2.5, 'MaxHeadSize', 0.5, 'HandleVisibility', 'off');
end

% Real-time physics engine loop
simTimer = tic;
frameCount = 0;

while isvalid(fig)
    % Step the mathematics engine
    sim.step();
    frameCount = frameCount + 1;
    
    % Sync graphics every N steps to save CPU
    if mod(frameCount, 2) == 0
        for i = 1:sim.getAgentCount()
            ag = sim.getAgent(i);
            
            % Update visual position
            st = ag.getState();
            pos = st.Position;
            if ~isempty(pos)
                set(hAgents(i), 'XData', pos(1), 'YData', pos(2));
                
                % Flash the border during dwell
                stateName = ag.getAgentState();
                if strcmpi(stateName, 'dwelling')
                    set(hAgents(i), 'MarkerEdgeColor', [1.0 0.8 0.2]); % Orange glow
                else
                    set(hAgents(i), 'MarkerEdgeColor', agentColors{i});
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
                        
                        % Query Tangent (T) and Normal (N) arrays dynamically
                        frenet = OceanMath.Geometry.FrenetFrame(splineObj);
                        [T_vec, N_vec, ~] = frenet.frame(st.Parameter);
                        
                        % Draw TN Vectors attached to agent, scaled by 6.0 purely for visual clarity
                        set(hTangent(i), 'XData', pos(1), 'YData', pos(2), 'UData', T_vec(1)*6.0, 'VData', T_vec(2)*6.0);
                        set(hNormal(i),  'XData', pos(1), 'YData', pos(2), 'UData', N_vec(1)*6.0, 'VData', N_vec(2)*6.0);
                    else
                        % Hide vectors if no active spline
                        set(hTangent(i), 'XData', NaN, 'YData', NaN, 'UData', NaN, 'VData', NaN);
                        set(hNormal(i),  'XData', NaN, 'YData', NaN, 'UData', NaN, 'VData', NaN);
                    end
                    
                    progress = ag.getProgress() * 100;
                    
                    msg = sprintf('Status: %s\nVelocity: %.1f units/s\nAccel: %.1f units/s^2\nEdge Distance: %s\nProgress: %.2f%%', ...
                        upper(stateName), v, a, distText, progress);
                    set(telemetryText, 'String', msg);
                end
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
