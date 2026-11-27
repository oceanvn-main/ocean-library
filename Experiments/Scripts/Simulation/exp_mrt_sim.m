% EXP_MRT_SIM  Animated MRT simulation demo.
%
%   Extends exp_mrt_demo with kinematic train agents:
%     - 3 trains (NSL, EWL, DTL) moving along their routes
%     - Dwell at stations, smooth spline traversal
%     - Real-time animated rendering using OceanViz
%
%   Uses: TransitNetwork, TransitAgent, TransitSimulation,
%          CurveSimulator, BezierCurve, OceanViz.Core

fprintf('\n=== MRT SIMULATION DEMO ===\n\n');

%% 1. Build the transit network (same as exp_mrt_demo)
tn = OceanMath.Transit.TransitNetwork();

tn.addLine('NSL', 'North-South Line', [0.85, 0.15, 0.15]);
tn.addLine('EWL', 'East-West Line',   [0.15, 0.65, 0.15]);
tn.addLine('DTL', 'Downtown Line',    [0.15, 0.45, 0.85]);

tn.getLine('NSL').setPolicy('SpeedLimit', 80);
tn.getLine('EWL').setPolicy('SpeedLimit', 90);
tn.getLine('DTL').setPolicy('SpeedLimit', 100);

tn.addStation('Jurong East',     [1, 5]);      % id=1
tn.addStation('Orchard',         [5, 7]);      % id=2
tn.addStation('Dhoby Ghaut',     [5, 5]);      % id=3
tn.addStation('City Hall',       [9, 5]);      % id=4
tn.addStation('Marina Bay',      [5, 3]);      % id=5
tn.addStation('Bugis',           [9, 7]);      % id=6
tn.addStation('Botanic Gardens', [1, 8]);      % id=7
tn.addStation('Newton',          [5, 9]);      % id=8
tn.addStation('Clementi',        [3, 5]);      % id=9
tn.addStation('Bayfront',        [9, 1]);      % id=10

% NSL (red): Newton -> Orchard -> Dhoby Ghaut -> City Hall -> Marina Bay
%   NOTE: Dhoby Ghaut -> City Hall is SHARED with EWL (parallel rendering test)
tn.addRouteToLine('NSL', 8, 2, OceanMath.Geometry.BezierCurve([5 9; 5 8.3; 5 7.7; 5 7]));
tn.addRouteToLine('NSL', 2, 3, OceanMath.Geometry.BezierCurve([5 7; 5 6.3; 5 5.7; 5 5]));
tn.addRouteToLine('NSL', 3, 4, OceanMath.Geometry.BezierCurve([5 5; 6.3 5; 7.7 5; 9 5]));  % shared with EWL
tn.addRouteToLine('NSL', 4, 5, OceanMath.Geometry.BezierCurve([9 5; 8 4; 6.5 3.5; 5 3]));

% EWL (green): Jurong East -> Clementi -> Dhoby Ghaut -> City Hall
tn.addRouteToLine('EWL', 1, 9, OceanMath.Geometry.BezierCurve([1 5; 1.7 5; 2.3 5; 3 5]));
tn.addRouteToLine('EWL', 9, 3, OceanMath.Geometry.BezierCurve([3 5; 3.7 5; 4.3 5; 5 5]));
tn.addRouteToLine('EWL', 3, 4, OceanMath.Geometry.BezierCurve([5 5; 6.3 5; 7.7 5; 9 5]));

% DTL (blue): Botanic Gardens -> Dhoby Ghaut -> Bugis -> Bayfront
tn.addRouteToLine('DTL', 7, 3, OceanMath.Geometry.BezierCurve([1 8; 2 7.2; 3.5 6; 5 5]));
tn.addRouteToLine('DTL', 3, 6, OceanMath.Geometry.BezierCurve([5 5; 6.5 5.5; 7.5 6.2; 9 7]));
tn.addRouteToLine('DTL', 6, 10, OceanMath.Geometry.BezierCurve([9 7; 9 5.5; 9 3; 9 1]));

fprintf('Network built: %d stations, %d routes, %d lines\n', ...
    tn.getStationCount(), tn.getRouteCount(), tn.getLineCount());

%% 2. Create the simulation with 3 train agents
sim = OceanMath.Transit.TransitSimulation(tn, 'Dt', 0.005);

sim.addAgent('NSL', 'Speed', 0.8, 'DwellTime', 0.8);
sim.addAgent('EWL', 'Speed', 1.0, 'DwellTime', 0.8);
sim.addAgent('DTL', 'Speed', 0.9, 'DwellTime', 0.8);

fprintf('Agents: %d trains ready\n\n', sim.getAgentCount());

%% 3. Set up visualization
viz = OceanViz.Core('Theme', 'Dark');
fig = figure('Name', 'MRT Simulation', 'Color', viz.Colors.Background, ...
    'Position', [100 100 1000 700]);
ax = axes('Parent', fig);
hold(ax, 'on');

% Static layer: render the network map
data = tn.toVizData();
nNodes = tn.getStationCount();
nEdges = tn.getRouteCount();
style = OceanViz.Adapters.GraphStyle(nNodes, nEdges);

codes = tn.getLineNames();
for i = 1:numel(codes)
    line = tn.getLine(codes{i});
    style.setLineColor(codes{i}, line.Color, data.EdgeLabels);
end
for i = 1:nEdges
    style.setEdgeWidth(i, 3.5);
end
ixIds = tn.getInterchanges();
for i = 1:numel(ixIds)
    style.setNodeIcon(ixIds(i), 'interchange');
    style.setNodeSize(ixIds(i), 10);
end
for i = 1:nNodes
    if ~ismember(i, ixIds)
        style.setNodeSize(i, 7);
    end
end

data.Style = style;
strat = OceanViz.Strategies.TransitGraphStrategy();
strat.render(ax, data, viz.Colors);

% Legend
lm = OceanViz.Adapters.LegendManager();
lm.fromTransitNetwork(tn);
lm.render(ax, 'Position', 'northeast');

title(ax, 'MRT Simulation', 'Color', viz.Colors.Text, ...
    'FontSize', 14, 'FontWeight', 'bold');

% Lock axes so nothing blows up the view
xlim(ax, [-1, 12]);
ylim(ax, [-1, 11]);
set(ax, 'XLimMode', 'manual', 'YLimMode', 'manual');

%% 4. Animated simulation loop
nAgents = sim.getAgentCount();
trainHandles = gobjects(1, nAgents);
agentColors  = sim.getColors();

for i = 1:nAgents
    trainHandles(i) = plot(ax, NaN, NaN, 'o', ...
        'MarkerSize', 14, ...
        'MarkerFaceColor', agentColors(i,:), ...
        'MarkerEdgeColor', [1 1 1], ...
        'LineWidth', 2.5);
end

% Clock text
clockText = text(ax, 0.02, 0.97, 't = 0.00 s', ...
    'Units', 'normalized', ...
    'Color', viz.Colors.Text, ...
    'FontSize', 11, ...
    'FontWeight', 'bold', ...
    'VerticalAlignment', 'top');

fprintf('Running simulation...\n');
frameCount = 0;
maxFrames  = 50000;

while ~sim.isAllArrived() && frameCount < maxFrames
    if ~isvalid(fig), break; end
    sim.step();
    frameCount = frameCount + 1;

    if mod(frameCount, 5) == 0
        positions = sim.getPositions();
        for i = 1:nAgents
            if isvalid(trainHandles(i))
                set(trainHandles(i), 'XData', positions(i,1), ...
                                     'YData', positions(i,2));
            end
        end
        if isvalid(clockText)
            clockText.String = sprintf('t = %.2f s', sim.getClock());
        end
        drawnow limitrate;
    end
end

% Final update
try
    positions = sim.getPositions();
    for i = 1:nAgents
        set(trainHandles(i), 'XData', positions(i,1), 'YData', positions(i,2));
    end
    clockText.String = sprintf('t = %.2f s (DONE)', sim.getClock());
    drawnow;
catch
end

%% 5. Print final status
fprintf('\n--- Simulation Complete ---\n');
sim.describe();
fprintf('\n=== DEMO COMPLETE ===\n');
