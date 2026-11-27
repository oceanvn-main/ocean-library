% EXP_FRENET_DEMO  Animated Frenet frame visualization: 2D curve + 3D helix.
%
%   Two subplots side by side:
%     LEFT:  2D CatmullRomSpline S-curve with T (red) and N (green) arrows
%     RIGHT: 3D Helix (BezierCurve) with full T, N, B arrows
%
%   Both panels show:
%     - Static curvature comb
%     - Animated agent with TNB arrows
%     - Osculating circle (2D only)
%     - Live info readout
%
%   Uses: CatmullRomSpline, BezierCurve, FrenetFrame, CurveSimulator,
%          FrenetVizStrategy, OceanViz.Core

fprintf('\n=== FRENET FRAME VISUALIZATION DEMO ===\n\n');

%% 1. Define Curves

% 2D: S-curve with varying curvature
pts2D = [0 0; 1 3; 3 4; 5 2; 7 4; 9 1; 10 3];
curve2D = OceanMath.Geometry.CatmullRomSpline(pts2D);
fprintf('2D Curve: CatmullRomSpline (%d pts, L=%.2f)\n', ...
    size(pts2D,1), curve2D.totalLength());

% 3D: Helix approximated with a high-degree Bezier
nHelixPts = 30;
tH = linspace(0, 4*pi, nHelixPts)';
helixRadius = 2;
helixPitch  = 0.5;
helixCtrl = [helixRadius*cos(tH), helixRadius*sin(tH), helixPitch*tH];
curve3D = OceanMath.Geometry.BezierCurve(helixCtrl);
fprintf('3D Curve: BezierCurve helix (%d pts, L=%.2f)\n', ...
    nHelixPts, curve3D.totalLength());

%% 2. Set up figure with 2 panels
viz = OceanViz.Core('Theme', 'Dark');
fig = figure('Name', 'Frenet Frame Demo — 2D & 3D', ...
    'Color', viz.Colors.Background, ...
    'Position', [50 80 1400 600]);

% --- LEFT: 2D ---
ax2D = subplot(1, 2, 1, 'Parent', fig);
viz.styleAxes(ax2D);
hold(ax2D, 'on');
axis(ax2D, 'equal');
xlim(ax2D, [-1, 11]); ylim(ax2D, [-2, 6]);
set(ax2D, 'XLimMode', 'manual', 'YLimMode', 'manual');
title(ax2D, '2D: T (red), N (green)', ...
    'Color', viz.Colors.Text, 'FontSize', 12, 'FontWeight', 'bold');

% Draw the curve
tPts = curve2D.parameterize(200);
cPts = curve2D.evaluate(tPts);
plot(ax2D, cPts(:,1), cPts(:,2), '-', ...
    'Color', [0.5 0.8 1.0 0.4], 'LineWidth', 2);

% Draw curvature comb
frenet2D = OceanMath.Geometry.FrenetFrame(curve2D);
combT = linspace(0.02, 0.98, 50)';
combPts = curve2D.evaluate(combT);
combK = frenet2D.curvature(combT);
[~, combN, ~] = frenet2D.frame(combT);
maxK = max(combK);
if maxK > 1e-10
    scaledK = (combK / maxK) * 0.8;
else
    scaledK = zeros(size(combK));
end
for ci = 1:numel(combT)
    ep = combPts(ci,:) + combN(ci,:) * scaledK(ci);
    plot(ax2D, [combPts(ci,1), ep(1)], [combPts(ci,2), ep(2)], '-', ...
        'Color', [0.9 0.6 1.0 0.5], 'LineWidth', 1);
end

% Animated handles: agent dot + TNB arrows + osculating circle
dot2D = plot(ax2D, NaN, NaN, 'o', 'MarkerSize', 10, ...
    'MarkerFaceColor', [0.2 1.0 0.8], 'MarkerEdgeColor', [1 1 1], 'LineWidth', 1.5);
trail2D = plot(ax2D, NaN, NaN, '-', 'Color', [0.2 1.0 0.8 0.25], 'LineWidth', 1.5);

% T and N quivers
qT2 = quiver(ax2D, NaN, NaN, NaN, NaN, 0, ...
    'Color', [1.0 0.3 0.3], 'LineWidth', 2.5, 'MaxHeadSize', 0.6);
qN2 = quiver(ax2D, NaN, NaN, NaN, NaN, 0, ...
    'Color', [0.3 1.0 0.3], 'LineWidth', 2.5, 'MaxHeadSize', 0.6);
lblT2 = text(ax2D, NaN, NaN, 'T', 'Color', [1.0 0.3 0.3], ...
    'FontWeight', 'bold', 'FontSize', 10);
lblN2 = text(ax2D, NaN, NaN, 'N', 'Color', [0.3 1.0 0.3], ...
    'FontWeight', 'bold', 'FontSize', 10);

% Osculating circle
oscCircle2D = plot(ax2D, NaN, NaN, '--', ...
    'Color', [1.0 0.85 0.3 0.6], 'LineWidth', 1.5);
oscCenter2D = plot(ax2D, NaN, NaN, '+', ...
    'Color', [1.0 0.85 0.3], 'MarkerSize', 6, 'LineWidth', 1.2);
oscText2D = text(ax2D, NaN, NaN, '', 'Color', [1.0 0.85 0.3], 'FontSize', 8);

info2D = text(ax2D, 0.02, 0.97, '', 'Units', 'normalized', ...
    'Color', viz.Colors.Text, 'FontSize', 9, ...
    'VerticalAlignment', 'top', 'FontName', 'Consolas');

% --- RIGHT: 3D ---
ax3D = subplot(1, 2, 2, 'Parent', fig);
viz.styleAxes(ax3D);
hold(ax3D, 'on');
view(ax3D, 30, 25);
grid(ax3D, 'on');
set(ax3D, 'GridColor', [1 1 1], 'GridAlpha', 0.15);
title(ax3D, '3D Helix: T (red), N (green), B (blue)', ...
    'Color', viz.Colors.Text, 'FontSize', 12, 'FontWeight', 'bold');

% Draw helix curve
tPts3 = curve3D.parameterize(300);
cPts3 = curve3D.evaluate(tPts3);
plot3(ax3D, cPts3(:,1), cPts3(:,2), cPts3(:,3), '-', ...
    'Color', [0.5 0.8 1.0 0.4], 'LineWidth', 2);

% Animated handles: 3D
dot3D = plot3(ax3D, NaN, NaN, NaN, 'o', 'MarkerSize', 10, ...
    'MarkerFaceColor', [0.2 1.0 0.8], 'MarkerEdgeColor', [1 1 1], 'LineWidth', 1.5);
trail3D = plot3(ax3D, NaN, NaN, NaN, '-', 'Color', [0.2 1.0 0.8 0.25], 'LineWidth', 1.5);

% T, N, B quivers (3D)
arrowScale3D = 1.5;
qT3 = quiver3(ax3D, NaN, NaN, NaN, NaN, NaN, NaN, 0, ...
    'Color', [1.0 0.3 0.3], 'LineWidth', 2.5, 'MaxHeadSize', 0.4);
qN3 = quiver3(ax3D, NaN, NaN, NaN, NaN, NaN, NaN, 0, ...
    'Color', [0.3 1.0 0.3], 'LineWidth', 2.5, 'MaxHeadSize', 0.4);
qB3 = quiver3(ax3D, NaN, NaN, NaN, NaN, NaN, NaN, 0, ...
    'Color', [0.3 0.5 1.0], 'LineWidth', 2.5, 'MaxHeadSize', 0.4);
lblT3 = text(ax3D, NaN, NaN, NaN, 'T', 'Color', [1.0 0.3 0.3], ...
    'FontWeight', 'bold', 'FontSize', 10);
lblN3 = text(ax3D, NaN, NaN, NaN, 'N', 'Color', [0.3 1.0 0.3], ...
    'FontWeight', 'bold', 'FontSize', 10);
lblB3 = text(ax3D, NaN, NaN, NaN, 'B', 'Color', [0.3 0.5 1.0], ...
    'FontWeight', 'bold', 'FontSize', 10);

info3D = text(ax3D, 0.02, 0.97, '', 'Units', 'normalized', ...
    'Color', viz.Colors.Text, 'FontSize', 9, ...
    'VerticalAlignment', 'top', 'FontName', 'Consolas');

%% 3. Create simulators
sim2D = OceanMath.Simulation.CurveSimulator(curve2D, ...
    'InitialVelocity', 1.2, 'Acceleration', 0, 'Dt', 0.005, 'Loop', false);
sim3D = OceanMath.Simulation.CurveSimulator(curve3D, ...
    'InitialVelocity', 1.0, 'Acceleration', 0, 'Dt', 0.005, 'Loop', false);

%% 4. Animation loop
fprintf('Animating...\n');
trailX2 = []; trailY2 = [];
trailX3 = []; trailY3 = []; trailZ3 = [];
frameCount = 0;
arrowScale2D = 0.8;
frenet3D = OceanMath.Geometry.FrenetFrame(curve3D);

while (sim2D.getState().Parameter < 1 || sim3D.getState().Parameter < 1) ...
        && frameCount < 100000
    if ~isvalid(fig), break; end

    if sim2D.getState().Parameter < 1, sim2D.step(); end
    if sim3D.getState().Parameter < 1, sim3D.step(); end
    frameCount = frameCount + 1;

    % Record trail
    s2 = sim2D.getState();
    s3 = sim3D.getState();
    trailX2(end+1) = s2.Position(1); trailY2(end+1) = s2.Position(2); %#ok
    trailX3(end+1) = s3.Position(1); trailY3(end+1) = s3.Position(2); %#ok
    trailZ3(end+1) = s3.Position(3); %#ok

    if mod(frameCount, 5) == 0
        % --- Update 2D panel ---
        t2 = s2.Parameter;
        pos2 = s2.Position;
        [Tv2, Nv2, ~] = frenet2D.frame(t2);
        Ts2 = Tv2 * arrowScale2D;
        Ns2 = Nv2 * arrowScale2D;

        set(dot2D, 'XData', pos2(1), 'YData', pos2(2));
        set(trail2D, 'XData', trailX2, 'YData', trailY2);
        set(qT2, 'XData', pos2(1), 'YData', pos2(2), 'UData', Ts2(1), 'VData', Ts2(2));
        set(qN2, 'XData', pos2(1), 'YData', pos2(2), 'UData', Ns2(1), 'VData', Ns2(2));
        set(lblT2, 'Position', [pos2(1)+Ts2(1)*1.2, pos2(2)+Ts2(2)*1.2, 0]);
        set(lblN2, 'Position', [pos2(1)+Ns2(1)*1.2, pos2(2)+Ns2(2)*1.2, 0]);

        % Osculating circle 2D
        kappa2 = frenet2D.curvature(t2);
        if kappa2 > 0.1
            R2 = 1 / kappa2;
            if R2 < 8
                cen2 = pos2 + Nv2 * R2;
                th = linspace(0, 2*pi, 64);
                set(oscCircle2D, 'XData', cen2(1)+R2*cos(th), 'YData', cen2(2)+R2*sin(th));
                set(oscCenter2D, 'XData', cen2(1), 'YData', cen2(2));
                set(oscText2D, 'Position', [cen2(1)+0.15, cen2(2)+0.15, 0], ...
                    'String', sprintf('R=%.2f', R2));
            else
                set(oscCircle2D, 'XData', NaN, 'YData', NaN);
                set(oscCenter2D, 'XData', NaN, 'YData', NaN);
                set(oscText2D, 'String', '');
            end
        else
            set(oscCircle2D, 'XData', NaN, 'YData', NaN);
            set(oscCenter2D, 'XData', NaN, 'YData', NaN);
            set(oscText2D, 'String', '');
        end

        info2D.String = sprintf('t=%.3f  v=%.2f\nkappa=%.4f', ...
            t2, s2.Velocity, kappa2);

        % --- Update 3D panel ---
        t3 = s3.Parameter;
        pos3 = s3.Position;
        [Tv3, Nv3, Bv3] = frenet3D.frame(t3);
        Ts3 = Tv3 * arrowScale3D;
        Ns3 = Nv3 * arrowScale3D;
        Bs3 = Bv3 * arrowScale3D;

        set(dot3D, 'XData', pos3(1), 'YData', pos3(2), 'ZData', pos3(3));
        set(trail3D, 'XData', trailX3, 'YData', trailY3, 'ZData', trailZ3);

        set(qT3, 'XData', pos3(1), 'YData', pos3(2), 'ZData', pos3(3), ...
                  'UData', Ts3(1), 'VData', Ts3(2), 'WData', Ts3(3));
        set(qN3, 'XData', pos3(1), 'YData', pos3(2), 'ZData', pos3(3), ...
                  'UData', Ns3(1), 'VData', Ns3(2), 'WData', Ns3(3));
        set(qB3, 'XData', pos3(1), 'YData', pos3(2), 'ZData', pos3(3), ...
                  'UData', Bs3(1), 'VData', Bs3(2), 'WData', Bs3(3));

        set(lblT3, 'Position', pos3 + Ts3*1.2);
        set(lblN3, 'Position', pos3 + Ns3*1.2);
        set(lblB3, 'Position', pos3 + Bs3*1.2);

        kappa3 = frenet3D.curvature(t3);
        info3D.String = sprintf('t=%.3f  v=%.2f\nkappa=%.4f', ...
            t3, s3.Velocity, kappa3);

        drawnow limitrate;
    end
end

fprintf('\n--- Done ---\n');
fprintf('  2D: t=%.4f, arc=%.4f, steps=%d\n', ...
    sim2D.getState().Parameter, sim2D.getState().ArcDistance, sim2D.getStepCount());
fprintf('  3D: t=%.4f, arc=%.4f, steps=%d\n', ...
    sim3D.getState().Parameter, sim3D.getState().ArcDistance, sim3D.getStepCount());
fprintf('\n=== DEMO COMPLETE ===\n');
