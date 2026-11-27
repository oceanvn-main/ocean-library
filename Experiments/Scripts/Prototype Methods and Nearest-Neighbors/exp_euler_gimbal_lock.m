% EXP_EULER_GIMBAL_LOCK  Explore and visualize 3D Euler rotations and Gimbal Lock.
%
%   Demonstrates why Euler angles (specifically ZYX roll-pitch-yaw) suffer
%   from mathematical singularities (Gimbal Lock) at Pitch = +/- 90 degrees,
%   and compares their path interpolation against smooth unit quaternions (SLERP).
%
%   Task workflow:
%     1. Proves algebraically the loss of a degree of freedom at Pitch = 90.
%     2. Visualizes the alignment of active gimbal axes in 3D.
%     3. Compares linear Euler interpolation vs. Quaternion SLERP through the lock zone.
%
%   Developed for Ocean Library computer vision primitives.

% Clean up workspace
clear;
close all;
clc;

% Initialize visualization engine
viz = OceanViz.Core('Theme', 'Dark');
fprintf('========================================================================\n');
fprintf('         3D EULER ROTATION AND GIMBAL LOCK EXPLORATION EXPERIMENT        \n');
fprintf('========================================================================\n\n');

%% =====================================================================
%% PHASE 1: ALGEBRAIC LOCK VERIFICATION
%% =====================================================================
fprintf('--- PHASE 1: ALGEBRAIC LOCK VERIFICATION ---\n');
fprintf('In ZYX Euler convention: R = Rz(yaw) * Ry(pitch) * Rx(roll).\n');
fprintf('At Pitch = 90 degrees, Yaw and Roll axes align in world space.\n');
fprintf('This causes the rotation matrix to depend only on (Yaw - Roll).\n');
fprintf('Thus, different (Yaw, Roll) pairs with the same difference yield identical matrices.\n\n');

% Set Pitch exactly to 90 degrees
pitch_lock = 90;

% Combination A: Yaw = 45, Roll = 15 (Difference = 30)
yaw_A = 45;
roll_A = 15;
R_A = OceanMath.Geometry.Rotation3D.eulerToMatrix(roll_A, pitch_lock, yaw_A);

% Combination B: Yaw = 30, Roll = 0 (Difference = 30)
yaw_B = 30;
roll_B = 0;
R_B = OceanMath.Geometry.Rotation3D.eulerToMatrix(roll_B, pitch_lock, yaw_B);

% Compute Frobenius norm of difference
frob_diff = norm(R_A - R_B, 'fro');

fprintf('Combination A (Yaw = %.1f, Pitch = %.1f, Roll = %.1f) yields matrix R_A:\n', yaw_A, pitch_lock, roll_A);
disp(R_A);
fprintf('Combination B (Yaw = %.1f, Pitch = %.1f, Roll = %.1f) yields matrix R_B:\n', yaw_B, pitch_lock, roll_B);
disp(R_B);
fprintf('Euclidean (Frobenius) norm difference ||R_A - R_B||_F = %.4e\n', frob_diff);

if frob_diff < 1e-14
    fprintf('SUCCESS: Proved algebraically that different Yaw/Roll angles produce identical rotations.\n');
    fprintf('One degree of freedom is completely lost (Gimbal Lock verified).\n\n');
else
    error('Algebraic verification failed. Rotation matrices differ.');
end


%% =====================================================================
%% PHASE 2: 3D GIMBAL AXIS ALIGNMENT VISUALIZATION
%% =====================================================================
fprintf('--- PHASE 2: VISUALIZING GIMBAL AXIS ALIGNMENT ---\n');
fprintf('Extracting active rotation axes in world space:\n');
fprintf('  Yaw Axis   : Fixed world Z-axis [0; 0; 1]\n');
fprintf('  Pitch Axis : Intermediate Y-axis rotated by Yaw\n');
fprintf('  Roll Axis  : Intermediate X-axis rotated by Yaw and Pitch\n\n');

% Create Figure for Gimbal Axes
fig_axes = viz.createFigure('Gimbal Axes Alignment Exploration');

% Case 2A: Orthogonal (No Lock, Pitch = 30 deg)
ax1 = subplot(1, 2, 1);
viz.styleAxes(ax1);
hold(ax1, 'on');
grid(ax1, 'on');
view(ax1, 3);

yaw_orth = 45; pitch_orth = 30; roll_orth = 20;
plotGimbalAxes(ax1, yaw_orth, pitch_orth, roll_orth, 'Normal State (Pitch = 30 deg)');

% Case 2B: Locked State (Pitch = 90 deg)
ax2 = subplot(1, 2, 2);
viz.styleAxes(ax2);
hold(ax2, 'on');
grid(ax2, 'on');
view(ax2, 3);

yaw_lock = 45; pitch_lock = 90; roll_lock = 20;
plotGimbalAxes(ax2, yaw_lock, pitch_lock, roll_lock, 'Gimbal Lock State (Pitch = 90 deg)');

% Save axes figure
scriptDir = fileparts(mfilename('fullpath'));
resultsDir = fullfile(scriptDir, 'Results');
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end
saveas(fig_axes, fullfile(resultsDir, 'gimbal_axes_alignment.png'));
fprintf('Saved gimbal axes alignment plot to Results/gimbal_axes_alignment.png\n\n');


%% =====================================================================
%% PHASE 3: INTERPOLATION COMPARISON (EULER VS. QUATERNION SLERP)
%% =====================================================================
fprintf('--- PHASE 3: INTERPOLATION COMPARISON THROUGH LOCKED ZONE ---\n');
fprintf('Interpolating through the Gimbal Lock boundary (Pitch passes through 90).\n');
fprintf('  Start state: Yaw = -60, Pitch = 45,  Roll = -45\n');
fprintf('  End state  : Yaw = 60,  Pitch = 135, Roll = 45\n\n');

% Start and End angles
roll1 = -45; pitch1 = 45;  yaw1 = -60;
roll2 = 45;  pitch2 = 135; yaw2 = 60;

% Convert endpoints to quaternions
R_start = OceanMath.Geometry.Rotation3D.eulerToMatrix(roll1, pitch1, yaw1);
R_end = OceanMath.Geometry.Rotation3D.eulerToMatrix(roll2, pitch2, yaw2);
q_start = OceanMath.Geometry.Rotation3D.matrixToQuat(R_start);
q_end = OceanMath.Geometry.Rotation3D.matrixToQuat(R_end);

% Setup interpolation steps
steps = 100;
t_vals = linspace(0, 1, steps);

% Reference vector to trace
v_ref = [0; 0; 1];

pts_euler = zeros(3, steps);
pts_quat = zeros(3, steps);

% Arrays to store angular trajectories
pitch_euler_path = zeros(1, steps);
pitch_quat_path = zeros(1, steps);

for i = 1:steps
    t = t_vals(i);
    
    % 1. Linear Euler Interpolation
    roll_t = roll1 + t * (roll2 - roll1);
    pitch_t = pitch1 + t * (pitch2 - pitch1);
    yaw_t = yaw1 + t * (yaw2 - yaw1);
    
    R_euler = OceanMath.Geometry.Rotation3D.eulerToMatrix(roll_t, pitch_t, yaw_t);
    pts_euler(:, i) = R_euler * v_ref;
    pitch_euler_path(i) = pitch_t;
    
    % 2. Quaternion SLERP Interpolation
    q_t = OceanMath.Geometry.Rotation3D.slerp(q_start, q_end, t);
    R_quat = OceanMath.Geometry.Rotation3D.quatToMatrix(q_t);
    pts_quat(:, i) = R_quat * v_ref;
    
    % Recover equivalent pitch from quaternion rotation matrix for comparison
    [~, p_recovered, ~] = OceanMath.Geometry.Rotation3D.matrixToEuler(R_quat);
    pitch_quat_path(i) = p_recovered;
end

% Create Figure for Interpolation Paths
fig_interp = viz.createFigure('Interpolation Trajectory Exploration');

% Subplot 1: 3D Trajectory Trace on Unit Sphere
ax_traj = subplot(1, 2, 1);
viz.styleAxes(ax_traj);
hold(ax_traj, 'on');
grid(ax_traj, 'on');
view(ax_traj, 35, 30);

% Render reference unit sphere mesh for context
[Xs, Ys, Zs] = sphere(30);
surf(ax_traj, Xs, Ys, Zs, 'FaceColor', [0.15, 0.15, 0.25], 'EdgeColor', [0.25, 0.25, 0.35], ...
     'FaceAlpha', 0.15, 'HandleVisibility', 'off');

% Plot Euler trajectory (Red/Magenta)
plot3(ax_traj, pts_euler(1, :), pts_euler(2, :), pts_euler(3, :), ...
      'Color', [0.95, 0.1, 0.4], 'LineWidth', 2.5, 'DisplayName', 'Euler Linear Interpolation');

% Plot Quaternion SLERP trajectory (Vibrant Green)
plot3(ax_traj, pts_quat(1, :), pts_quat(2, :), pts_quat(3, :), ...
      'Color', [0.1, 0.85, 0.3], 'LineWidth', 2.5, 'DisplayName', 'Quaternion SLERP');

% Plot endpoints
plot3(ax_traj, pts_euler(1,1), pts_euler(2,1), pts_euler(3,1), 'o', ...
      'MarkerFaceColor', [0.9, 0.9, 0.9], 'MarkerEdgeColor', [0.1, 0.1, 0.1], ...
      'MarkerSize', 8, 'DisplayName', 'Start State');
plot3(ax_traj, pts_euler(1,end), pts_euler(2,end), pts_euler(3,end), 'h', ...
      'MarkerFaceColor', [0.1, 0.6, 0.9], 'MarkerEdgeColor', [0.1, 0.1, 0.1], ...
      'MarkerSize', 10, 'DisplayName', 'End State');

title(ax_traj, '3D Vector Trajectory on Sphere', 'FontSize', 12, 'Color', 'white');
xlabel(ax_traj, 'X-axis', 'Color', 'white');
ylabel(ax_traj, 'Y-axis', 'Color', 'white');
zlabel(ax_traj, 'Z-axis', 'Color', 'white');
legend(ax_traj, 'TextColor', 'white', 'Location', 'southoutside');
axis(ax_traj, 'equal');

% Subplot 2: Pitch Angle Trajectory Comparison
ax_angle = subplot(1, 2, 2);
viz.styleAxes(ax_angle);
hold(ax_angle, 'on');
grid(ax_angle, 'on');

plot(ax_angle, t_vals * 100, pitch_euler_path, 'Color', [0.95, 0.1, 0.4], 'LineWidth', 2, 'DisplayName', 'Euler Angle Pitch');
plot(ax_angle, t_vals * 100, pitch_quat_path, 'Color', [0.1, 0.85, 0.3], 'LineWidth', 2, 'DisplayName', 'SLERP Recovered Pitch');
yline(ax_angle, 90, '--', 'Color', [0.8, 0.8, 0.3], 'LineWidth', 1.5, 'DisplayName', 'Gimbal Lock Boundary');

title(ax_angle, 'Pitch Angle Transition Curve', 'FontSize', 12, 'Color', 'white');
xlabel(ax_angle, 'Interpolation Progress (percent)', 'Color', 'white');
ylabel(ax_angle, 'Pitch Angle (degrees)', 'Color', 'white');
legend(ax_angle, 'TextColor', 'white', 'Location', 'south');

% Save interpolation figure
saveas(fig_interp, fullfile(resultsDir, 'gimbal_interpolation_path.png'));
fprintf('Saved trajectory interpolation plot to Results/gimbal_interpolation_path.png\n');
fprintf('========================================================================\n\n');


%% =====================================================================
%% SUPPORTING UTILITY FUNCTIONS
%% =====================================================================
function plotGimbalAxes(ax, yaw, pitch, roll, plotTitle)
    % PLOTGIMBALAXES  Helper to draw world coordinate intermediate gimbals.
    
    r = deg2rad(roll);
    p = deg2rad(pitch);
    y = deg2rad(yaw);
    
    % Rotation steps
    Rz = [cos(y), -sin(y), 0; sin(y), cos(y), 0; 0, 0, 1];
    Ry = [cos(p), 0, sin(p); 0, 1, 0; -sin(p), 0, cos(p)];
    
    % 1. Gimbal 1 (Yaw Axis): Rotates around world Z axis
    yaw_axis = [0; 0; 1];
    
    % 2. Gimbal 2 (Pitch Axis): Intermediate Y-axis (rotated by Yaw Rz)
    pitch_axis = Rz * [0; 1; 0];
    
    % 3. Gimbal 3 (Roll Axis): Intermediate X-axis (rotated by Yaw and Pitch Rz * Ry)
    roll_axis = Rz * Ry * [1; 0; 0];
    
    % Define origin
    org = [0; 0; 0];
    
    % Plot the world frame base coordinates as thin dotted lines
    plot3(ax, [-1.2, 1.2], [0, 0], [0, 0], ':', 'Color', [0.4, 0.4, 0.4], 'HandleVisibility', 'off');
    plot3(ax, [0, 0], [-1.2, 1.2], [0, 0], ':', 'Color', [0.4, 0.4, 0.4], 'HandleVisibility', 'off');
    plot3(ax, [0, 0], [0, 0], [-1.2, 1.2], ':', 'Color', [0.4, 0.4, 0.4], 'HandleVisibility', 'off');
    
    % Plot intermediate gimbal vectors as heavy solid lines
    % Yaw (Z-gimbal) -> Blue
    quiver3(ax, org(1), org(2), org(3), yaw_axis(1), yaw_axis(2), yaw_axis(3), 0, ...
            'Color', [0.1, 0.5, 0.95], 'LineWidth', 3, 'MaxHeadSize', 0.5, 'DisplayName', 'Yaw Axis (World Z)');
            
    % Pitch (Y-gimbal) -> Green
    quiver3(ax, org(1), org(2), org(3), pitch_axis(1), pitch_axis(2), pitch_axis(3), 0, ...
            'Color', [0.1, 0.85, 0.3], 'LineWidth', 3, 'MaxHeadSize', 0.5, 'DisplayName', 'Pitch Axis (Gimbal Y)');
            
    % Roll (X-gimbal) -> Red/Orange
    quiver3(ax, org(1), org(2), org(3), roll_axis(1), roll_axis(2), roll_axis(3), 0, ...
            'Color', [0.95, 0.35, 0.1], 'LineWidth', 3, 'MaxHeadSize', 0.5, 'DisplayName', 'Roll Axis (Gimbal X)');
            
    % Add text markers at endpoints
    text(ax, yaw_axis(1)*1.1, yaw_axis(2)*1.1, yaw_axis(3)*1.1, 'Z_y', 'Color', [0.1, 0.5, 0.95], 'FontWeight', 'bold');
    text(ax, pitch_axis(1)*1.1, pitch_axis(2)*1.1, pitch_axis(3)*1.1, 'Y_p', 'Color', [0.1, 0.85, 0.3], 'FontWeight', 'bold');
    text(ax, roll_axis(1)*1.1, roll_axis(2)*1.1, roll_axis(3)*1.1, 'X_r', 'Color', [0.95, 0.35, 0.1], 'FontWeight', 'bold');
    
    title(ax, plotTitle, 'FontSize', 12, 'Color', 'white');
    xlabel(ax, 'World X', 'Color', 'white');
    ylabel(ax, 'World Y', 'Color', 'white');
    zlabel(ax, 'World Z', 'Color', 'white');
    legend(ax, 'TextColor', 'white', 'Location', 'southoutside');
    axis(ax, 'equal');
    axis(ax, [-1.3, 1.3, -1.3, 1.3, -1.3, 1.3]);
end
