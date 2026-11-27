% EXP_COVECTOR_NORMAL_TRANSFORM  Co-vectors vs. Vectors (The Inverse Transpose).
%
%   Demonstrates why geometric co-vectors (like line equations in 2D or plane 
%   normals in 3D) must be transformed using the transposed inverse (H^-T) 
%   of the point transformation matrix, rather than the matrix itself (H).
%   Validates this concept algebraically and visually under a non-orthogonal
%   shearing and scaling transformation.
%
%   DATASET:        Synthetically generated 2D lines and 3D surface meshes
%   MODEL:          Algebraic co-vector transformation l' = H^-T * l
%   ASSUMPTIONS:
%     - Projective linearity: Geometric relations remain invariant under coordinate transformations.
%     - Matrix invertibility: The spatial transformation matrix H is non-singular.
%   REPRODUCIBILITY: rng(42, 'twister') at top of script
%
%   Ref: Hartley & Zisserman (2004). Multiple View Geometry in Computer Vision, Chapter 2.
%
clc; clear; close all; rng(42, 'twister');

%% === 1. Setup 2D Points and Line ===
fprintf('=== Step 1: 2D Points and Line Setup ===\n');

% Define a 2D line equation in homogeneous coordinates: x - y = 0
% l = [a; b; c] representing the algebraic boundary: a*x + b*y + c = 0
l = [1; -1; 0]; 

% Generate points lying exactly on this line (x = y)
x_vals = linspace(-2, 2, 11);
y_vals = x_vals;
% Convert to homogeneous points: X = [x; y; 1] (size: 3 x N)
X = [x_vals; y_vals; ones(1, numel(x_vals))];

% Verify the starting algebraic intersection state: l^T * X = 0
algebraic_start_error = norm(l' * X);
fprintf('  Original line co-vector l: [%s]''\n', num2str(l'));
fprintf('  Homogeneous intersection error (l^T * X) at start: %.4e (Exact Zero)\n', algebraic_start_error);

%% === 2. Define Non-Orthogonal 2D Transformation ===
fprintf('\n=== Step 2: Non-Orthogonal 2D Transformation ===\n');

% Define a homography H containing shearing and non-uniform scaling
% H = Scale * Shear
% S = [1.8, 0, 0; 0, 0.7, 0; 0, 0, 1] (scale)
% Sh = [1.0, 1.3, 0; 0, 1.0, 0; 0, 0, 1] (horizontal shear)
H = [1.8, 2.34, 0; ...
     0,   0.7,  0; ...
     0,   0,    1];

fprintf('  Transformation matrix H (non-uniform scale + shear):\n');
disp(H);

% Transform the point coordinates: X' = H * X
X_warped = H * X;

%% === 3. Transform 2D Line (Naive vs. Transposed Inverse) ===
fprintf('\n=== Step 3: Transforming 2D Line Equations ===\n');

% Naive Method: treating the line co-vector as a standard point vector
l_naive = H * l;

% Correct Method: using the Transposed Inverse of the point matrix (H^-T)
H_inv_trans = inv(H)';
l_correct = H_inv_trans * l;

% Normalize line vectors for comparative analysis (making the first coefficient = 1)
l_naive_norm = l_naive / l_naive(1);
l_correct_norm = l_correct / l_correct(1);

fprintf('  Naive transformed line l_naive:   [%s]''\n', num2str(l_naive_norm'));
fprintf('  Correct transformed line l_correct: [%s]''\n', num2str(l_correct_norm'));

%% === 4. Verify 2D Algebraic Intersection Invariance ===
fprintf('\n=== Step 4: Algebraic Intersection Verification ===\n');

% For correct transformation, the warped points must lie on the new line: (l_correct)^T * X' = 0
naive_intersection_error = norm(l_naive' * X_warped);
correct_intersection_error = norm(l_correct' * X_warped);

fprintf('  Algebraic error using Naive line (l_naive^T * X_warped)     : %.4f (Failed! Points do not lie on line)\n', ...
    naive_intersection_error);
fprintf('  Algebraic error using Correct line (l_correct^T * X_warped) : %.4e (Passed! Perfect intersection preserved)\n', ...
    correct_intersection_error);

%% === 5. Setup 3D Mesh and Analytic Vertex Normals ===
fprintf('\n=== Step 5: 3D Paraboloid Mesh and Analytic Normals Setup ===\n');

% Create a mesh of points on a 3D elliptic paraboloid: z = -(x^2 + y^2)
u = linspace(-1.5, 1.5, 15);
v = linspace(-1.5, 1.5, 15);
[U_grid, V_grid] = meshgrid(u, v);

% Coordinate equations for each mesh vertex
X_3d = U_grid;
Y_3d = V_grid;
Z_3d = -(X_3d.^2 + Y_3d.^2);

% Flatten vertices for transformation [3 x N]
vertices = [X_3d(:)'; Y_3d(:)'; Z_3d(:)'];
nVertices = size(vertices, 2);

% Analytical tangent vectors (partial derivatives of surface equation)
% Tx = [1; 0; -2x], Ty = [0; 1; -2y]
T_x = [ones(1, nVertices); zeros(1, nVertices); -2 * vertices(1, :)];
T_y = [zeros(1, nVertices); ones(1, nVertices); -2 * vertices(2, :)];

% Analytical unit normal vectors: N = Tx x Ty = [2x; 2y; 1]
normals = [2 * vertices(1, :); 2 * vertices(2, :); ones(1, nVertices)];
% Normalize to unit vectors
normals_unit = normals ./ sqrt(sum(normals.^2, 1));

fprintf('  Mesh constructed with %d vertices.\n', nVertices);
fprintf('  Verified starting orthogonality (Normals ^T Tangents): %.4e (Exact Zero)\n', ...
    norm(sum(normals_unit .* T_x, 1)) + norm(sum(normals_unit .* T_y, 1)));

%% === 6. Define 3D Non-Orthogonal Warp and Transform Geometry ===
fprintf('\n=== Step 6: 3D Warp and Mesh Transformation ===\n');

% Define a 3x3 non-orthogonal transformation matrix (shearing and non-uniform scaling)
H_3d = [1.5, 1.0, 0.0; ...
        0.0, 0.8, 0.0; ...
        0.5, 0.0, 1.2];

fprintf('  3D transformation matrix H_3d:\n');
disp(H_3d);

% Transform the mesh vertices: V' = H_3d * V
vertices_warped = H_3d * vertices;

% The transformed surface tangent vectors warp directly using H_3d (by the chain rule)
T_x_warped = H_3d * T_x;
T_y_warped = H_3d * T_y;

%% === 7. Transform 3D Surface Normals (Naive vs. Transposed Inverse) ===
fprintf('\n=== Step 7: Transforming 3D Surface Normals ===\n');

% Naive Method: transform using standard matrix multiplication and normalize
normals_naive = H_3d * normals_unit;
normals_naive_unit = normals_naive ./ sqrt(sum(normals_naive.^2, 1));

% Correct Method: transform using the Transposed Inverse of H_3d and normalize
H_3d_inv_trans = inv(H_3d)';
normals_correct = H_3d_inv_trans * normals_unit;
normals_correct_unit = normals_correct ./ sqrt(sum(normals_correct.^2, 1));

% Verify perpendicularity on the deformed surface
% The normals must remain orthogonal to both warped surface tangents Tx' and Ty'
% Calculate dot products (should be strictly zero for the correct method)
err_naive_Tx = sum(normals_naive_unit .* T_x_warped, 1);
err_naive_Ty = sum(normals_naive_unit .* T_y_warped, 1);
err_correct_Tx = sum(normals_correct_unit .* T_x_warped, 1);
err_correct_Ty = sum(normals_correct_unit .* T_y_warped, 1);

mean_err_naive = mean(abs(err_naive_Tx) + abs(err_naive_Ty));
mean_err_correct = mean(abs(err_correct_Tx) + abs(err_correct_Ty));

fprintf('  Mean orthogonality error using Naive normals (H_3d * N)      : %.4f (Failed! Normals not perpendicular)\n', ...
    mean_err_naive);
fprintf('  Mean orthogonality error using Correct normals (H_3d^-T * N) : %.4e (Passed! Normals strictly perpendicular)\n', ...
    mean_err_correct);

%% === 8. Visualization 1: 2D Line Transformation Breakdown ===
fprintf('\n=== Step 8: Visualizing 2D Geometric Breakdown ===\n');

viz = OceanViz.Core('Theme', 'Dark');
fig1 = viz.createFigure('2D Line Transformation Duality');
fig1.Position = [100, 100, 800, 600];
ax1 = axes('Parent', fig1);
hold(ax1, 'on');
viz.styleAxes(ax1);

% Colors for visualization
color_orig = [0.5, 0.5, 0.5];        % Gray
color_pts = viz.Colors.Primary;      % Blue
color_naive = viz.Colors.Secondary;  % Red/Orange
color_correct = [0.2, 0.8, 0.4];     % Green

% Plot Original Points and Line
plot(ax1, x_vals, y_vals, '-', 'Color', color_orig, 'LineWidth', 2, 'DisplayName', 'Original Line (x - y = 0)');
plot(ax1, X(1,:), X(2,:), 'o', 'MarkerFaceColor', color_orig, 'MarkerEdgeColor', 'w', 'MarkerSize', 8, 'HandleVisibility', 'off');

% Plot Transformed Points
plot(ax1, X_warped(1,:), X_warped(2,:), 'o', 'MarkerFaceColor', color_pts, 'MarkerEdgeColor', 'w', ...
    'MarkerSize', 9, 'DisplayName', 'Transformed Points (H * x)');

% Plot Naive Line: l_naive(1)*x + l_naive(2)*y + l_naive(3) = 0
% Express y as a function of x: y = -(l_naive(1)*x + l_naive(3)) / l_naive(2)
x_plot = linspace(min(X_warped(1,:))-1, max(X_warped(1,:))+1, 100);
y_naive_plot = -(l_naive(1)*x_plot + l_naive(3)) / l_naive(2);
plot(ax1, x_plot, y_naive_plot, '--', 'Color', color_naive, 'LineWidth', 2.5, ...
    'DisplayName', sprintf('Naive Transformed Line (H * l) [Err=%.2f]', naive_intersection_error));

% Plot Correct Line (Transposed Inverse)
y_correct_plot = -(l_correct(1)*x_plot + l_correct(3)) / l_correct(2);
plot(ax1, x_plot, y_correct_plot, '-', 'Color', color_correct, 'LineWidth', 3, ...
    'DisplayName', sprintf('Correct Line (H^{-T} * l) [Err=%.1e]', correct_intersection_error));

% Add annotation showing detachment
text(ax1, mean(x_plot)-0.8, mean(y_naive_plot)+0.8, 'Naive Line Detached!', ...
    'Color', color_naive, 'FontSize', 11, 'FontWeight', 'bold');

xlabel(ax1, 'X', 'Color', viz.Colors.Text);
ylabel(ax1, 'Y', 'Color', viz.Colors.Text);
title(ax1, '2D Line Transformation: Naive (H * l) vs. Correct (H^{-T} * l)', ...
    'Color', viz.Colors.Text, 'FontSize', 13, 'FontWeight', 'bold');
legend(ax1, 'Location', 'northwest', 'TextColor', viz.Colors.Text, ...
    'Color', viz.Colors.Background, 'EdgeColor', [0.3 0.3 0.3]);
axis(ax1, 'equal');
grid(ax1, 'on');
hold(ax1, 'off');

%% === 9. Visualization 2: 3D Surface Normal Breakdown ===
fprintf('=== Step 9: Visualizing 3D Surface Normal Breakdown ===\n');

fig2 = viz.createFigure('3D Normal Transformation');
fig2.Position = [150, 150, 1000, 700];

% Subplot 1: Naive Normal transformation
ax3d_naive = subplot(1, 2, 1, 'Parent', fig2);
hold(ax3d_naive, 'on');
viz.styleAxes(ax3d_naive);

% Reshape warped vertices back to grid for surface plotting
X_w_grid = reshape(vertices_warped(1,:), size(X_3d));
Y_w_grid = reshape(vertices_warped(2,:), size(Y_3d));
Z_w_grid = reshape(vertices_warped(3,:), size(Z_3d));

% Draw sheared mesh surface
surf(ax3d_naive, X_w_grid, Y_w_grid, Z_w_grid, 'FaceColor', [0.15, 0.18, 0.25], ...
    'EdgeColor', [0.4, 0.4, 0.5], 'FaceAlpha', 0.85);

% Plot Naive normals (Red vectors) at a subset of grid vertices for clarity
skip = 2;
idx_vec = 1:skip:nVertices;
quiver3(ax3d_naive, vertices_warped(1, idx_vec), vertices_warped(2, idx_vec), vertices_warped(3, idx_vec), ...
    normals_naive_unit(1, idx_vec), normals_naive_unit(2, idx_vec), normals_naive_unit(3, idx_vec), ...
    0.6, 'Color', color_naive, 'LineWidth', 2);

xlabel(ax3d_naive, 'X', 'Color', viz.Colors.Text);
ylabel(ax3d_naive, 'Y', 'Color', viz.Colors.Text);
zlabel(ax3d_naive, 'Z', 'Color', viz.Colors.Text);
title(ax3d_naive, sprintf('Naive Normals (H * N) [Ortho Err=%.3f]', mean_err_naive), ...
    'Color', color_naive, 'FontSize', 12, 'FontWeight', 'bold');
view(ax3d_naive, [-38, 24]);
grid(ax3d_naive, 'on');
hold(ax3d_naive, 'off');

% Subplot 2: Correct Normal transformation (H^-T)
ax3d_correct = subplot(1, 2, 2, 'Parent', fig2);
hold(ax3d_correct, 'on');
viz.styleAxes(ax3d_correct);

% Draw sheared mesh surface
surf(ax3d_correct, X_w_grid, Y_w_grid, Z_w_grid, 'FaceColor', [0.15, 0.18, 0.25], ...
    'EdgeColor', [0.4, 0.4, 0.5], 'FaceAlpha', 0.85);

% Plot Correct normals (Green vectors)
quiver3(ax3d_correct, vertices_warped(1, idx_vec), vertices_warped(2, idx_vec), vertices_warped(3, idx_vec), ...
    normals_correct_unit(1, idx_vec), normals_correct_unit(2, idx_vec), normals_correct_unit(3, idx_vec), ...
    0.6, 'Color', color_correct, 'LineWidth', 2.5);

xlabel(ax3d_correct, 'X', 'Color', viz.Colors.Text);
ylabel(ax3d_correct, 'Y', 'Color', viz.Colors.Text);
zlabel(ax3d_correct, 'Z', 'Color', viz.Colors.Text);
title(ax3d_correct, sprintf('Correct Normals (H^{-T} * N) [Ortho Err=%.1e]', mean_err_correct), ...
    'Color', color_correct, 'FontSize', 12, 'FontWeight', 'bold');
view(ax3d_correct, [-38, 24]);
grid(ax3d_correct, 'on');
hold(ax3d_correct, 'off');

sgtitle(fig2, '3D Mesh Transformation: Naive (H * N) vs. Correct (H^{-T} * N) Normals', ...
    'Color', viz.Colors.Text, 'FontSize', 14, 'FontWeight', 'bold');

%% === 10. Save Figures ===
resultsDir = fullfile(fileparts(mfilename('fullpath')), 'Results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
saveas(fig1, fullfile(resultsDir, 'fig_covector_2d_line.png'));
saveas(fig2, fullfile(resultsDir, 'fig_covector_3d_mesh.png'));
fprintf('\n  Saved 2 analysis figures in Results folder.\n');

%% === 11. Summary ===
fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Concepts Tested     : Standard Vector (Points) vs. Co-vector (Lines & Normals)\n');
fprintf('  2D Scale & Shear H  :\n');
disp(H);
fprintf('  3D Scale & Shear H  :\n');
disp(H_3d);
fprintf('  Numerical Results:\n');
fprintf('    - 2D Point-Line Intersection Error:\n');
fprintf('        * Naive Method (H * l)           : %.4f  (Intersection Destroyed!)\n', naive_intersection_error);
fprintf('        * Correct Method (H^-T * l)      : %.4e  (Intersection Preserved!)\n', correct_intersection_error);
fprintf('    - 3D Surface Normal Orthogonality Error:\n');
fprintf('        * Naive Method (H * N)           : %.4f  (Normals not perpendicular)\n', mean_err_naive);
fprintf('        * Correct Method (H^-T * N)      : %.4e  (Normals strictly perpendicular)\n', mean_err_correct);
fprintf('  Key Finding:\n');
fprintf('    - Standard points transform via H * x.\n');
fprintf('    - Coordinate boundaries and normals (co-vectors) MUST transform via H^-T * l.\n');
fprintf('    - Under non-orthogonal shear/scale, H^-T preserves physical perpendicularity and intersection properties, whereas standard H fails completely.\n');
fprintf('==========================\n');
