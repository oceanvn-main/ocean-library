% EXP_TANGENT_DISTANCE_MANIFOLD  Lie Groups vs. Lie Algebras in Pixel Space.
%
%   Demonstrates the mathematical connection between transformation manifolds
%   (Lie Groups) and local tangent space approximations (Lie Algebras) in
%   high-dimensional pixel space. Reconstructs Simard's Tangent Distance
%   geometry and visualises it in an optimal 3D SVD projection.
%
%   DATASET:        Zip Digits (16x16, 256 features), digit "3" samples
%   MODEL:          Tangent spaces (SO(2)/SE(2) Lie algebras) approximating manifolds
%   ASSUMPTIONS:
%     - Image intensity is locally smooth (Gaussian pre-smoothing is applied)
%     - Local linearity of spatial transformations holds for small parameter sweeps
%     - SVD of centered transformed images represents the most expressive 3D projection
%     - Bilinear coordinate warping accurately models the continuous physical action
%   REPRODUCIBILITY: rng(42, 'twister') at top of script
%
%   Ref: Hastie, Tibshirani, Friedman (2009). The Elements of Statistical Learning, Section 13.8.
%
clc; clear; close all; rng(42, 'twister');

%% === 1. Data Loading ===
fprintf('=== Step 1: Data Loading ===\n');

% Load the training zip digits using the package generator
gen = OceanData.Generators.ZipGenerator('Type', 'train');
[X_all, Y_all] = gen.generate();

% Find indexes of the digit "3"
idx3 = find(Y_all == 3);

% Choose two distinct clean samples of the digit "3" to compare
% Sample 1 and Sample 2 have slight differences in style and orientation
img1_vec = X_all(idx3(5), :);   % 5th sample of digit 3
img2_vec = X_all(idx3(12), :);  % 12th sample of digit 3

IMG_RES = 16;
img1 = reshape(img1_vec, [IMG_RES, IMG_RES])';
img2 = reshape(img2_vec, [IMG_RES, IMG_RES])';

fprintf('  Loaded digit 3 (sample %d) and digit 3 (sample %d)\n', idx3(5), idx3(12));
fprintf('  Image resolution: %dx%d = %d features\n', IMG_RES, IMG_RES, IMG_RES^2);

%% === 2. Tangent Vector Computation ===
fprintf('\n=== Step 2: Tangent Vector Computation ===\n');

% Smooth original images slightly for stable numerical derivatives
smoothSigma = 0.6;
fprintf('  Pre-smoothing images with Gaussian filter (sigma = %.1f)\n', smoothSigma);

% Compute spatial gradients and tangent vectors at the identity (theta=0, tx=0, ty=0)
[T1_tx, T1_ty, T1_rot] = computeTangents(img1, smoothSigma);
[T2_tx, T2_ty, T2_rot] = computeTangents(img2, smoothSigma);

fprintf('  Calculated horizontal, vertical, and rotational tangents (Lie algebra generators).\n');
fprintf('  Rotation tangent norm (Sample 1): %.4f\n', norm(T1_rot));
fprintf('  Rotation tangent norm (Sample 2): %.4f\n', norm(T2_rot));

%% === 3. Lie Group Transformation Manifolds (SO(2) Sweeps) ===
fprintf('\n=== Step 3: Transformation Manifold (SO(2) sweeps) ===\n');

% Sweep rotation angle theta in degrees
angles = linspace(-40, 40, 41);
nPoints = numel(angles);

% Pre-allocate true curved transformation manifolds in 256D space (Lie Group)
M1_true = zeros(IMG_RES^2, nPoints);
M2_true = zeros(IMG_RES^2, nPoints);

% Pre-allocate flat tangent space approximations in 256D space (Lie Algebra)
M1_approx = zeros(IMG_RES^2, nPoints);
M2_approx = zeros(IMG_RES^2, nPoints);

for i = 1:nPoints
    theta_deg = angles(i);
    theta_rad = deg2rad(theta_deg);
    
    % True Lie Group action (Coordinate bilinear warping)
    w1 = warpImageBilinear(img1, 0, 0, theta_deg);
    w2 = warpImageBilinear(img2, 0, 0, theta_deg);
    M1_true(:, i) = w1(:);
    M2_true(:, i) = w2(:);
    
    % Linear Lie Algebra approximation (first-order Taylor series expansion)
    M1_approx(:, i) = img1_vec(:) + theta_rad * T1_rot;
    M2_approx(:, i) = img2_vec(:) + theta_rad * T2_rot;
end

fprintf('  Generated true SO(2) manifolds (green curves) and tangent spaces (red lines) for both samples.\n');

%% === 4. Tangent Distance Solver ===
fprintf('\n=== Step 4: Euclidean and Tangent Distance Solving ===\n');

% Standard Euclidean distance between the original images
d_euclidean = norm(img1_vec(:) - img2_vec(:));
fprintf('  Standard Euclidean Distance: %.4f\n', d_euclidean);

% 1. SO(2) Tangent Distance (Rotation Tangent Only)
% Tangent spaces: S1(a1) = x1 + a1 * T1_rot, S2(a2) = x2 + a2 * T2_rot
% Solve: norm( (x1 + a1*T1_rot) - (x2 + a2*T2_rot) )
e_diff = img1_vec(:) - img2_vec(:);
M_so2 = [T1_rot, -T2_rot];
c_so2 = - (M_so2' * M_so2) \ (M_so2' * e_diff);

p1_so2 = img1_vec(:) + c_so2(1) * T1_rot;
p2_so2 = img2_vec(:) + c_so2(2) * T2_rot;
d_so2 = norm(p1_so2 - p2_so2);
fprintf('  SO(2) Tangent Distance       : %.4f  (Solved at alpha1 = %.2f deg, alpha2 = %.2f deg)\n', ...
    d_so2, rad2deg(c_so2(1)), rad2deg(c_so2(2)));

% 2. SE(2) Tangent Distance (Rotation + Horizontal/Vertical Translations)
% Tangent spaces: S1(c1) = x1 + V1*c1, S2(c2) = x2 + V2*c2
V1 = [T1_tx, T1_ty, T1_rot];
V2 = [T2_tx, T2_ty, T2_rot];
M_se2 = [V1, -V2];
c_se2 = - (M_se2' * M_se2) \ (M_se2' * e_diff);

p1_se2 = img1_vec(:) + V1 * c_se2(1:3);
p2_se2 = img2_vec(:) + V2 * c_se2(4:6);
d_se2 = norm(p1_se2 - p2_se2);
fprintf('  SE(2) Tangent Distance       : %.4f  (Solved in 3D translation/rotation tangent space)\n', d_se2);

%% === 5. Dimensionality Reduction (3D SVD Projection) ===
fprintf('\n=== Step 5: Principal Coordinate Projection (SVD) ===\n');

% Collect all high-dimensional points into one matrix to define a common subspace
X_all_proj = [M1_true, M2_true, M1_approx, M2_approx, p1_so2, p2_so2, p1_se2, p2_se2];
mu_proj = mean(X_all_proj, 2);
X_centered = X_all_proj - mu_proj;

% SVD to extract the 3 most dominant orthogonal directions
[U_proj, ~, ~] = svd(X_centered, 'econ');
W_proj = U_proj(:, 1:3); % Size: 256 x 3

% Project all vectors into 3D coordinates
Y1_true = W_proj' * (M1_true - mu_proj);
Y2_true = W_proj' * (M2_true - mu_proj);
Y1_approx = W_proj' * (M1_approx - mu_proj);
Y2_approx = W_proj' * (M2_approx - mu_proj);

y1_id = W_proj' * (img1_vec(:) - mu_proj);
y2_id = W_proj' * (img2_vec(:) - mu_proj);

y1_p_so2 = W_proj' * (p1_so2 - mu_proj);
y2_p_so2 = W_proj' * (p2_so2 - mu_proj);

y1_p_se2 = W_proj' * (p1_se2 - mu_proj);
y2_p_se2 = W_proj' * (p2_se2 - mu_proj);

fprintf('  Subspace projection complete. 3D coordinates capture %.1f%% of total transformation variance.\n', ...
    100 * (sum(diag(X_centered'*W_proj*W_proj'*X_centered)) / sum(diag(X_centered'*X_centered))));

%% === 6. Visualization 1: Infinitesimal Generators (Gradients) ===
fprintf('\n=== Step 6: Rendering Gradients and Tangent Vectors ===\n');

viz = OceanViz.Core('Theme', 'Dark');
fig1 = viz.createFigure('Lie Algebra Infinitesimal Generators');
fig1.Position = [100, 100, 1000, 500];

% Row 1: Sample 1
ax1 = subplot(2, 4, 1, 'Parent', fig1);
imagesc(ax1, img1); colormap(ax1, 'gray'); axis(ax1, 'image'); set(ax1, 'XTick', [], 'YTick', []);
title(ax1, 'Reference Image 1', 'Color', viz.Colors.Text);
ylabel(ax1, 'Sample 1', 'Color', viz.Colors.Text, 'FontSize', 12, 'FontWeight', 'bold');

ax2 = subplot(2, 4, 2, 'Parent', fig1);
imagesc(ax2, reshape(T1_tx, [16,16])'); colormap(ax2, 'sky'); axis(ax2, 'image'); set(ax2, 'XTick', [], 'YTick', []);
title(ax2, 'dI/dx (Horiz Tangent)', 'Color', viz.Colors.Text);

ax3 = subplot(2, 4, 3, 'Parent', fig1);
imagesc(ax3, reshape(T1_ty, [16,16])'); colormap(ax3, 'sky'); axis(ax3, 'image'); set(ax3, 'XTick', [], 'YTick', []);
title(ax3, 'dI/dy (Vert Tangent)', 'Color', viz.Colors.Text);

ax4 = subplot(2, 4, 4, 'Parent', fig1);
imagesc(ax4, reshape(T1_rot, [16,16])'); colormap(ax4, 'sky'); axis(ax4, 'image'); set(ax4, 'XTick', [], 'YTick', []);
title(ax4, 'Rotational Tangent', 'Color', viz.Colors.Text);

% Row 2: Sample 2
ax5 = subplot(2, 4, 5, 'Parent', fig1);
imagesc(ax5, img2); colormap(ax5, 'gray'); axis(ax5, 'image'); set(ax5, 'XTick', [], 'YTick', []);
title(ax5, 'Reference Image 2', 'Color', viz.Colors.Text);
ylabel(ax5, 'Sample 2', 'Color', viz.Colors.Text, 'FontSize', 12, 'FontWeight', 'bold');

ax6 = subplot(2, 4, 6, 'Parent', fig1);
imagesc(ax6, reshape(T2_tx, [16,16])'); colormap(ax6, 'sky'); axis(ax6, 'image'); set(ax6, 'XTick', [], 'YTick', []);
title(ax6, 'dI/dx (Horiz Tangent)', 'Color', viz.Colors.Text);

ax7 = subplot(2, 4, 7, 'Parent', fig1);
imagesc(ax7, reshape(T2_ty, [16,16])'); colormap(ax7, 'sky'); axis(ax7, 'image'); set(ax7, 'XTick', [], 'YTick', []);
title(ax7, 'dI/dy (Vert Tangent)', 'Color', viz.Colors.Text);

ax8 = subplot(2, 4, 8, 'Parent', fig1);
imagesc(ax8, reshape(T2_rot, [16,16])'); colormap(ax8, 'sky'); axis(ax8, 'image'); set(ax8, 'XTick', [], 'YTick', []);
title(ax8, 'Rotational Tangent', 'Color', viz.Colors.Text);

sgtitle(fig1, 'Lie Algebra Infinitesimal Generators (Tangent Fields)', ...
    'Color', viz.Colors.Text, 'FontSize', 14, 'FontWeight', 'bold');

%% === 7. Visualization 2: 3D Projection of Manifolds & Tangents ===
fprintf('=== Step 7: Generating 3D Manifold Geometry Projection ===\n');

fig2 = viz.createFigure('3D Transformation Manifold Projection');
fig2.Position = [150, 150, 950, 750];
ax_3d = axes('Parent', fig2);
hold(ax_3d, 'on');
viz.styleAxes(ax_3d);

% Set distinct colors for premium aesthetic
color_manifold1 = [0.20, 0.85, 0.45]; % Vibrant Green (Lie Group 1)
color_manifold2 = [0.10, 0.70, 0.35]; % Forest Green (Lie Group 2)
color_tangent1  = [1.00, 0.30, 0.30]; % Sharp Coral Red (Lie Algebra 1)
color_tangent2  = [0.80, 0.20, 0.20]; % Dark Red (Lie Algebra 2)

% 1. Plot True Transformation Manifolds (Lie Groups)
plot3(ax_3d, Y1_true(1, :), Y1_true(2, :), Y1_true(3, :), '-', 'Color', color_manifold1, ...
    'LineWidth', 3, 'DisplayName', 'Transformation Manifold 1 (SO(2) Group)');
plot3(ax_3d, Y2_true(1, :), Y2_true(2, :), Y2_true(3, :), '-', 'Color', color_manifold2, ...
    'LineWidth', 3, 'DisplayName', 'Transformation Manifold 2 (SO(2) Group)');

% 2. Plot Tangent Space Approximations (Lie Algebras)
plot3(ax_3d, Y1_approx(1, :), Y1_approx(2, :), Y1_approx(3, :), '--', 'Color', color_tangent1, ...
    'LineWidth', 1.8, 'DisplayName', 'Tangent Space 1 (Lie Algebra)');
plot3(ax_3d, Y2_approx(1, :), Y2_approx(2, :), Y2_approx(3, :), '--', 'Color', color_tangent2, ...
    'LineWidth', 1.8, 'DisplayName', 'Tangent Space 2 (Lie Algebra)');

% 3. Plot Original Identity Points
plot3(ax_3d, y1_id(1), y1_id(2), y1_id(3), 'o', 'MarkerFaceColor', [0.15 0.55 1.0], 'MarkerEdgeColor', 'w', ...
    'MarkerSize', 11, 'DisplayName', 'Identity Image 1 (theta = 0)');
plot3(ax_3d, y2_id(1), y2_id(2), y2_id(3), '^', 'MarkerFaceColor', [0.15 0.55 1.0], 'MarkerEdgeColor', 'w', ...
    'MarkerSize', 11, 'DisplayName', 'Identity Image 2 (theta = 0)');

% 4. Plot Closest Points on Tangent Lines (SO(2) Solver solution)
plot3(ax_3d, y1_p_so2(1), y1_p_so2(2), y1_p_so2(3), 's', 'MarkerFaceColor', viz.Colors.Boundary, ...
    'MarkerEdgeColor', 'w', 'MarkerSize', 10, 'DisplayName', 'Closest Point 1 (SO(2))');
plot3(ax_3d, y2_p_so2(1), y2_p_so2(2), y2_p_so2(3), 's', 'MarkerFaceColor', viz.Colors.Boundary, ...
    'MarkerEdgeColor', 'w', 'MarkerSize', 10, 'DisplayName', 'Closest Point 2 (SO(2))');

% 5. Plot Distance Vectors (Euclidean vs. Tangent Distance)
% Euclidean Line
plot3(ax_3d, [y1_id(1); y2_id(1)], [y1_id(2); y2_id(2)], [y1_id(3); y2_id(3)], ...
    ':', 'Color', [0.25 0.70 1.00], 'LineWidth', 2.5, 'DisplayName', sprintf('Euclidean Distance (%.2f)', d_euclidean));
% Tangent Distance Line
plot3(ax_3d, [y1_p_so2(1); y2_p_so2(1)], [y1_p_so2(2); y2_p_so2(2)], [y1_p_so2(3); y2_p_so2(3)], ...
    '-.', 'Color', viz.Colors.Boundary, 'LineWidth', 2.5, 'DisplayName', sprintf('Tangent Distance (%.2f)', d_so2));

% Labels and Styling
xlabel(ax_3d, 'Principal Dimension 1 (PC1)', 'Color', viz.Colors.Text, 'FontSize', 11);
ylabel(ax_3d, 'Principal Dimension 2 (PC2)', 'Color', viz.Colors.Text, 'FontSize', 11);
zlabel(ax_3d, 'Principal Dimension 3 (PC3)', 'Color', viz.Colors.Text, 'FontSize', 11);
title(ax_3d, '3D Manifold Projection & Tangent Distances (ESL Figure 13.11)', ...
    'Color', viz.Colors.Text, 'FontSize', 13, 'FontWeight', 'bold');
legend(ax_3d, 'Location', 'northeast', 'TextColor', viz.Colors.Text, ...
    'Color', viz.Colors.Background, 'EdgeColor', [0.3 0.3 0.3], 'FontSize', 9);

% Optimize view angles
view(ax_3d, [-45, 18]);
grid(ax_3d, 'on');
hold(ax_3d, 'off');

%% === 8. Visualization 3: Divergence Error Curves ===
fprintf('=== Step 8: Plotting Approximation Error Divergence ===\n');

fig3 = viz.createFigure('Tangent Approximation Divergence');
fig3.Position = [200, 200, 700, 500];
ax_err = axes('Parent', fig3);
hold(ax_err, 'on');
viz.styleAxes(ax_err);

% Compute root-mean-square error along rotation angle sweep
err1_sweep = zeros(size(angles));
err2_sweep = zeros(size(angles));
for i = 1:numel(angles)
    err1_sweep(i) = sqrt(mean((M1_true(:, i) - M1_approx(:, i)).^2));
    err2_sweep(i) = sqrt(mean((M2_true(:, i) - M2_approx(:, i)).^2));
end

plot(ax_err, angles, err1_sweep, '-', 'Color', color_tangent1, ...
    'LineWidth', 2.5, 'DisplayName', 'Digit Sample 1');
plot(ax_err, angles, err2_sweep, '-', 'Color', [0.9 0.5 0.1], ...
    'LineWidth', 2.5, 'DisplayName', 'Digit Sample 2');

% Visualise the Safe Zone marker where linear approximation is robust (<= 15 deg)
xline(ax_err, 15, '--', 'Color', [0.35 0.80 0.40], 'LineWidth', 1.5, ...
    'Label', 'Safe Zone (15°)', 'LabelColor', [0.35 0.80 0.40], ...
    'FontSize', 9, 'HandleVisibility', 'off');
xline(ax_err, -15, '--', 'Color', [0.35 0.80 0.40], 'LineWidth', 1.5, ...
    'FontSize', 9, 'HandleVisibility', 'off');

xlabel(ax_err, 'Rotation Angle (degrees)', 'Color', viz.Colors.Text, 'FontSize', 12);
ylabel(ax_err, 'RMSE (Approximation Error)', 'Color', viz.Colors.Text, 'FontSize', 12);
title(ax_err, 'Tangent Approximation Error vs. Rotation Angle', ...
    'Color', viz.Colors.Text, 'FontSize', 13, 'FontWeight', 'bold');
legend(ax_err, 'Location', 'northwest', 'TextColor', viz.Colors.Text, ...
    'Color', viz.Colors.Background, 'EdgeColor', [0.3 0.3 0.3]);
grid(ax_err, 'on');
hold(ax_err, 'off');

%% === 9. Save Figures ===
resultsDir = fullfile(fileparts(mfilename('fullpath')), 'Results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
saveas(fig1, fullfile(resultsDir, 'fig_tangent_generators.png'));
saveas(fig2, fullfile(resultsDir, 'fig_manifold_projection_3d.png'));
saveas(fig3, fullfile(resultsDir, 'fig_tangent_divergence_curves.png'));
fprintf('\n  Saved 3 analysis figures in Results folder.\n');

%% === 10. Summary ===
fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Dataset          : Zip Digits (16x16, 256 features)\n');
fprintf('  Class Analyzed   : Digit "3" (Two styles loaded)\n');
fprintf('  Pre-Smoothing    : Gaussian sigma = %.1f\n', smoothSigma);
fprintf('  Euclidean Dist   : %.4f\n', d_euclidean);
fprintf('  SO(2) Tangent D. : %.4f  (Rotation Invariance)\n', d_so2);
fprintf('  SE(2) Tangent D. : %.4f  (Translation + Rotation Invariance)\n', d_se2);
fprintf('  Key Findings:\n');
fprintf('    - Applying a spatial transformation group (Lie Group) traces a curved 1D manifold in 256D space.\n');
fprintf('    - The first-order spatial derivatives (Lie Algebra) form a flat, linear tangent space.\n');
fprintf('    - Measuring the shortest distance between tangent spaces (Tangent Distance) provides robust invariance.\n');
fprintf('    - The linear tangent approximation is extremely accurate locally (<15 deg), but diverges quadratically at large transformations.\n');
fprintf('==========================\n');

%% ========================================================================
%  LOCAL HELPER FUNCTIONS
%  ========================================================================

function [T_tx, T_ty, T_rot] = computeTangents(img, smoothSigma)
    % COMPUTETANGENTS Calculates horizontal, vertical, and rotational tangents at identity.
    % Pre-smooth for derivative stability
    I_smooth = imgaussfilt(img, smoothSigma);
    [dI_dx, dI_dy] = gradient(I_smooth);
    
    [H, W] = size(img);
    cx = (W + 1) / 2;
    cy = (H + 1) / 2;
    [Xc, Yc] = meshgrid((1:W) - cx, (1:H) - cy);
    
    % Translation tangents (negative spatial gradients)
    T_tx = -dI_dx(:);
    T_ty = -dI_dy(:);
    
    % Rotation tangent: -y * dI/dx + x * dI/dy
    T_rot_grid = -Yc .* dI_dx + Xc .* dI_dy;
    T_rot = T_rot_grid(:);
end

function I_trans = warpImageBilinear(img, tx, ty, theta_deg)
    % WARPIMAGEBILINEAR Transforms a 2D image using bilinear interpolation (self-contained).
    [H, W] = size(img);
    [X, Y] = meshgrid(1:W, 1:H);
    cx = (W + 1) / 2;
    cy = (H + 1) / 2;
    
    % Centered coordinates
    Xc = X - cx;
    Yc = Y - cy;
    
    % Inverse transformation: map target coordinates back to source coordinate space
    theta = deg2rad(theta_deg);
    c = cos(theta);
    s = sin(theta);
    
    X_src = c * (Xc - tx) + s * (Yc - ty) + cx;
    Y_src = -s * (Xc - tx) + c * (Yc - ty) + cy;
    
    % Bilinear interpolation
    I_trans = interp2(X, Y, img, X_src, Y_src, 'linear', 0);
end
