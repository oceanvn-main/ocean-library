% EXP_TANGENT_BREAKDOWN  Tangent Distance — Linear Approximation Breakdown.
%
%   Demonstrates that the Tangent Distance algorithm's linear approximation
%   (first-order Taylor expansion) breaks down catastrophically for large
%   geometric transformations.
%
%   DATASET:        Zip Digits (16x16, 256 features), digits "3" and "8"
%   MODEL:          Tangent vector T = -X*dI/dy + Y*dI/dx (rotation invariance)
%   ASSUMPTIONS:
%     - Image intensity is locally smooth (Gaussian pre-smoothing, sigma=0.5)
%     - Tangent vector is evaluated at the identity (0° rotation)
%     - Linear approximation: I_approx = I_original + theta * T_rotation
%     - True rotation computed via MATLAB imrotate (bicubic interpolation)
%   REPRODUCIBILITY: rng(42, 'twister')
%
%   Part A: Static multi-panel comparison at fixed angles (0°-90°)
%   Part B: Interactive uifigure slider for real-time exploration
%
% Ref: Hastie, Tibshirani, Friedman (2009). ESL, Section 13.8.
clc; clear; close all; rng(42, 'twister');

%% === 1. Data Loading ===
fprintf('=== Step 1: Data Loading ===\n');

gen = OceanData.Generators.ZipGenerator('Type', 'train');
[X_all, Y_all] = gen.generate();

% Pick one clean "3" and one clean "8"
idx3 = find(Y_all == 3);
idx8 = find(Y_all == 8);

% Choose representative samples (not too noisy)
img3_vec = X_all(idx3(5), :);   % 5th sample of digit 3
img8_vec = X_all(idx8(3), :);   % 3rd sample of digit 8

IMG_RES = 16;
img3 = reshape(img3_vec, [IMG_RES, IMG_RES])';
img8 = reshape(img8_vec, [IMG_RES, IMG_RES])';

fprintf('  Loaded digit 3 (sample %d) and digit 8 (sample %d)\n', idx3(5), idx8(3));
fprintf('  Image resolution: %dx%d = %d features\n', IMG_RES, IMG_RES, IMG_RES^2);

%% === 2. Tangent Vector Computation ===
fprintf('\n=== Step 2: Tangent Vector Computation ===\n');

% ROTATION tangent vector: T_rot(i,j) = -x * dI/dy + y * dI/dx
% From theory: dI/dx and dI/dy : How fast is the brightness changing as you move horizontally across the pixels?
% (see computeRotationTangent at bottom of file)
smoothSigma = 0.5;
T3 = computeRotationTangent(img3, smoothSigma);
T8 = computeRotationTangent(img8, smoothSigma);

fprintf('  Gaussian smoothing sigma = %.1f\n', smoothSigma);
fprintf('  Tangent vector norm (digit 3): %.4f\n', norm(T3(:)));
fprintf('  Tangent vector norm (digit 8): %.4f\n', norm(T8(:)));

%% === 3. True Rotation vs. Tangent Approximation ===
fprintf('\n=== Step 3: Computing rotations ===\n');

angles_deg = [0, 5, 15, 30, 45, 90];
nAngles = numel(angles_deg);

% Compute for both digits (see computeAllRotations at bottom of file)
[true3, approx3, diff3, err3] = computeAllRotations(img3, T3, angles_deg);
[true8, approx8, diff8, err8] = computeAllRotations(img8, T8, angles_deg);

for i = 1:nAngles
    fprintf('  %2d°: RMSE(3)=%.4f  RMSE(8)=%.4f\n', angles_deg(i), err3(i), err8(i));
end

%% === 4. Part A: Static Multi-Panel Visualization ===
fprintf('\n=== Step 4: Part A — Static Visualization ===\n');

viz = OceanViz.Core('Theme', 'Dark');

% --- Figure 1: Digit "3" ---
fig1 = viz.createFigure('Tangent Breakdown — Digit 3');
fig1.Position = [30, 100, 1400, 600];

for i = 1:nAngles
    % Row 1: True rotation
    ax = subplot(3, nAngles, i, 'Parent', fig1);
    imagesc(ax, true3(:,:,i)); colormap(ax, 'gray'); clim(ax, [-1 1]);
    set(ax, 'XTick', [], 'YTick', []);
    if i == 1, ylabel(ax, 'True', 'Color', viz.Colors.Text, 'FontSize', 11, 'FontWeight', 'bold'); end
    title(ax, sprintf('%d°', angles_deg(i)), 'Color', viz.Colors.Text, 'FontSize', 11);
    set(ax, 'Color', viz.Colors.Background);
    
    % Row 2: Tangent approximation
    ax = subplot(3, nAngles, nAngles + i, 'Parent', fig1);
    imagesc(ax, approx3(:,:,i)); colormap(ax, 'gray'); clim(ax, [-1 1]);
    set(ax, 'XTick', [], 'YTick', []);
    if i == 1, ylabel(ax, 'Tangent', 'Color', viz.Colors.Text, 'FontSize', 11, 'FontWeight', 'bold'); end
    set(ax, 'Color', viz.Colors.Background);
    
    % Row 3: Difference
    ax = subplot(3, nAngles, 2*nAngles + i, 'Parent', fig1);
    imagesc(ax, diff3(:,:,i)); colormap(ax, 'hot'); clim(ax, [0 1]);
    set(ax, 'XTick', [], 'YTick', []);
    if i == 1, ylabel(ax, '|Diff|', 'Color', viz.Colors.Text, 'FontSize', 11, 'FontWeight', 'bold'); end
    xlabel(ax, sprintf('RMSE=%.3f', err3(i)), 'Color', [1 0.6 0.2], 'FontSize', 9);
    set(ax, 'Color', viz.Colors.Background);
end

sgtitle(fig1, 'Tangent Approximation Breakdown — Digit "3"', ...
    'Color', viz.Colors.Text, 'FontSize', 14, 'FontWeight', 'bold');

% --- Figure 2: Digit "8" ---
fig2 = viz.createFigure('Tangent Breakdown — Digit 8');
fig2.Position = [30, 100, 1400, 600];

for i = 1:nAngles
    ax = subplot(3, nAngles, i, 'Parent', fig2);
    imagesc(ax, true8(:,:,i)); colormap(ax, 'gray'); clim(ax, [-1 1]);
    set(ax, 'XTick', [], 'YTick', []);
    if i == 1, ylabel(ax, 'True', 'Color', viz.Colors.Text, 'FontSize', 11, 'FontWeight', 'bold'); end
    title(ax, sprintf('%d°', angles_deg(i)), 'Color', viz.Colors.Text, 'FontSize', 11);
    set(ax, 'Color', viz.Colors.Background);
    
    ax = subplot(3, nAngles, nAngles + i, 'Parent', fig2);
    imagesc(ax, approx8(:,:,i)); colormap(ax, 'gray'); clim(ax, [-1 1]);
    set(ax, 'XTick', [], 'YTick', []);
    if i == 1, ylabel(ax, 'Tangent', 'Color', viz.Colors.Text, 'FontSize', 11, 'FontWeight', 'bold'); end
    set(ax, 'Color', viz.Colors.Background);
    
    ax = subplot(3, nAngles, 2*nAngles + i, 'Parent', fig2);
    imagesc(ax, diff8(:,:,i)); colormap(ax, 'hot'); clim(ax, [0 1]);
    set(ax, 'XTick', [], 'YTick', []);
    if i == 1, ylabel(ax, '|Diff|', 'Color', viz.Colors.Text, 'FontSize', 11, 'FontWeight', 'bold'); end
    xlabel(ax, sprintf('RMSE=%.3f', err8(i)), 'Color', [1 0.6 0.2], 'FontSize', 9);
    set(ax, 'Color', viz.Colors.Background);
end

sgtitle(fig2, 'Tangent Approximation Breakdown — Digit "8"', ...
    'Color', viz.Colors.Text, 'FontSize', 14, 'FontWeight', 'bold');

% --- Figure 3: Error curves ---
fig3 = viz.createFigure('RMSE vs Rotation Angle');
fig3.Position = [100, 200, 700, 500];
ax3 = axes('Parent', fig3);
hold(ax3, 'on'); viz.styleAxes(ax3);

% Dense sweep for smooth curve
angles_sweep = 0:1:90;
err3_sweep = zeros(size(angles_sweep));
err8_sweep = zeros(size(angles_sweep));
for i = 1:numel(angles_sweep)
    theta_r = angles_sweep(i) * pi / 180;
    I_true3 = imrotate(img3, angles_sweep(i), 'bicubic', 'crop');
    I_true8 = imrotate(img8, angles_sweep(i), 'bicubic', 'crop');
    I_approx3 = img3 + theta_r * T3;
    I_approx8 = img8 + theta_r * T8;
    err3_sweep(i) = sqrt(mean((I_true3(:) - I_approx3(:)).^2));
    err8_sweep(i) = sqrt(mean((I_true8(:) - I_approx8(:)).^2));
end

plot(ax3, angles_sweep, err3_sweep, '-', 'Color', [0.30 0.65 0.95], ...
    'LineWidth', 2.5, 'DisplayName', 'Digit 3');
plot(ax3, angles_sweep, err8_sweep, '-', 'Color', [1.00 0.50 0.15], ...
    'LineWidth', 2.5, 'DisplayName', 'Digit 8');

% Mark safe zone
xline(ax3, 15, '--', 'Color', [0.35 0.80 0.40], 'LineWidth', 1.5, ...
    'Label', 'Safe Zone (15°)', 'LabelColor', [0.35 0.80 0.40], ...
    'FontSize', 9, 'HandleVisibility', 'off');

xlabel(ax3, 'Rotation Angle (degrees)', 'Color', viz.Colors.Text, 'FontSize', 12);
ylabel(ax3, 'RMSE (Approximation Error)', 'Color', viz.Colors.Text, 'FontSize', 12);
title(ax3, 'Tangent Approximation Error vs. Rotation Angle', ...
    'Color', viz.Colors.Text, 'FontSize', 14, 'FontWeight', 'bold');
legend(ax3, 'Location', 'northwest', 'TextColor', viz.Colors.Text, ...
    'Color', viz.Colors.Background, 'EdgeColor', [0.3 0.3 0.3]);
grid(ax3, 'on'); hold(ax3, 'off');

% Save figures
resultsDir = fullfile(fileparts(mfilename('fullpath')), 'Results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
saveas(fig1, fullfile(resultsDir, 'fig_tangent_digit3.png'));
saveas(fig2, fullfile(resultsDir, 'fig_tangent_digit8.png'));
saveas(fig3, fullfile(resultsDir, 'fig_tangent_error_curve.png'));
fprintf('[Part A] Saved 3 figures.\n');

%% === 5. Part B: Interactive Slider App ===
fprintf('\n=== Step 5: Part B — Interactive Slider ===\n');

launchTangentExplorer(img3, T3, img8, T8, viz);
fprintf('[Part B] Interactive explorer launched.\n');

%% === 6. Summary ===
fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Dataset       : Zip Digits (16x16, 256 features)\n');
fprintf('  Digits tested : 3 and 8\n');
fprintf('  Smoothing     : Gaussian sigma=%.1f\n', smoothSigma);
fprintf('  Angles tested : %s degrees\n', mat2str(angles_deg));
fprintf('  Key findings:\n');
fprintf('    RMSE at  5° : %.4f (digit 3), %.4f (digit 8)\n', err3(2), err8(2));
fprintf('    RMSE at 45° : %.4f (digit 3), %.4f (digit 8)\n', err3(5), err8(5));
fprintf('    RMSE at 90° : %.4f (digit 3), %.4f (digit 8)\n', err3(6), err8(6));
fprintf('  Conclusion: Linear approximation valid for <15°, breaks down beyond.\n');
fprintf('==========================\n');

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================

function T = computeRotationTangent(img, smoothSigma)
    % COMPUTEROTATIONTANGENT  Rotation tangent vector at identity.
    %   T = -X * dI/dy + Y * dI/dx
    I_smooth = imgaussfilt(img, smoothSigma); %Apply 2D Gauss kernel to discrete matrix I
    [dI_dx, dI_dy] = gradient(I_smooth);
    
    %Center the anchor
    [cols, rows] = meshgrid(1:size(img,2), 1:size(img,1));
    cx = (size(img,2) + 1) / 2;
    cy = (size(img,1) + 1) / 2;
    X_grid = cols - cx;
    Y_grid = rows - cy;
    
    T = -X_grid .* dI_dy + Y_grid .* dI_dx;
end

function [true_imgs, approx_imgs, diff_imgs, errors] = computeAllRotations(img, T, angles_deg)
    % COMPUTEALLROTATIONS  Compute true vs. tangent approx at multiple angles.
    nA = numel(angles_deg);
    [H, W] = size(img);
    true_imgs   = zeros(H, W, nA);
    approx_imgs = zeros(H, W, nA);
    diff_imgs   = zeros(H, W, nA);
    errors      = zeros(1, nA);
    
    for i = 1:nA
        theta_deg = angles_deg(i);
        theta_rad = theta_deg * pi / 180;
        
        I_true = imrotate(img, theta_deg, 'bicubic', 'crop'); %Matlab built-in. This function perform a highly complex ,non-linear geometric transformation using bicubic interpolation.
        %I_true represents the exact coordinate of the dot as it travels along the curved Transformation Manifold. 
        I_approx = img + theta_rad * T; %Linear Approximation. F(theta) approx ~ F_0  + theta * dF/dtheta
        I_diff = abs(I_true - I_approx);
        
        %Once we have the True Image and the Approximated Image, we need to know exactly how badly the straight-line math "lied" to us compared to the true physics.
        errors(i) = sqrt(mean((I_true(:) - I_approx(:)).^2)); % Root Mean Square (RMS) Euclidean Distance.
        
        true_imgs(:,:,i)   = I_true;
        approx_imgs(:,:,i) = I_approx;
        diff_imgs(:,:,i)   = I_diff;
    end
end

function launchTangentExplorer(img3, T3, img8, T8, viz)
    % LAUNCHTANGENTEXPLORER  Interactive uifigure with rotation slider.
    
    bgColor = viz.Colors.Background;
    txtColor = viz.Colors.Text;
    
    % Create modern uifigure
    app = uifigure('Name', 'Tangent Distance Explorer', ...
        'Position', [150, 100, 1100, 650], ...
        'Color', bgColor);
    
    % --- Layout: Grid ---
    gl = uigridlayout(app, [4, 6], ...
        'BackgroundColor', bgColor, ...
        'RowHeight', {30, '1x', '1x', 60}, ...
        'ColumnWidth', {'1x','1x','1x','1x','1x','1x'});
    
    % --- Row 1: Title + Angle display ---
    titleLbl = uilabel(gl, 'Text', 'Tangent Distance Explorer', ...
        'FontSize', 16, 'FontWeight', 'bold', 'FontColor', txtColor, ...
        'HorizontalAlignment', 'center');
    titleLbl.Layout.Row = 1; titleLbl.Layout.Column = [1, 3];
    
    angleLbl = uilabel(gl, 'Text', 'Angle: 0°  |  RMSE: 0.0000', ...
        'FontSize', 14, 'FontWeight', 'bold', 'FontColor', [1 0.6 0.2], ...
        'HorizontalAlignment', 'center');
    angleLbl.Layout.Row = 1; angleLbl.Layout.Column = [4, 6];
    
    % --- Row 2: Digit 3 panels ---
    
    ax3_true   = uiaxes(gl); ax3_true.Layout.Row = 2;   ax3_true.Layout.Column = 1;
    ax3_approx = uiaxes(gl); ax3_approx.Layout.Row = 2; ax3_approx.Layout.Column = 2;
    ax3_diff   = uiaxes(gl); ax3_diff.Layout.Row = 2;   ax3_diff.Layout.Column = 3;
    
    % --- Row 3: Digit 8 panels ---
    ax8_true   = uiaxes(gl); ax8_true.Layout.Row = 3;   ax8_true.Layout.Column = 1;
    ax8_approx = uiaxes(gl); ax8_approx.Layout.Row = 3; ax8_approx.Layout.Column = 2;
    ax8_diff   = uiaxes(gl); ax8_diff.Layout.Row = 3;   ax8_diff.Layout.Column = 3;
    
    % --- Error curve axes ---
    ax_err = uiaxes(gl); ax_err.Layout.Row = [2, 3]; ax_err.Layout.Column = [4, 6];
    
    % Style all axes
    allAxes = [ax3_true, ax3_approx, ax3_diff, ax8_true, ax8_approx, ax8_diff];
    for a = allAxes
        a.Color = bgColor; a.XColor = 'none'; a.YColor = 'none';
        a.XTick = []; a.YTick = [];
    end
    ax_err.Color = bgColor; ax_err.XColor = txtColor; ax_err.YColor = txtColor;
    
    % Titles
    title(ax3_true, 'True (3)', 'Color', txtColor, 'FontSize', 10);
    title(ax3_approx, 'Tangent (3)', 'Color', txtColor, 'FontSize', 10);
    title(ax3_diff, '|Diff| (3)', 'Color', txtColor, 'FontSize', 10);
    title(ax8_true, 'True (8)', 'Color', txtColor, 'FontSize', 10);
    title(ax8_approx, 'Tangent (8)', 'Color', txtColor, 'FontSize', 10);
    title(ax8_diff, '|Diff| (8)', 'Color', txtColor, 'FontSize', 10);
    
    % Error curve setup
    title(ax_err, 'RMSE vs Angle', 'Color', txtColor, 'FontSize', 12, 'FontWeight', 'bold');
    xlabel(ax_err, 'Angle (°)', 'Color', txtColor);
    ylabel(ax_err, 'RMSE', 'Color', txtColor);
    hold(ax_err, 'on'); grid(ax_err, 'on');
    ax_err.GridColor = [0.3 0.3 0.3];
    
    % Pre-compute error curves
    sweep = 0:1:90;
    rmse3 = arrayfun(@(a) computeRMSE(img3, T3, a), sweep);
    rmse8 = arrayfun(@(a) computeRMSE(img8, T8, a), sweep);
    
    plot(ax_err, sweep, rmse3, '-', 'Color', [0.30 0.65 0.95], ...
        'LineWidth', 2, 'DisplayName', 'Digit 3');
    plot(ax_err, sweep, rmse8, '-', 'Color', [1.00 0.50 0.15], ...
        'LineWidth', 2, 'DisplayName', 'Digit 8');
    xline(ax_err, 15, '--', 'Color', [0.35 0.80 0.40], 'LineWidth', 1.2, ...
        'HandleVisibility', 'off');
    legend(ax_err, 'TextColor', txtColor, 'Color', bgColor, ...
        'EdgeColor', [0.3 0.3 0.3], 'Location', 'northwest');
    
    % Current angle marker
    markerLine = xline(ax_err, 0, '-', 'Color', [1 0.3 0.3], 'LineWidth', 2, ...
        'HandleVisibility', 'off');
    
    % --- Row 4: Slider ---
    sliderPanel = uigridlayout(gl, [1, 3], 'BackgroundColor', bgColor, ...
        'ColumnWidth', {80, '1x', 80});
    sliderPanel.Layout.Row = 4; sliderPanel.Layout.Column = [1, 6];
    
    uilabel(sliderPanel, 'Text', '0°', 'FontColor', txtColor, ...
        'HorizontalAlignment', 'right', 'FontSize', 12);
    
    slider = uislider(sliderPanel, 'Limits', [0 90], 'Value', 0, ...
        'MajorTicks', [0 15 30 45 60 75 90], ...
        'FontColor', txtColor);
    
    uilabel(sliderPanel, 'Text', '90°', 'FontColor', txtColor, ...
        'HorizontalAlignment', 'left', 'FontSize', 12);
    
    % Initial render
    updateDisplay(0);
    
    % Callback
    slider.ValueChangedFcn = @(src, ~) updateDisplay(src.Value);
    
    % --- Nested update function ---
    function updateDisplay(angle_deg)
        theta_rad = angle_deg * pi / 180;
        
        % Digit 3
        I3_true = imrotate(img3, angle_deg, 'bicubic', 'crop');
        I3_approx = img3 + theta_rad * T3;
        I3_diff = abs(I3_true - I3_approx);
        rmse3_val = sqrt(mean((I3_true(:) - I3_approx(:)).^2));
        
        imagesc(ax3_true, I3_true);     colormap(ax3_true, 'gray');   clim(ax3_true, [-1 1]);
        imagesc(ax3_approx, I3_approx); colormap(ax3_approx, 'gray'); clim(ax3_approx, [-1 1]);
        imagesc(ax3_diff, I3_diff);     colormap(ax3_diff, 'hot');    clim(ax3_diff, [0 1]);
        
        % Digit 8
        I8_true = imrotate(img8, angle_deg, 'bicubic', 'crop');
        I8_approx = img8 + theta_rad * T8;
        I8_diff = abs(I8_true - I8_approx);
        rmse8_val = sqrt(mean((I8_true(:) - I8_approx(:)).^2));
        
        imagesc(ax8_true, I8_true);     colormap(ax8_true, 'gray');   clim(ax8_true, [-1 1]);
        imagesc(ax8_approx, I8_approx); colormap(ax8_approx, 'gray'); clim(ax8_approx, [-1 1]);
        imagesc(ax8_diff, I8_diff);     colormap(ax8_diff, 'hot');    clim(ax8_diff, [0 1]);
        
        % Remove tick marks after imagesc resets them
        for a = allAxes
            a.XTick = []; a.YTick = [];
        end
        
        % Update angle marker on error curve
        markerLine.Value = angle_deg;
        
        % Zone-based color
        if angle_deg <= 15
            zoneColor = [0.35 0.80 0.40];  % green = safe
            zoneText = 'SAFE';
        elseif angle_deg <= 30
            zoneColor = [1.00 0.85 0.20];  % yellow = caution
            zoneText = 'CAUTION';
        else
            zoneColor = [1.00 0.30 0.30];  % red = breakdown
            zoneText = 'BREAKDOWN';
        end
        
        angleLbl.Text = sprintf('Angle: %.0f°  |  RMSE(3): %.4f  |  RMSE(8): %.4f  |  %s', ...
            angle_deg, rmse3_val, rmse8_val, zoneText);
        angleLbl.FontColor = zoneColor;
    end
end

function rmse = computeRMSE(img, T, angle_deg)
    % COMPUTERMSE  Quick RMSE for sweep.
    theta_rad = angle_deg * pi / 180;
    I_true = imrotate(img, angle_deg, 'bicubic', 'crop');
    I_approx = img + theta_rad * T;
    rmse = sqrt(mean((I_true(:) - I_approx(:)).^2));
end
