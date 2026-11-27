% EXP_DOMAIN_SCALING_DIRECTION Verifying the Domain Scaling Law (3.76) Directionally
%
%   DATASET:        Synthetic 2D separable signal: f(x,y) = cos(2*pi*fx*x) + cos(2*pi*fy*y)
%   TRANSFORM:      Non-uniform affine warp A = diag(sx, sy) with sx = 2.0, sy = 0.4
%   ASSUMPTIONS:    Output grid has spacing = 1.0 (Nyquist = 0.5 cycles/pixel).
%                   Warp is applied via inverse mapping (Gather, Algorithm 3.2).
%                   No anti-aliasing prefilter is used.
%                   Bilinear interpolation is used.
%   REPRODUCIBILITY: rng(42, 'twister') at top of script
%
%   Author: Antigravity Code Assistant
%   Date: 2026-06-25

clear;
close all;
clc;

% Set seed for reproducibility
rng(42, 'twister');

disp('Initializing Experiment G: Directional Domain Scaling...');
script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..');
addpath(root_dir);

% Create Results directory if not exists
results_dir = fullfile(script_dir, 'Results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
output_snapshot_path = fullfile(results_dir, 'domain_scaling_direction.png');

%% 1. Setup Signal Parameters and Warping Configuration
N_src = 512; % Source grid size (square)
N_tgt = 512; % Target grid size (square)

fx = 0.15; % Input frequency along X (cycles/pixel)
fy = 0.25; % Input frequency along Y (cycles/pixel)

sx = 2.0;  % Magnify X-axis (stretch)
sy = 0.4;  % Minify Y-axis (squash)

% Expected output continuous frequencies
fx_theory = fx / sx; % 0.15 / 2.0 = 0.075 (cycles/pixel)
fy_theory = fy / sy; % 0.25 / 0.4 = 0.625 (cycles/pixel)

% Nyquist limit of output grid
nyquist = 0.5;

% Expected aliased frequency for Y (since fy_theory > nyquist)
% fy_theory folds back around 0.5 cycles/pixel.
% Specifically, since 0.5 < fy_theory < 1.5, the aliased peak appears at 1.0 - fy_theory.
fy_theory_alias = abs(fy_theory - round(fy_theory)); % |0.625 - 1.0| = 0.375

%% 2. Generate Source Signal
[X_src, Y_src] = meshgrid(1:N_src, 1:N_src);
% Separable continuous-form signal evaluated on integer grid
I_src = cos(2 * pi * fx * X_src) + cos(2 * pi * fy * Y_src);

%% 3. Run Inverse Mapping (Gather, Algorithm 3.2)
disp('Running Inverse Warp (Gather-Resampling) with Bilinear Interpolation...');
xc_src = (N_src + 1) / 2;
yc_src = (N_src + 1) / 2;
xc_tgt = (N_tgt + 1) / 2;
yc_tgt = (N_tgt + 1) / 2;

I_dst = zeros(N_tgt, N_tgt);

for yd = 1:N_tgt
    for xd = 1:N_tgt
        % Center the destination coordinate, scale, and shift to source
        xs = (xd - xc_tgt) / sx + xc_src;
        ys = (yd - yc_tgt) / sy + yc_src;
        
        % Bilinear Interpolation
        if xs >= 1 && xs <= N_src && ys >= 1 && ys <= N_src
            x0 = floor(xs); x1 = min(N_src, x0 + 1); dx = xs - x0;
            y0 = floor(ys); y1 = min(N_src, y0 + 1); dy = ys - y0;
            
            % Interpolate
            w00 = (1-dx)*(1-dy);
            w01 = dx*(1-dy);
            w10 = (1-dx)*dy;
            w11 = dx*dy;
            
            I_dst(yd, xd) = w00 * I_src(y0, x0) + w01 * I_src(y0, x1) + ...
                            w10 * I_src(y1, x0) + w11 * I_src(y1, x1);
        else
            I_dst(yd, xd) = 0; % Out of bounds padding
        end
    end
end

%% 4. Spectral Analysis along X and Y Axes
disp('Performing 1D Spectral Analysis along X and Y...');

% Find the active range along Y dynamically to avoid boundary zero-padding artifacts
active_y = find(sum(abs(I_dst), 2) > 1e-4);
y_start = min(active_y) + 5; % Margin to avoid transition boundaries
y_end = max(active_y) - 5;

% Extract 1D signals
x_sig = I_dst(round(N_tgt/2), :);
y_sig = I_dst(y_start:y_end, round(N_tgt/2))';

% Zero-padded FFT for high spectral resolution
N_fft = 8192;
freq_axis = (-N_fft/2 : N_fft/2 - 1) / N_fft;

% Process X signal
x_sig_detrend = x_sig - mean(x_sig);
win_x = hann(length(x_sig_detrend))';
x_sig_windowed = x_sig_detrend .* win_x;
X_fft = fftshift(fft(x_sig_windowed, N_fft));
X_mag = abs(X_fft) / sum(win_x);

% Process Y signal
y_sig_detrend = y_sig - mean(y_sig);
win_y = hann(length(y_sig_detrend))';
y_sig_windowed = y_sig_detrend .* win_y;
Y_fft = fftshift(fft(y_sig_windowed, N_fft));
Y_mag = abs(Y_fft) / sum(win_y);

% Peak detection in the positive frequency spectrum (0 to 0.5)
pos_indices = find(freq_axis >= 0);
freq_pos = freq_axis(pos_indices);

X_mag_pos = X_mag(pos_indices);
[~, max_idx_x] = max(X_mag_pos);
fx_empirical = freq_pos(max_idx_x);

Y_mag_pos = Y_mag(pos_indices);
[~, max_idx_y] = max(Y_mag_pos);
fy_empirical = freq_pos(max_idx_y);

err_x = abs(fx_empirical - fx_theory);
err_y = abs(fy_empirical - fy_theory_alias);

%% 5. Visualization Dashboard (OceanViz Theme)
disp('Generating Dashboard Visualization...');
viz = OceanViz.Core('Theme', 'Dark');

fig = figure('Name', 'Exp G: Directional Domain Scaling Law', ...
             'Color', viz.Colors.Background, ...
             'Position', [50, 50, 1400, 900]);

% Panel 1: Original Spatial Signal
ax1 = subplot(2, 2, 1);
imagesc(ax1, 1:N_src, 1:N_src, I_src);
colormap(ax1, 'parula');
title(ax1, sprintf('Original Signal (fx = %.2f, fy = %.2f)', fx, fy), 'Color', viz.Colors.Text);
xlabel(ax1, 'X (pixels)');
ylabel(ax1, 'Y (pixels)');
axis(ax1, 'image');
viz.styleAxes(ax1);

% Panel 2: Warped Spatial Signal (Gathered)
ax2 = subplot(2, 2, 2);
imagesc(ax2, 1:N_tgt, 1:N_tgt, I_dst);
colormap(ax2, 'parula');
title(ax2, sprintf('Warped Signal (sx = %.1f, sy = %.1f)', sx, sy), 'Color', viz.Colors.Text);
xlabel(ax2, 'X (pixels)');
ylabel(ax2, 'Y (pixels)');
axis(ax2, 'image');
viz.styleAxes(ax2);

% Draw lines indicating where the 1D cross-sections were extracted
hold(ax2, 'on');
y_mid = round(N_tgt/2);
x_mid = round(N_tgt/2);
plot(ax2, [1, N_tgt], [y_mid, y_mid], '--', 'Color', viz.Colors.Primary, 'LineWidth', 1.5, 'DisplayName', 'X Slice');
plot(ax2, [x_mid, x_mid], [y_start, y_end], '--', 'Color', viz.Colors.Secondary, 'LineWidth', 1.5, 'DisplayName', 'Y Slice');
hold(ax2, 'off');

% Panel 3: FFT along X-axis (Magnified / Lower Frequency / No Aliasing)
ax3 = subplot(2, 2, 3);
plot(ax3, freq_axis, X_mag, 'Color', viz.Colors.Primary, 'LineWidth', 2.0, 'DisplayName', 'Empirical FFT');
hold(ax3, 'on');
xline(ax3, fx_theory, 'w--', 'LineWidth', 1.5, 'DisplayName', sprintf('Theory fx/sx = %.3f', fx_theory));
xline(ax3, nyquist, 'r:', 'LineWidth', 2.0, 'DisplayName', 'Nyquist Limit (0.5)');
hold(ax3, 'off');
title(ax3, 'X-Axis Frequency Spectrum (Magnified / Stretched)', 'Color', viz.Colors.Text);
xlabel(ax3, 'Frequency (cycles/pixel)');
ylabel(ax3, 'Magnitude');
xlim(ax3, [0, 0.8]);
grid(ax3, 'on');
legend(ax3, 'Location', 'northeast', 'TextColor', viz.Colors.Text);
text(ax3, fx_theory - 0.09, max(X_mag)*0.7, sprintf('Peak: %.3f', fx_empirical), 'Color', viz.Colors.Text, 'FontWeight', 'bold');
viz.styleAxes(ax3);

% Panel 4: FFT along Y-axis (Minified / Higher Frequency / Aliasing)
ax4 = subplot(2, 2, 4);

% Periodically extend spectrum to show periodic behavior above Nyquist
freq_ext = [freq_axis, freq_axis + 1.0];
Y_mag_ext = [Y_mag, Y_mag];

plot(ax4, freq_ext, Y_mag_ext, 'Color', viz.Colors.Secondary, 'LineWidth', 2.0, 'DisplayName', 'Empirical FFT');
hold(ax4, 'on');

xlim(ax4, [0, 0.8]);
y_lims = ylim(ax4);

% Shade the Aliasing Zone (above Nyquist 0.5)
p = patch(ax4, [0.5, 0.8, 0.8, 0.5], [y_lims(1), y_lims(1), y_lims(2), y_lims(2)], ...
      [1, 0.3, 0.3], 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', 'Aliased Zone');
uistack(p, 'bottom');

% Draw lines
xline(ax4, nyquist, 'r:', 'LineWidth', 2.0, 'DisplayName', 'Nyquist Limit (0.5)');
xline(ax4, fy_theory, 'g--', 'LineWidth', 1.5, 'DisplayName', sprintf('Theory fy/sy = %.3f', fy_theory));
xline(ax4, fy_theory_alias, 'w--', 'LineWidth', 1.5, 'DisplayName', sprintf('Aliased Peak = %.3f', fy_theory_alias));

% Draw folding-back arrow pointing from theory frequency to aliased frequency
arrow_y = max(Y_mag) * 0.4;
plot(ax4, [fy_theory_alias, fy_theory], [arrow_y, arrow_y], 'w--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
plot(ax4, fy_theory_alias, arrow_y, 'w<', 'MarkerFaceColor', 'w', 'MarkerSize', 8, 'HandleVisibility', 'off');
text(ax4, (fy_theory_alias + fy_theory)/2, arrow_y + max(Y_mag)*0.08, ...
     'Aliasing Fold-back', 'Color', viz.Colors.Text, 'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');

hold(ax4, 'off');
title(ax4, 'Y-Axis Frequency Spectrum (Minified / Squashed)', 'Color', viz.Colors.Text);
xlabel(ax4, 'Frequency (cycles/pixel)');
ylabel(ax4, 'Magnitude');
grid(ax4, 'on');
legend(ax4, 'Location', 'northeast', 'TextColor', viz.Colors.Text);
viz.styleAxes(ax4);

% Save Dashboard
drawnow;
set(fig, 'PaperUnits', 'inches');
set(fig, 'PaperPosition', [0, 0, 14.0, 9.0]);
set(fig, 'PaperSize', [14.0, 9.0]);
print(fig, output_snapshot_path, '-dpng', '-r100');

%% 6. Print Summary Report
fprintf('\n=== EXPERIMENT G SUMMARY: DOMAIN SCALING DIRECTIONALLY ===\n');
fprintf('  Source Image Grid : %d x %d\n', N_src, N_src);
fprintf('  Target Image Grid : %d x %d\n', N_tgt, N_tgt);
fprintf('  Nyquist Limit     : %.3f cycles/pixel\n', nyquist);
fprintf('\n  X-Axis (Magnification / Stretch):\n');
fprintf('    Scale Factor (sx)      : %.1f\n', sx);
fprintf('    Input Frequency (fx)   : %.3f cycles/pixel\n', fx);
fprintf('    Theoretical Freq (fx/sx): %.3f cycles/pixel (No Aliasing expected)\n', fx_theory);
fprintf('    Empirical Peak Freq    : %.3f cycles/pixel\n', fx_empirical);
fprintf('    Discrepancy Error      : %.6f\n', err_x);
fprintf('\n  Y-Axis (Minification / Squash):\n');
fprintf('    Scale Factor (sy)      : %.1f\n', sy);
fprintf('    Input Frequency (fy)   : %.3f cycles/pixel\n', fy);
fprintf('    Theoretical Freq (fy/sy): %.3f cycles/pixel (Exceeds Nyquist, Aliasing expected)\n', fy_theory);
fprintf('    Expected Aliased Freq  : %.3f cycles/pixel\n', fy_theory_alias);
fprintf('    Empirical Peak Freq    : %.3f cycles/pixel\n', fy_empirical);
fprintf('    Discrepancy Error      : %.6f\n', err_y);
fprintf('=========================================================\n');

if err_x < 0.005 && err_y < 0.005
    fprintf('SUCCESS: Directional Domain Scaling Law verified to within numerical precision.\n');
else
    fprintf('FAILURE: Peak discrepancy too large.\n');
end
