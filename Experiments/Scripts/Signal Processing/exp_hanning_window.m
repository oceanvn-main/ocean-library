%% exp_hanning_window.m
%
% Experiment: Analysis of Rectangular vs. Raised Cosine (Hann) Window
% Solving Oppenheim & Schafer Discrete-Time Signal Processing (Problem 2.17)
%
% Author : Ocean Library — Signal Processing Experiments
% Date   : 2026-06-03
% -----------------------------------------------------------------------

clc; clear; close all;

%% -----------------------------------------------------------------------
%  COLOR SCHEME & THEME SETUP
% -----------------------------------------------------------------------
BG      = [0.08 0.08 0.10];   % Deep slate background
AX_BG   = [0.05 0.05 0.07];   % Card background
GRID_C  = [0.20 0.20 0.25];   % Subtle grid
TEXT_C  = [0.90 0.90 0.95];   % Bright text
C_RECT  = [0.20 0.60 1.00];   % Electric blue for rectangular window
C_HANN  = [1.00 0.50 0.20];   % Radiant orange/gold for Hanning window

% Ensure the Outputs directory exists
if ~exist('Outputs', 'dir')
    mkdir('Outputs');
end

%% -----------------------------------------------------------------------
%  TIME-DOMAIN DEFINITION (M = 4)
% -----------------------------------------------------------------------
M = 4;
n = -2:6;  % Range of samples to show window boundaries

% Rectangular window r[n]
r = double(n >= 0 & n <= M);

% Hanning window w[n]
w = zeros(size(n));
idx_active = (n >= 0 & n <= M);
w(idx_active) = 0.5 * (1 - cos(2 * pi * n(idx_active) / M));

%% -----------------------------------------------------------------------
%  FREQUENCY-DOMAIN COMPUTATION (DTFT)
% -----------------------------------------------------------------------
w_grid = linspace(-pi, pi, 2000);

% Built-in DTFT for r[n] using freqz (coefficients = r[n] for 0 <= n <= M, i.e., ones(1, M+1))
R_dtft = freqz(ones(1, M+1), 1, w_grid);

% Built-in DTFT for w[n] using freqz (coefficients = w[n] for 0 <= n <= M)
w_active = w(idx_active);
W_dtft_analytical = freqz(w_active, 1, w_grid);

% We verify the frequency shifting formula from part (b) using freqz:
% W(e^{jw}) = 0.5 * R(e^{jw}) - 0.25 * R(e^{j(w - 2pi/M)}) - 0.25 * R(e^{j(w + 2pi/M)})
eval_R = @(w_val) freqz(ones(1, M+1), 1, w_val);

W_dtft_shift = 0.5 * eval_R(w_grid) ...
              - 0.25 * eval_R(w_grid - 2*pi/M) ...
              - 0.25 * eval_R(w_grid + 2*pi/M);

% Verify difference between direct computation and frequency-shift formula
max_diff = max(abs(W_dtft_analytical - W_dtft_shift));
fprintf('========================================================\n');
% Format verification results in terminal
fprintf('Verification of DTFT shift relation for W(e^{j\\omega}):\n');
fprintf('  Max difference: %e\n', max_diff);
if max_diff < 1e-12
    fprintf('  Status: VERIFIED (Discrepancy is below 1e-12)\n');
else
    fprintf('  Status: WARNING (Discrepancy exceeds 1e-12)\n');
end
fprintf('========================================================\n\n');

%% -----------------------------------------------------------------------
%  FIGURE 1: TIME DOMAIN PLOTS (r[n] and w[n] for M=4)
% -----------------------------------------------------------------------
fig1 = figure('Color', BG, 'Position', [100 100 900 700]);

% Subplot 1: Rectangular Window r[n]
ax1 = subplot(2, 1, 1);
styleAxes(ax1, AX_BG, GRID_C, TEXT_C);
hold(ax1, 'on');
stem(ax1, n, r, 'Color', C_RECT, 'LineWidth', 2.0, 'MarkerFaceColor', C_RECT, 'MarkerSize', 8);
title(ax1, 'Rectangular Window r[n] (M = 4)', 'Color', TEXT_C, 'FontSize', 12, 'FontWeight', 'bold');
xlabel(ax1, 'n (samples)', 'Color', TEXT_C);
ylabel(ax1, 'r[n]', 'Color', TEXT_C);
xlim(ax1, [min(n)-0.5, max(n)+0.5]);
ylim(ax1, [-0.2, 1.3]);
hold(ax1, 'off');

% Subplot 2: Hanning Window w[n]
ax2 = subplot(2, 1, 2);
styleAxes(ax2, AX_BG, GRID_C, TEXT_C);
hold(ax2, 'on');
stem(ax2, n, w, 'Color', C_HANN, 'LineWidth', 2.0, 'MarkerFaceColor', C_HANN, 'MarkerSize', 8);
title(ax2, 'Hanning Window w[n] = 0.5 * [1 - cos(2\pi n / M)] (M = 4)', 'Color', TEXT_C, 'FontSize', 12, 'FontWeight', 'bold');
xlabel(ax2, 'n (samples)', 'Color', TEXT_C);
ylabel(ax2, 'w[n]', 'Color', TEXT_C);
xlim(ax2, [min(n)-0.5, max(n)+0.5]);
ylim(ax2, [-0.2, 1.3]);
hold(ax2, 'off');

sgtitle(fig1, 'Time Domain Windows (M = 4)', 'Color', TEXT_C, 'FontSize', 14, 'FontWeight', 'bold');

% Save time domain plot
saveas(fig1, 'Outputs/hanning_window_time.png');

%% -----------------------------------------------------------------------
%  FIGURE 2: FREQUENCY DOMAIN MAGNITUDE PLOTS (|R(e^{jw})| and |W(e^{jw})|)
% -----------------------------------------------------------------------
fig2 = figure('Color', BG, 'Position', [150 150 1000 700]);

ax3 = subplot(1, 1, 1);
styleAxes(ax3, AX_BG, GRID_C, TEXT_C);
hold(ax3, 'on');

% Plot |R(e^{jw})|
plot(ax3, w_grid, abs(R_dtft), 'Color', C_RECT, 'LineWidth', 2.5);

% Plot |W(e^{jw})|
plot(ax3, w_grid, abs(W_dtft_analytical), 'Color', C_HANN, 'LineWidth', 2.5);

% Set axes limits and ticks
xlim(ax3, [-pi, pi]);
ylim(ax3, [-0.2, 5.5]);
set(ax3, 'XTick', -pi:pi/2:pi);
set(ax3, 'XTickLabel', {'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});

% Legend and Labels
lgd = legend(ax3, {'|R(e^{j\omega})| (Rectangular)', '|W(e^{j\omega})| (Hanning)'}, ...
             'Location', 'northeast', 'Color', AX_BG, 'TextColor', TEXT_C, 'EdgeColor', GRID_C);
title(ax3, 'DTFT Magnitude Spectra |R(e^{j\omega})| and |W(e^{j\omega})| (M = 4)', ...
      'Color', TEXT_C, 'FontSize', 14, 'FontWeight', 'bold');
xlabel(ax3, 'Frequency \omega (rad/sample)', 'Color', TEXT_C);
ylabel(ax3, 'Magnitude', 'Color', TEXT_C);

grid(ax3, 'on');
hold(ax3, 'off');

% Save frequency domain plot
saveas(fig2, 'Outputs/hanning_window_freq.png');

fprintf('Successfully saved plots:\n');
fprintf('  - Outputs/hanning_window_time.png\n');
fprintf('  - Outputs/hanning_window_freq.png\n\n');

%% -----------------------------------------------------------------------
%  HELPER FUNCTIONS
% -----------------------------------------------------------------------
function styleAxes(ax, bg, grid_c, text_c)
    set(ax, 'Color', bg, ...
            'XColor', text_c * 0.7, 'YColor', text_c * 0.7, ...
            'GridColor', grid_c, 'GridAlpha', 0.5, ...
            'LineWidth', 1.2, 'FontSize', 10);
    grid(ax, 'on');
    box(ax, 'on');
end
