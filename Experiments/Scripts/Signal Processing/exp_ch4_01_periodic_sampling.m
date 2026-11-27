% exp_ch4_01_periodic_sampling.m
%
% EXPERIMENT 01: Oppenheim Fig 4.3 (True Physical Fourier Transform Math)
% ===========================================================================
% Chapter 4, Oppenheim & Schafer — Sampling of Continuous-Time Signals, §4.1
%
% SCIENTIFIC PIPELINE DECOMPOSITION:
%
%   A. PROBLEM MOTIVATION:
%      Demonstrate exact un-normalized physical Fourier Transform math (Oppenheim Fig 4.3).
%      Show that:
%        - Panel (a): Original continuous signal spectrum X_c(f) has Peak = 1.0 (unsampled).
%        - Panel (b): Sampling impulse spectrum S(f) has impulses with height = 1/T = fs (500).
%        - Panel (c): Non-overlapping replicas X_s(f) at fs = 500 Hz have peak height = 1/T = 500.
%        - Panel (d): Overlapping replicas X_s(f) at fs = 200 Hz have individual dashed peak = 1/T = 200,
%          and when they overlap, their ADDITION causes the solid sum spectrum to RISE ABOVE 200 to 236.4!
%
%   B. KNOWNS:
%      - Continuous Signal Bandwidth Limit: f_N = 220 Hz (Triangular spectrum from -220 to +220 Hz).
%      - Nyquist Rate: fs_nyquist = 2 * f_N = 440 Hz.
%      - Safe Rate (c): fs_safe = 500 Hz (Ts = 1/500 = 0.0020 s, 1/T = 500).
%      - Aliased Rate (d): fs_alias = 200 Hz (Ts = 1/200 = 0.0050 s, 1/T = 200).
%
%   C. UNKNOWNS:
%      - Exact physical replica placement and heights scaled by (1/T) = fs.
%      - Exact overlapping sum spectrum showing total amplitude addition rising above 1/T = 200.
%
%   D. MATHEMATICAL MODEL (Oppenheim §4.1 Eq. 4.9 - 4.15):
%      - Original spectrum: X_c(f) = max(0, 1 - |f| / f_N)  [Peak = 1.0]
%      - Sampling function spectrum: S(f) = (1/T) * sum_{k} delta(f - k*fs)
%      - Total Sampled Spectrum: X_s(f) = (1/T) * sum_{k=-K}^{+K} X_c(f - k*fs)
%
% Author: Ocean Library — Chapter 4 Signal Processing Series
% Date:   2026-08-26

rng(42, 'twister');

%% ═══════════════════════════════════════════════════════════════════════
%  SECTION 1: Setup Frequency Grid & Analytical Triangle Spectrum
%% ═══════════════════════════════════════════════════════════════════════
fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║   EXPERIMENT 01: Oppenheim Fig 4.3 (True Physical Math)     ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

F_N      = 220;      % Hz (Signal Bandwidth Limit: [-220, +220] Hz)
FS_NYQ   = 2 * F_N;  % 440 Hz

FS_SAFE  = 500;      % Hz (fs > 440 Hz: Clean copies, 1/T = 500)
FS_ALIAS = 200;      % Hz (fs < 440 Hz: Overlapping copies, 1/T = 200)

fprintf('Signal & Sampling Configuration:\n');
fprintf('  Bandwidth Limit (f_N)   : %g Hz  ([-220, +220] Hz triangle)\n', F_N);
fprintf('  Nyquist Rate (2*f_N)    : %g Hz\n', FS_NYQ);
fprintf('  Safe Rate (fs > 2*f_N)  : %g Hz  (Gain 1/T = %g)\n', FS_SAFE, FS_SAFE);
fprintf('  Aliased Rate (fs < 2*f_N): %g Hz (Gain 1/T = %g)\n\n', FS_ALIAS, FS_ALIAS);

% Frequency axis for plots
f = linspace(-900, 900, 2000);

% Analytical triangular spectrum helper
triangle_spec = @(f_in, f_center, f_bw) max(0, 1 - abs(f_in - f_center) / f_bw);

% Panel (a) Continuous Spectrum (Peak height = 1.0)
X_c = arrayfun(@(f_val) triangle_spec(f_val, 0, F_N), f);

%% ═══════════════════════════════════════════════════════════════════════
%  SECTION 2: Create Figure Oppenheim 4.3 Layout (True Un-Normalized Math)
%% ═══════════════════════════════════════════════════════════════════════
viz = OceanViz.Core('Theme', 'Dark');

fig = viz.createFigure('Oppenheim & Schafer Fig 4.3: Sampling (True Physical Fourier Math)');
fig.Position = [50, 50, 1100, 950];

% ── Panel (a): Spectrum of Original Continuous Signal X_c(f) ───────────
ax_a = subplot(4, 1, 1);
viz.styleAxes(ax_a);

plot(ax_a, f, X_c, '-', 'Color', viz.Colors.Primary, 'LineWidth', 2.0);
hold(ax_a, 'on');

xline(ax_a, F_N, '--', 'Color', viz.Colors.Boundary, 'LineWidth', 1.2);
xline(ax_a, -F_N, '--', 'Color', viz.Colors.Boundary, 'LineWidth', 1.2);

text(ax_a, F_N + 15, 0.5, '\Omega_N (f_N = +220 Hz)', 'Color', viz.Colors.Boundary, 'FontSize', 8);
text(ax_a, -F_N - 150, 0.5, '-\Omega_N (-f_N = -220 Hz)', 'Color', viz.Colors.Boundary, 'FontSize', 8);
text(ax_a, 20, 1.0, 'Peak = 1.0 (Unsampled)', 'Color', viz.Colors.Primary, 'FontSize', 8, 'FontWeight', 'bold');

title(ax_a, '(a) Spectrum of Original Continuous Signal X_c(\\Omega)  [Peak = 1.0]', 'FontSize', 9);
xlabel(ax_a, 'Frequency \Omega (Hz)');
ylabel(ax_a, '|X_c(\\Omega)|');
xlim(ax_a, [-900, 900]);
ylim(ax_a, [0, 1.3]);

lgd_a = legend(ax_a, {'Original Spectrum X_c(\\Omega) (Peak = 1.0)', 'Bandwidth Limits (+/-\Omega_N)'}, 'Location', 'northeast');
set(lgd_a, 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background, 'EdgeColor', viz.Colors.Grid, 'FontSize', 8);

% ── Panel (b): Spectrum of Sampling Function S(f) (Impulse Height = 1/T = fs = 500) ────
ax_b = subplot(4, 1, 2);
viz.styleAxes(ax_b);

k_vec = -2:2;
f_impulses = k_vec * FS_SAFE;
stem(ax_b, f_impulses, FS_SAFE * ones(size(f_impulses)), 'filled', ...
    'Color', viz.Colors.Boundary, 'LineWidth', 1.8, 'MarkerSize', 6);
hold(ax_b, 'on');

text(ax_b, 20, FS_SAFE * 0.85, sprintf('Impulse Height = 1/T = %d', FS_SAFE), 'Color', viz.Colors.Boundary, 'FontSize', 8, 'FontWeight', 'bold');

title(ax_b, sprintf('(b) Spectrum of Sampling Function S(\\Omega) [Impulses with Height = 1/T = %d]', FS_SAFE), 'FontSize', 9);
xlabel(ax_b, 'Frequency \Omega (Hz)');
ylabel(ax_b, '|S(\\Omega)|');
xlim(ax_b, [-900, 900]);
ylim(ax_b, [0, FS_SAFE * 1.25]);

lgd_b = legend(ax_b, {sprintf('Sampling Impulses (Height = 1/T = %d)', FS_SAFE)}, 'Location', 'northeast');
set(lgd_b, 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background, 'EdgeColor', viz.Colors.Grid, 'FontSize', 8);

% ── Panel (c): Sampled Spectrum with fs > 2*f_N (Replicas Scaled by 1/T = 500) ──
ax_c = subplot(4, 1, 3);
viz.styleAxes(ax_c);

% Plot explicit copies scaled by 1/T = FS_SAFE = 500
k_copies = -2:2;
colors_rep = [0.20, 0.60, 1.00];

for k = k_copies
    X_k = FS_SAFE * arrayfun(@(f_val) triangle_spec(f_val, k * FS_SAFE, F_N), f);
    plot(ax_c, f, X_k, '-', 'Color', colors_rep, 'LineWidth', 1.8);
    hold(ax_c, 'on');
    xline(ax_c, k * FS_SAFE, ':', 'Color', viz.Colors.Tertiary, 'LineWidth', 1.0);
end

text(ax_c, -10, FS_SAFE * 0.7, sprintf('k=0 (Peak = 1/T = %d)', FS_SAFE), 'Color', viz.Colors.Text, 'FontSize', 8);
text(ax_c, FS_SAFE - 10, FS_SAFE * 0.7, sprintf('\\Omega_s (%d)', FS_SAFE), 'Color', viz.Colors.Text, 'FontSize', 8);

title(ax_c, sprintf('(c) Spectrum of Sampled Signal X_s(\\Omega) with \\Omega_s > 2\\Omega_N [Replica Heights = 1/T = %d]', FS_SAFE), 'FontSize', 9);
xlabel(ax_c, 'Frequency \Omega (Hz)');
ylabel(ax_c, '|X_s(\\Omega)|');
xlim(ax_c, [-900, 900]);
ylim(ax_c, [0, FS_SAFE * 1.25]);

lgd_c = legend(ax_c, {sprintf('Periodic Replicas X_c(\\Omega - k\\Omega_s) / T (Peak = %d)', FS_SAFE)}, 'Location', 'northeast');
set(lgd_c, 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background, 'EdgeColor', viz.Colors.Grid, 'FontSize', 8);

% ── Panel (d): Sampled Spectrum with fs < 2*f_N (Dashed Replicas + Addition Rising to 236.4!) ──
ax_d = subplot(4, 1, 4);
viz.styleAxes(ax_d);

% Compute individual dashed replicas (scaled by 1/T = FS_ALIAS = 200) and total sum
k_alias = -4:4;
X_sum = zeros(size(f));

for k = k_alias
    X_k = FS_ALIAS * arrayfun(@(f_val) triangle_spec(f_val, k * FS_ALIAS, F_N), f);
    X_sum = X_sum + X_k;
    % Plot individual shifted copy as DASHED line (Individual Peak = 200)
    plot(ax_d, f, X_k, '--', 'Color', [0.7, 0.7, 0.7], 'LineWidth', 1.0);
    hold(ax_d, 'on');
end

% Plot total summed spectrum as SOLID line on top (Addition rises to 236.4 at peaks!)
h_sum = plot(ax_d, f, X_sum, '-', 'Color', viz.Colors.Secondary, 'LineWidth', 2.2);

xline(ax_d, FS_ALIAS/2, '--', 'Color', viz.Colors.Boundary, 'LineWidth', 1.2);
xline(ax_d, -FS_ALIAS/2, '--', 'Color', viz.Colors.Boundary, 'LineWidth', 1.2);

text(ax_d, -160, 250, 'OVERLAPPING REPLICAS ADD (Peak Rises Above 200 to 236.4!)', 'Color', viz.Colors.Secondary, ...
    'FontSize', 9, 'FontWeight', 'bold');

title(ax_d, sprintf('(d) Spectrum of Sampled Signal X_s(\\Omega) with \\Omega_s < 2\\Omega_N [Individual Replica = %d, Sum Rises to 236.4!]', FS_ALIAS), 'FontSize', 9);
xlabel(ax_d, 'Frequency \Omega (Hz)');
ylabel(ax_d, '|X_s(\\Omega)|');
xlim(ax_d, [-900, 900]);
ylim(ax_d, [0, 280]);

lgd_d = legend(ax_d, [h_sum], {'Solid Red Line: Total Summed Spectrum (Addition of Overlapping Replicas)'}, 'Location', 'northeast');
set(lgd_d, 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background, 'EdgeColor', viz.Colors.Grid, 'FontSize', 8);

fprintf('Experiment 01 finished successfully.\n');
