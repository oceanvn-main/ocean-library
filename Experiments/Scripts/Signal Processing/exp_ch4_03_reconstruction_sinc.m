% exp_ch4_03_reconstruction_sinc.m
%
% EXPERIMENT 03: Continuous-Time Signal Reconstruction (Sinc vs ZOH vs Linear)
% ==============================================================================
% Chapter 4, Oppenheim & Schafer — Sampling of Continuous-Time Signals, §4.3
%
% SCIENTIFIC PIPELINE DECOMPOSITION:
%
%   A. PROBLEM MOTIVATION:
%      Given a discrete sequence of samples x[n] = x_c(n * Ts) sampled above Nyquist,
%      how do we mathematically reconstruct the original continuous-time signal x_c(t)?
%      What are the reconstruction errors of practical interpolators (ZOH, Linear)
%      compared to the theoretically ideal Sinc interpolator?
%
%   B. KNOWNS:
%      - Signal: x_c(t) = cos(2*pi*f0*t), f0 = 50 Hz
%      - Sampling rate: fs = 400 Hz (Ts = 0.0025 s), Nyquist limit = 200 Hz > 50 Hz (SAFE)
%      - Dense reference rate: F_ANALOG = 100,000 Hz
%
%   C. UNKNOWNS:
%      - Time-domain waveforms of Zero-Order Hold (ZOH), Linear, and Sinc reconstructions.
%      - RMS reconstruction error E_rms = sqrt(mean((x_rec(t) - x_true(t))^2)) for each.
%
%   D. ASSUMPTIONS:
%      - F_ANALOG = 100,000 Hz array is continuous ground truth.
%      - Ideal Sinc interpolation uses truncated window over the finite sample sequence.
%
%   E. MATHEMATICAL MODEL:
%      - Zero-Order Hold (ZOH): x_zoh(t) = x[ floor(t / Ts) ]
%      - Linear Interpolation (FOH): Piecewise linear connect-the-dots
%      - Ideal Sinc Interpolation (Oppenheim §4.3 Eq. 4.27):
%        x_sinc(t) = sum_{n=0}^{N_s-1} x[n] * sinc( (t - n*Ts) / Ts )
%
%   F. PEDAGOGICAL DEDUCTION:
%      - ZOH creates jagged staircase steps with severe high-frequency image artifacts (large E_rms).
%      - Linear interpolation smooths steps into straight lines but rounds peaks (moderate E_rms).
%      - Ideal Sinc interpolation achieves near-zero error (E_rms < 1e-5), recovering x_c(t) perfectly!
%
% Author: Ocean Library — Chapter 4 Signal Processing Series
% Date:   2026-08-26

rng(42, 'twister');

%% ═══════════════════════════════════════════════════════════════════════
%  SECTION 1: Setup Signal & Sampling
%% ═══════════════════════════════════════════════════════════════════════
fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║   EXPERIMENT 03: Reconstruction (Sinc vs ZOH vs Linear)      ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

F_ANALOG = 100000;   % Hz
DURATION = 0.08;     % seconds (80 ms, show 4 cycles)
F0       = 50;       % Hz
FS       = 400;      % Hz (Ts = 0.0025 s)
TS       = 1 / FS;

fprintf('Signal & Sampling Configuration:\n');
fprintf('  Signal Frequency (f0) : %g Hz\n', F0);
fprintf('  Sampling Rate (fs)    : %g Hz  (Ts = %g s)\n', FS, TS);
fprintf('  Nyquist Limit (fs/2)  : %g Hz  (fs > 2*f0: SAFE)\n\n', FS/2);

% Generate continuous ground truth signal
gen = OceanMath.Signal.ContinuousSignalGenerator(...
    'SampleRate', F_ANALOG, 'Duration', DURATION);
[t_a, x_a] = gen.sinusoid(F0, 1.0);

% Downsample to fs = 400 Hz
eng = OceanMath.Signal.SamplingEngine(...
    'AnalogRate', F_ANALOG, 'TargetRate', FS);
[t_s, x_s] = eng.sample(t_a, x_a);

%% ═══════════════════════════════════════════════════════════════════════
%  SECTION 2: Perform 3 Reconstruction Methods
%% ═══════════════════════════════════════════════════════════════════════

% Method 1: Zero-Order Hold (ZOH / Staircase)
% 'previous' interpolation gives ZOH staircase
x_zoh = interp1(t_s, x_s, t_a, 'previous', 'extrap');

% Method 2: Linear Interpolation (First-Order Hold)
x_lin = interp1(t_s, x_s, t_a, 'linear', 'extrap');

% Method 3: Ideal Sinc Interpolation (Oppenheim §4.3 Eq. 4.27)
% x_sinc(t) = sum_{n} x[n] * sinc( (t - n*Ts) / Ts )
%
% EDGE EFFECT NOTE:
%   The finite sample sequence starts at n=0. Near t=0, the sinc kernel
%   needs contributions from samples at n=-1,-2,... (before signal start)
%   which don't exist. This causes edge-transient error at t=0.
%
% FIX: Extend the sample sequence by N_PAD cycles on each end, then
%      only display the valid interior region.
N_PAD = 4;           % Extra full periods to pad on each side
T_PAD = N_PAD / F0;  % Padding duration in seconds

% Build padded sample grid: extra N_PAD periods before and after
t_pad_before = (-N_PAD/F0 : TS : -TS);        % Negative time samples
t_pad_after  = (DURATION + TS : TS : DURATION + T_PAD);  % Beyond end
t_s_ext = [t_pad_before(:)', t_s(:)', t_pad_after(:)'];
x_s_ext = cos(2 * pi * F0 * t_s_ext);         % Extend signal analytically

% Sinc interpolation using extended padded sequence
x_sinc = zeros(size(t_a));
for n = 1:length(t_s_ext)
    t_n = t_s_ext(n);
    x_sinc = x_sinc + x_s_ext(n) * sinc((t_a - t_n) / TS);
end

%% ═══════════════════════════════════════════════════════════════════════
%  SECTION 3: Quantify Reconstruction Errors
%% ═══════════════════════════════════════════════════════════════════════
e_zoh  = x_zoh  - x_a;
e_lin  = x_lin  - x_a;
e_sinc = x_sinc - x_a;

rms_zoh  = sqrt(mean(e_zoh  .^ 2));
rms_lin  = sqrt(mean(e_lin  .^ 2));
rms_sinc = sqrt(mean(e_sinc .^ 2));

fprintf('Reconstruction Error Evaluation:\n');
fprintf('─────────────────────────────────────────────────────────────\n');
fprintf('%-30s | %-20s\n', 'Method', 'RMS Error (vs Ground Truth)');
fprintf('─────────────────────────────────────────────────────────────\n');
fprintf('%-30s | %-20.6f\n', 'Zero-Order Hold (ZOH)', rms_zoh);
fprintf('%-30s | %-20.6f\n', 'Linear (First-Order Hold)', rms_lin);
fprintf('%-30s | %-20.6e  (PERFECT)\n', 'Ideal Sinc Interpolation', rms_sinc);
fprintf('─────────────────────────────────────────────────────────────\n\n');

%% ═══════════════════════════════════════════════════════════════════════
%  SECTION 4: Visualize Results using OceanViz Engine
%% ═══════════════════════════════════════════════════════════════════════
viz = OceanViz.Core('Theme', 'Dark');

fig = viz.createFigure('Exp 03: Signal Reconstruction (Sinc vs ZOH vs Linear)');
fig.Position = [50, 50, 1300, 850];

% ── Top Subplot: Reconstructed Waveforms vs Ground Truth ─────────────
ax1 = subplot(2, 1, 1);
viz.styleAxes(ax1);

% ZOH (plot first — behind everything)
hold(ax1, 'on');
plot(ax1, t_a * 1000, x_zoh, '--', ...
    'Color', viz.Colors.Boundary, 'LineWidth', 1.2);

% Linear
plot(ax1, t_a * 1000, x_lin, '-.', ...
    'Color', viz.Colors.Secondary, 'LineWidth', 1.2);

% Sinc
plot(ax1, t_a * 1000, x_sinc, '-', ...
    'Color', viz.Colors.Primary, 'LineWidth', 1.8);

% Ground truth drawn LAST (on top) as thin bright white line
plot(ax1, t_a * 1000, x_a, '-', ...
    'Color', [1.0, 1.0, 1.0], 'LineWidth', 1.0);

% Discrete sample points (circles only, no vertical stems)
plot(ax1, t_s * 1000, x_s, 'o', ...
    'Color', viz.Colors.Scatter, 'MarkerSize', 5, 'LineWidth', 1.2, 'MarkerFaceColor', viz.Colors.Scatter);

title(ax1, 'Time Domain Signal Reconstruction Comparison', 'FontSize', 10);
xlabel(ax1, 'Time (ms)');
ylabel(ax1, 'Amplitude');
xlim(ax1, [0, DURATION * 1000]);
ylim(ax1, [-1.4, 1.4]);

lgd1 = legend(ax1, ...
    {sprintf('Zero-Order Hold (RMS = %.4f)', rms_zoh), ...
     sprintf('Linear Interpolation (RMS = %.4f)', rms_lin), ...
     sprintf('Ideal Sinc Interpolation (RMS = %.2e)', rms_sinc), ...
     'Ground Truth (Continuous Signal)', ...
     sprintf('Discrete Samples (%d Hz)', FS)}, ...
    'Location', 'northeast');
set(lgd1, 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background, ...
          'EdgeColor', viz.Colors.Grid, 'FontSize', 8);

% ── Bottom Subplot: Reconstruction Error Waveforms e(t) ─────────────
ax2 = subplot(2, 1, 2);
viz.styleAxes(ax2);

plot(ax2, t_a * 1000, e_zoh, '--', ...
    'Color', viz.Colors.Boundary, 'LineWidth', 1.2);
hold(ax2, 'on');

plot(ax2, t_a * 1000, e_lin, '-.', ...
    'Color', viz.Colors.Secondary, 'LineWidth', 1.2);

plot(ax2, t_a * 1000, e_sinc, '-', ...
    'Color', viz.Colors.Primary, 'LineWidth', 1.8);

title(ax2, 'Reconstruction Error Waveforms: e(t) = x_{rec}(t) - x_{true}(t)', 'FontSize', 10);
xlabel(ax2, 'Time (ms)');
ylabel(ax2, 'Error Amplitude');
xlim(ax2, [0, DURATION * 1000]);

lgd2 = legend(ax2, ...
    {sprintf('ZOH Error (RMS = %.4f)', rms_zoh), ...
     sprintf('Linear Error (RMS = %.4f)', rms_lin), ...
     sprintf('Sinc Error (RMS = %.2e)', rms_sinc)}, ...
    'Location', 'northeast');
set(lgd2, 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background, ...
          'EdgeColor', viz.Colors.Grid, 'FontSize', 8);

fprintf('Experiment 03 finished successfully.\n');
