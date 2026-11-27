% exp_ch4_02_aliasing_spectral_folding.m
%
% EXPERIMENT 02: Aliasing Mechanism and Spectral Folding Trajectories
% ====================================================================
% Chapter 4, Oppenheim & Schafer — Sampling of Continuous-Time Signals, §4.2
%
% SCIENTIFIC PIPELINE DECOMPOSITION:
%
%   A. PROBLEM MOTIVATION:
%      When a continuous-time signal contains frequency components exceeding
%      the Nyquist ceiling (fs/2), why do those frequencies not simply vanish,
%      but instead appear as low-frequency "ghost" tones (aliasing)?
%      How can we mathematically predict and visually trace their trajectories?
%
%   B. KNOWNS
%      - Linear Chirp signal: f_inst(t) sweeps from f_start = 0 Hz to f_end = 1200 Hz over T = 2.0 s.
%      - Target discrete sample rate: fs = 400 Hz.
%      - Nyquist ceiling: f_nyq = fs / 2 = 200 Hz.
%
%   C. UNKNOWNS:
%      - The exact observed frequency trajectory in time-frequency space.
%      - Verification of the spectral folding formula f_alias = |mod(f_inst + fs/2, fs) - fs/2|.
%
%   D. ASSUMPTIONS:
%      - F_ANALOG = 100,000 Hz represents continuous ground truth.
%      - Direct downsampling without anti-aliasing prefilter.
%
%   E. MATHEMATICAL MODEL:
%      - Instantaneous frequency: f_inst(t) = f_start + ((f_end - f_start)/T) * t
%      - Spectral folding formula (Oppenheim §4.2 Eq. 4.28):
%        f_alias(t) = | mod(f_inst(t) + fs/2, fs) - fs/2 |
%      - Over 0 -> 1200 Hz with fs = 400 Hz:
%        0 -> 200 Hz:   rises  (0 -> 200 Hz)
%        200 -> 400 Hz: falls  (200 -> 0 Hz, folding back!)
%        400 -> 600 Hz: rises  (0 -> 200 Hz)
%        600 -> 800 Hz: falls  (200 -> 0 Hz)
%        800 -> 1000 Hz: rises (0 -> 200 Hz)
%        1000 -> 1200 Hz: falls (200 -> 0 Hz)
%
%   F. PEDAGOGICAL DEDUCTION:
%      - STFT Spectrogram will display a characteristic V-shaped sawtooth trajectory
%        that bounces off 0 Hz and 200 Hz (Nyquist ceiling) precisely matching formula.
%
% Author: Ocean Library — Chapter 4 Signal Processing Series
% Date:   2026-08-26

rng(42, 'twister');

%% ═══════════════════════════════════════════════════════════════════════
%  SECTION 1: Setup Linear Chirp & Ground Truth
%% ═══════════════════════════════════════════════════════════════════════
fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║   EXPERIMENT 02: Aliasing & Spectral Folding Trajectories    ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

F_ANALOG = 100000;   % Hz
DURATION = 2.0;      % seconds
F_START  = 0;        % Hz
F_END    = 1200;     % Hz (crosses Nyquist 200 Hz multiple times!)

FS_DISCRETE = 400;   % Hz
F_NYQUIST   = FS_DISCRETE / 2;  % 200 Hz

fprintf('Chirp Configuration:\n');
fprintf('  Sweep Range           : %g Hz -> %g Hz over %g s\n', F_START, F_END, DURATION);
fprintf('  Discrete Sample Rate  : %g Hz\n', FS_DISCRETE);
fprintf('  Nyquist Ceiling (fs/2): %g Hz\n\n', F_NYQUIST);

% Generate continuous chirp wave
gen = OceanMath.Signal.ContinuousSignalGenerator(...
    'SampleRate', F_ANALOG, 'Duration', DURATION);
[t_a, x_a] = gen.chirp(F_START, F_END);

%% ═══════════════════════════════════════════════════════════════════════
%  SECTION 2: Direct Downsampling & STFT Analysis
%% ═══════════════════════════════════════════════════════════════════════
% Downsample directly to 400 Hz (no prefilter)
eng = OceanMath.Signal.SamplingEngine(...
    'AnalogRate', F_ANALOG, 'TargetRate', FS_DISCRETE);
[t_s, x_s] = eng.sample(t_a, x_a);

% Compute STFT Spectrogram using SpectralAnalyzer (Padded for edge-to-edge alignment)
WIN_LEN = 64;
HOP     = 4;

% Zero-pad signal ends by WIN_LEN/2 so STFT centers start at t = 0.0 s
x_s_padded = [zeros(1, WIN_LEN/2), x_s, zeros(1, WIN_LEN/2)];
[t_stft, f_stft, S_mag] = OceanMath.Signal.SpectralAnalyzer.stft(...
    x_s_padded, FS_DISCRETE, 'WinLen', WIN_LEN, 'Hop', HOP);

% Align time axis to start at 0.0 s
t_stft = t_stft - (WIN_LEN / 2) / FS_DISCRETE;

% Theoretical instantaneous frequency curve
t_dense  = linspace(0, DURATION, 2000);
f_inst   = F_START + (F_END - F_START) * (t_dense / DURATION);

% Theoretical alias prediction using aliasFormula
f_pred = OceanMath.Signal.SpectralAnalyzer.aliasFormula(f_inst, FS_DISCRETE);

fprintf('STFT Spectrogram Computed:\n');
fprintf('  Time frames           : %d  ([0, %g] s)\n', length(t_stft), DURATION);
fprintf('  Frequency bins        : %d  ([0, %g] Hz)\n', length(f_stft), F_NYQUIST);
fprintf('  Window length         : %d samples (%.3f s)\n\n', WIN_LEN, WIN_LEN/FS_DISCRETE);

%% ═══════════════════════════════════════════════════════════════════════
%  SECTION 3: Visualize Spectrogram & Overlay Theoretical Folding Curve
%% ═══════════════════════════════════════════════════════════════════════
viz = OceanViz.Core('Theme', 'Dark');

fig = viz.createFigure('Exp 02: Aliasing & Spectral Folding Trajectories');
fig.Position = [50, 50, 1200, 800];

% ── Subplot 1: True Continuous Frequency Sweep vs Time ───────────────
ax1 = subplot(2, 1, 1);
viz.styleAxes(ax1);

plot(ax1, t_dense, f_inst, '-', ...
    'Color', viz.Colors.Primary, 'LineWidth', 2.0);
hold(ax1, 'on');

% Mark Nyquist boundaries
yline(ax1, F_NYQUIST, '--', 'Color', viz.Colors.Boundary, 'LineWidth', 1.5);
yline(ax1, 2*F_NYQUIST, ':', 'Color', viz.Colors.Tertiary, 'LineWidth', 1.0);
yline(ax1, 3*F_NYQUIST, ':', 'Color', viz.Colors.Tertiary, 'LineWidth', 1.0);

title(ax1, 'True Continuous Instantaneous Frequency Sweep: 0 Hz -> 1200 Hz', 'FontSize', 10);
xlabel(ax1, 'Time (seconds)');
ylabel(ax1, 'Frequency (Hz)');
xlim(ax1, [0, DURATION]);
ylim(ax1, [0, F_END + 50]);

lgd1 = legend(ax1, {'True Instantaneous Frequency f_{inst}(t)', 'Nyquist Ceiling (200 Hz)'}, ...
    'Location', 'northwest');
set(lgd1, 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background, ...
          'EdgeColor', viz.Colors.Grid, 'FontSize', 8);

% ── Subplot 2: STFT Spectrogram with Theoretical Folding Overlay ─────
ax2 = subplot(2, 1, 2);
viz.styleAxes(ax2);

% Spectrogram image plot (in dB scale for contrast)
S_db = 20 * log10(S_mag + 1e-4);
h_img = imagesc(ax2, t_stft, f_stft, S_db);
axis(ax2, 'xy');
colormap(ax2, 'parula');

hold(ax2, 'on');

% Overlay theoretical folding curve
h_line = plot(ax2, t_dense, f_pred, 'w--', 'LineWidth', 2.0);

title(ax2, 'Sampled STFT Spectrogram (fs = 400 Hz) with Theoretical Folding Overlay', 'FontSize', 10);
xlabel(ax2, 'Time (seconds)');
ylabel(ax2, 'Observed Frequency (Hz)');
xlim(ax2, [0, DURATION]);
ylim(ax2, [0, F_NYQUIST]);

lgd2 = legend(ax2, h_line, {'Theoretical Folding Formula: |mod(f+fs/2, fs) - fs/2|'}, ...
    'Location', 'northwest');
set(lgd2, 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background, ...
          'EdgeColor', viz.Colors.Grid, 'FontSize', 8);

fprintf('Experiment 02 finished successfully.\n');
