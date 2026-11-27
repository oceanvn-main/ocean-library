% EXP_SYNTH_LEAKAGE  Stages 7-8: spectral leakage from finite observation,
% window comparison (boxcar vs Hann), and frequency-resolution vs N.
%
% PURPOSE:
%   Stage 1-6 (exp_synth_core.m) used only ON-GRID frequencies, where the
%   finite Fourier basis reconstructs the signal exactly. This script
%   deliberately breaks that ideal condition:
%     finite observation -> rectangular window -> frequency convolution -> leakage
%   by shifting one component off the Fourier grid, and separately studies
%   how observation length N controls frequency-grid RESOLUTION (a related
%   but distinct concept from window-shape leakage).
%
% DATASET:
%   Same closed-form 3-tone signal as exp_synth_core.m, with omega_2
%   perturbed from an exact grid frequency (j=27) to an off-grid value
%   (j=27.37), amplitude and phase unchanged.
%
% ASSUMPTIONS:
%   1. Ts = 0.4 s/sample fixed throughout; all frequency axes are in
%      omega (rad/sample), never mixed with Hz in the same figure.
%   2. "Boxcar" window == no window (implicit rectangular truncation to N
%      samples) == the periodogram used in exp_synth_core.m.
%   3. Window power normalization U = mean(win.^2) is applied so periodogram
%      height is comparable across window choices.
%
% REPRODUCIBILITY:
%   Fully deterministic (no randomness).
%
% See also: exp_synth_core.m, exp_synth_stochastic.m

clc; clear; close all;

rootPath = fileparts(fileparts(fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))))));
addpath(rootPath);

resultsDir = fullfile(fileparts(mfilename('fullpath')), 'Results');
if ~exist(resultsDir, 'dir'); mkdir(resultsDir); end

fprintf('=== STAGE 7-8: LEAKAGE, WINDOWING, AND RESOLUTION ===\n\n');

viz = [];
try
    viz = OceanViz.Core('Theme', 'Dark');
    haveViz = true;
catch
    haveViz = false;
    fprintf('[Info] OceanViz unavailable, using fallback plotting.\n');
end

%% --- Shared signal definition ---
Ts = 0.4;
mu  = 3.7;
A   = [2.1, 1.35, 0.72];
phi = [0.43, -0.91, 1.27];

j1 = 9; j3 = 49;
j2_on  = 27;
j2_off = 27.37;                 % Stage 7: deliberately off the Fourier grid

makeSignal = @(N, j2) mu ...
    + A(1)*cos(2*pi*j1/N*(0:N-1)' + phi(1)) ...
    + A(2)*cos(2*pi*j2/N*(0:N-1)' + phi(2)) ...
    + A(3)*cos(2*pi*j3/N*(0:N-1)' + phi(3));

%% =====================================================================
%  STAGE 7 — On-grid vs off-grid: spectral leakage
%  =====================================================================
fprintf('--- Stage 7: On-grid vs off-grid leakage (N=127) ---\n');
N = 127;
n = (0:N-1)';
w2on  = 2*pi*j2_on/N;
w2off = 2*pi*j2_off/N;

xOn  = makeSignal(N, j2_on);  xOn  = xOn  - mean(xOn);
xOff = makeSignal(N, j2_off); xOff = xOff - mean(xOff);

% Exact N-point DFT bin energies near bin 27 (on-grid vs off-grid)
XonFFT  = fft(xOn);
XoffFFT = fft(xOff);
binsNear = 25:29;
fprintf('|X|^2 near bin 27 (on-grid) : %s\n', mat2str(round(abs(XonFFT(binsNear+1)).^2, 1)));
fprintf('|X|^2 near bin 27 (off-grid): %s\n', mat2str(round(abs(XoffFFT(binsNear+1)).^2, 1)));

% Dense (fine-grid, boxcar) periodogram around omega_2 to visualize the lobe shape
wDense = linspace(w2on - 15*(2*pi/N), w2on + 15*(2*pi/N), 3000)';
Pon  = densePeriodogram(xOn,  n, wDense, ones(N,1));
Poff = densePeriodogram(xOff, n, wDense, ones(N,1));

if haveViz
    fig7a = viz.createFigure('Stage 7 - On-grid vs Off-grid Leakage');
else
    fig7a = figure('Name', 'Stage 7 - On-grid vs Off-grid Leakage');
end
ax = gca;
plot(ax, wDense, 10*log10(Pon), 'LineWidth', 1.3, 'DisplayName', 'on-grid (j=27)'); hold(ax, 'on');
plot(ax, wDense, 10*log10(Poff), 'LineWidth', 1.3, 'DisplayName', 'off-grid (j=27.37)');
xline(ax, w2on, ':', 'DisplayName', '\omega grid bin 27');
xlabel(ax, '\omega (rad/sample)'); ylabel(ax, 's(\omega) [dB]');
title(ax, 'Stage 7: Boxcar periodogram near j=27 -- on-grid concentrates, off-grid leaks');
legend(ax, 'Location', 'best');
if haveViz; viz.styleAxes(ax); end
saveFigure(fig7a, fullfile(resultsDir, 'stage7_leakage_ongrid_vs_offgrid.png'), haveViz, viz);

%% --- Stage 7b: window comparison (boxcar vs Hann), off-grid signal only ---
fprintf('\n--- Stage 7b: Boxcar vs Hann window (off-grid signal) ---\n');
winBox  = ones(N,1);
winHann = 0.5 - 0.5*cos(2*pi*(0:N-1)'/(N-1));

Pbox  = densePeriodogram(xOff, n, wDense, winBox);
Phann = densePeriodogram(xOff, n, wDense, winHann);

wBox  = halfMaxWidth(Pbox,  wDense);
wHann = halfMaxWidth(Phann, wDense);
fprintf('Boxcar main-lobe half-max width : %.5f rad/sample\n', wBox);
fprintf('Hann   main-lobe half-max width : %.5f rad/sample\n', wHann);
fprintf('(Hann should be WIDER but should suppress distant sidelobes more)\n');

if haveViz
    fig7b = viz.createFigure('Stage 7b - Boxcar vs Hann');
else
    fig7b = figure('Name', 'Stage 7b - Boxcar vs Hann');
end
ax = gca;
plot(ax, wDense, 10*log10(Pbox),  'LineWidth', 1.3, 'DisplayName', sprintf('boxcar (width=%.4f)', wBox)); hold(ax, 'on');
plot(ax, wDense, 10*log10(Phann), 'LineWidth', 1.3, 'DisplayName', sprintf('Hann (width=%.4f)', wHann));
xlabel(ax, '\omega (rad/sample)'); ylabel(ax, 's(\omega) [dB]');
title(ax, 'Stage 7b: Window effect on off-grid leakage (same signal, different window)');
legend(ax, 'Location', 'best');
if haveViz; viz.styleAxes(ax); end
saveFigure(fig7b, fullfile(resultsDir, 'stage7b_boxcar_vs_hann.png'), haveViz, viz);

%% =====================================================================
%  STAGE 8 — Frequency resolution vs sample length N
%  =====================================================================
fprintf('\n--- Stage 8: Resolution vs N (Ts fixed = %.2f) ---\n', Ts);
Ns = [63, 127, 251];
colors = lines(numel(Ns));

if haveViz
    fig8 = viz.createFigure('Stage 8 - Resolution vs N');
else
    fig8 = figure('Name', 'Stage 8 - Resolution vs N');
end
ax = gca; hold(ax, 'on');
widths = zeros(size(Ns));
for i = 1:numel(Ns)
    Ni = Ns(i);
    ni = (0:Ni-1)';
    xi = makeSignal(Ni, j2_off); xi = xi - mean(xi);
    wi = linspace(w2on - 15*(2*pi/Ni), w2on + 15*(2*pi/Ni), 3000)';
    Pi = densePeriodogram(xi, ni, wi, ones(Ni,1));
    widths(i) = halfMaxWidth(Pi, wi);
    dOmega_i = 2*pi/Ni;
    fprintf('N=%4d: dOmega=2*pi/N=%.5f rad/sample, half-max lobe width=%.5f rad/sample\n', Ni, dOmega_i, widths(i));
    plot(ax, wi, 10*log10(Pi), 'LineWidth', 1.3, 'Color', colors(i,:), ...
        'DisplayName', sprintf('N=%d (dOmega=%.4f)', Ni, dOmega_i));
end
xlabel(ax, '\omega (rad/sample)'); ylabel(ax, 's(\omega) [dB]');
title(ax, 'Stage 8: Longer N => narrower Fourier-grid spacing => better resolution');
legend(ax, 'Location', 'best');
if haveViz; viz.styleAxes(ax); end
saveFigure(fig8, fullfile(resultsDir, 'stage8_resolution_vs_N.png'), haveViz, viz);

%% =====================================================================
%  Compact diagnostic report
%  =====================================================================
reportPath = fullfile(resultsDir, 'report_leakage.md');
fid = fopen(reportPath, 'w');
fprintf(fid, '# Stage 7-8 Diagnostic Report (exp_synth_leakage.m)\n\n');
fprintf(fid, '| Experiment | Main diagnostic | Result |\n');
fprintf(fid, '|---|---|---|\n');
fprintf(fid, '| Leakage (on- vs off-grid) | \\|X\\|^2 near bin 27, on-grid | %s |\n', mat2str(round(abs(XonFFT(binsNear+1)).^2, 1)));
fprintf(fid, '| Leakage (on- vs off-grid) | \\|X\\|^2 near bin 27, off-grid | %s |\n', mat2str(round(abs(XoffFFT(binsNear+1)).^2, 1)));
fprintf(fid, '| Window effect | half-max width, boxcar vs Hann | %.5f vs %.5f rad/sample |\n', wBox, wHann);
fprintf(fid, '| Resolution vs N | half-max width, N=63/127/251 | %.5f / %.5f / %.5f rad/sample |\n', widths(1), widths(2), widths(3));
fclose(fid);
fprintf('\nReport written to %s\n', reportPath);

fprintf('\n=== STAGE 7-8 COMPLETE ===\n');

function P = densePeriodogram(x, n, wVec, win)
    U = mean(win.^2);
    xw = x .* win;
    P = zeros(size(wVec));
    for k = 1:numel(wVec)
        Xw = sum(xw .* exp(-1i*wVec(k)*n));
        P(k) = abs(Xw)^2 / (2*pi*numel(n)*U);
    end
end

function w = halfMaxWidth(P, wVec)
    half = max(P) / 2;
    idx = find(P >= half);
    w = wVec(idx(end)) - wVec(idx(1));
end

function saveFigure(fig, path, haveViz, viz)
    try
        if haveViz
            viz.export(fig, path);
        else
            exportgraphics(fig, path, 'Resolution', 150);
        end
    catch
        saveas(fig, path);
    end
end
