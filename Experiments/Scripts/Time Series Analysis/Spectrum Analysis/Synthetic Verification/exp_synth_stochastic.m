% EXP_SYNTH_STOCHASTIC  Stages 9-12: white noise, MA(1), and AR(1) --
% population spectrum vs averaged periodogram, on processes with a KNOWN
% closed-form spectrum.
%
% PURPOSE:
%   Stages 1-8 were entirely deterministic. This script moves from finite
%   Fourier decomposition to stochastic processes, and demonstrates the
%   central distinction:
%       population spectrum  !=  periodogram from one finite realization
%   A single periodogram realization is a noisy, inconsistent estimator;
%   AVERAGING periodograms across many independent realizations converges
%   toward the theoretical population spectrum s_Y(omega).
%
% DATASET:
%   Simulated processes only:
%     Stage 9  : mu + 3 known sinusoids (as in exp_synth_core.m) + white noise
%     Stage 10 : pure white noise, sigma^2 = 1
%     Stage 11 : MA(1), Y_t = eps_t + theta*eps_{t-1}, theta in {0.71,-0.63}
%     Stage 12 : AR(1), Y_t = phi*Y_{t-1} + eps_t, phi in {0.84,-0.69}
%
% ASSUMPTIONS:
%   1. sigma_eps^2 = 0.83 for Stage 9 (as specified); sigma^2 = 1 for
%      Stages 10-12 (population spectrum formulas are linear in sigma^2,
%      so the unit-variance choice keeps the numbers directly readable).
%   2. AR(1)/MA(1) realizations use a burn-in (discarded initial samples)
%      to reach the stationary distribution before the N=127 window used
%      for estimation.
%   3. n_realizations = 300 independent realizations per averaged
%      periodogram (Monte Carlo estimate of E[periodogram]).
%
% REPRODUCIBILITY:
%   rng(42,'twister') fixes all randomness; re-running reproduces
%   identical figures and numbers.
%
% See also: exp_synth_core.m, exp_synth_leakage.m

clc; clear; close all;
rng(42, 'twister');

rootPath = fileparts(fileparts(fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))))));
addpath(rootPath);

resultsDir = fullfile(fileparts(mfilename('fullpath')), 'Results');
if ~exist(resultsDir, 'dir'); mkdir(resultsDir); end

fprintf('=== STAGE 9-12: STOCHASTIC PROCESSES -- POPULATION VS SAMPLE SPECTRUM ===\n\n');

viz = [];
try
    viz = OceanViz.Core('Theme', 'Dark');
    haveViz = true;
catch
    haveViz = false;
    fprintf('[Info] OceanViz unavailable, using fallback plotting.\n');
end

N = 127;
n = (0:N-1)';
dOmega = 2*pi/N;
wGrid = 2*pi*(0:N-1)'/N;
nRealizations = 300;

%% =====================================================================
%  STAGE 9 — Known sinusoids + white noise
%  =====================================================================
fprintf('--- Stage 9: 3 known sinusoids + white noise (sigma^2=0.83) ---\n');
mu  = 3.7;
A   = [2.1, 1.35, 0.72];
phi = [0.43, -0.91, 1.27];
jTrue = [9, 27, 49];
wTrue = 2*pi*jTrue/N;
sigma2_9 = 0.83;

signalClean = mu ...
    + A(1)*cos(wTrue(1)*n + phi(1)) ...
    + A(2)*cos(wTrue(2)*n + phi(2)) ...
    + A(3)*cos(wTrue(3)*n + phi(3));
signalClean = signalClean - mean(signalClean);

epsOne = sqrt(sigma2_9) * randn(N,1);
yOneRealization = signalClean + (epsOne - mean(epsOne));
P_oneRealization = abs(fft(yOneRealization)).^2 / (2*pi*N);

P_stack9 = zeros(N, nRealizations);
for r = 1:nRealizations
    eps = sqrt(sigma2_9) * randn(N,1);
    yr = signalClean + (eps - mean(eps));
    P_stack9(:, r) = abs(fft(yr)).^2 / (2*pi*N);
end
P_avg9 = mean(P_stack9, 2);

noiseFloorTheory = sigma2_9 / (2*pi);
maskFarFromPeaks = true(N,1);
for j = jTrue
    idxs = mod((j-2:j+2), N) + 1;
    maskFarFromPeaks(idxs) = false;
    idxs2 = mod((N-j-2:N-j+2), N) + 1;
    maskFarFromPeaks(idxs2) = false;
end
fprintf('Averaged-periodogram noise floor (away from peaks) = %.4f, theory sigma^2/(2*pi) = %.4f\n', ...
    mean(P_avg9(maskFarFromPeaks)), noiseFloorTheory);

if haveViz
    fig9 = viz.createFigure('Stage 9 - White Noise Averaging');
else
    fig9 = figure('Name', 'Stage 9 - White Noise Averaging');
end
tiledlayout(2,1);
ax1 = nexttile;
plot(ax1, wGrid(1:floor(N/2)), P_oneRealization(1:floor(N/2)), 'LineWidth', 1.0);
title(ax1, 'Stage 9: single realization (noisy periodogram)');
xlabel(ax1, '\omega (rad/sample)'); ylabel(ax1, 's(\omega)');
if haveViz; viz.styleAxes(ax1); end
ax2 = nexttile;
plot(ax2, wGrid(1:floor(N/2)), P_avg9(1:floor(N/2)), 'LineWidth', 1.0); hold(ax2, 'on');
yline(ax2, noiseFloorTheory, '--', 'DisplayName', 'noise floor \sigma^2/(2\pi)');
title(ax2, sprintf('Stage 9: average of %d realizations (smoother, floor -> theory)', nRealizations));
xlabel(ax2, '\omega (rad/sample)'); ylabel(ax2, 's(\omega)');
legend(ax2, 'Location', 'best');
if haveViz; viz.styleAxes(ax2); end
saveFigure(fig9, fullfile(resultsDir, 'stage9_white_noise_plus_signal.png'), haveViz, viz);

%% =====================================================================
%  STAGE 10 — Pure white noise: population vs sample autocovariance
%  =====================================================================
fprintf('\n--- Stage 10: Pure white noise, sigma^2=1 ---\n');
sigma2_10 = 1.0;

gammaOne = zeros(6,1);
epsOne10 = sqrt(sigma2_10) * randn(N,1);
for k = 0:5
    gammaOne(k+1) = sum(epsOne10(1:N-k) .* epsOne10(1+k:N)) / N;
end
fprintf('Single-realization hat_gamma_k (k=0..5): %s\n', mat2str(round(gammaOne,3)'));
fprintf('  (population gamma_0=1, gamma_k=0 for k~=0 -- one realization will NOT match exactly)\n');

gammaAvg = zeros(6,1);
P_stack10 = zeros(N, nRealizations);
for r = 1:nRealizations
    eps = sqrt(sigma2_10) * randn(N,1);
    for k = 0:5
        gammaAvg(k+1) = gammaAvg(k+1) + sum(eps(1:N-k) .* eps(1+k:N)) / N;
    end
    P_stack10(:, r) = abs(fft(eps)).^2 / (2*pi*N);
end
gammaAvg = gammaAvg / nRealizations;
P_avg10 = mean(P_stack10, 2);
flatTheory10 = sigma2_10 / (2*pi);

fprintf('Averaged hat_gamma_k over %d realizations (k=0..5): %s\n', nRealizations, mat2str(round(gammaAvg,3)'));
fprintf('Averaged periodogram mean level = %.5f, theory s_Y(w)=sigma^2/(2*pi) = %.5f\n', mean(P_avg10), flatTheory10);

if haveViz
    fig10 = viz.createFigure('Stage 10 - White Noise Theory');
else
    fig10 = figure('Name', 'Stage 10 - White Noise Theory');
end
ax = gca;
plot(ax, wGrid, P_avg10, 'LineWidth', 1.0, 'DisplayName', sprintf('avg of %d periodograms', nRealizations)); hold(ax, 'on');
yline(ax, flatTheory10, '--', 'LineWidth', 1.5, 'DisplayName', 's_Y(\omega) = \sigma^2/(2\pi)');
xlabel(ax, '\omega (rad/sample)'); ylabel(ax, 's(\omega)');
title(ax, 'Stage 10: Averaged periodogram -> flat population spectrum of white noise');
xlim(ax, [0, pi]);
legend(ax, 'Location', 'best');
if haveViz; viz.styleAxes(ax); end
saveFigure(fig10, fullfile(resultsDir, 'stage10_white_noise_theory.png'), haveViz, viz);

%% =====================================================================
%  STAGE 11 — MA(1) as a spectral filter
%  =====================================================================
fprintf('\n--- Stage 11: MA(1), sigma^2=1 ---\n');
thetas = [0.71, -0.63];
burnMA = 200;

if haveViz
    fig11 = viz.createFigure('Stage 11 - MA(1) Spectrum');
else
    fig11 = figure('Name', 'Stage 11 - MA(1) Spectrum');
end
tiledlayout(1, numel(thetas));
for i = 1:numel(thetas)
    theta = thetas(i);
    s0 = (1/(2*pi)) * (1+theta)^2;
    sPi = (1/(2*pi)) * (1-theta)^2;
    fprintf('theta=%+.2f: s(0)=%.4f, s(pi)=%.4f  [theory: sigma^2/(2*pi)*(1+/-theta)^2]\n', theta, s0, sPi);

    P_stack = zeros(N, nRealizations);
    for r = 1:nRealizations
        epsFull = randn(N + burnMA + 1, 1);
        yma = epsFull(2:end) + theta * epsFull(1:end-1);
        yma = yma(burnMA+1:end);
        P_stack(:, r) = abs(fft(yma)).^2 / (2*pi*N);
    end
    P_avg = mean(P_stack, 2);
    sTheory = (1/(2*pi)) * (1 + theta^2 + 2*theta*cos(wGrid));

    ax = nexttile;
    plot(ax, wGrid(1:floor(N/2)), P_avg(1:floor(N/2)), 'LineWidth', 1.2, 'DisplayName', 'averaged periodogram'); hold(ax, 'on');
    plot(ax, wGrid(1:floor(N/2)), sTheory(1:floor(N/2)), '--', 'LineWidth', 1.5, 'DisplayName', 'theory s_Y(\omega)');
    title(ax, sprintf('MA(1), \\theta=%+.2f', theta));
    xlabel(ax, '\omega (rad/sample)'); ylabel(ax, 's(\omega)');
    legend(ax, 'Location', 'best');
    if haveViz; viz.styleAxes(ax); end
end
saveFigure(fig11, fullfile(resultsDir, 'stage11_ma1_spectrum.png'), haveViz, viz);

%% =====================================================================
%  STAGE 12 — AR(1)
%  =====================================================================
fprintf('\n--- Stage 12: AR(1), sigma^2=1 ---\n');
phis = [0.84, -0.69];
burnAR = 500;

if haveViz
    fig12 = viz.createFigure('Stage 12 - AR(1) Spectrum');
else
    fig12 = figure('Name', 'Stage 12 - AR(1) Spectrum');
end
tiledlayout(2, numel(phis));
for i = 1:numel(phis)
    ph = phis(i);
    gamma0Theory = 1 / (1 - ph^2);
    fprintf('phi=%+.2f: theoretical gamma_0=%.4f, gamma_k=gamma_0*phi^k\n', ph, gamma0Theory);

    gammaAvgAR = zeros(8,1);
    P_stack = zeros(N, nRealizations);
    for r = 1:nRealizations
        epsFull = randn(N + burnAR, 1);
        yar = zeros(N + burnAR, 1);
        for t = 2:N+burnAR
            yar(t) = ph*yar(t-1) + epsFull(t);
        end
        yar = yar(burnAR+1:end);
        for k = 0:7
            gammaAvgAR(k+1) = gammaAvgAR(k+1) + sum(yar(1:N-k) .* yar(1+k:N)) / N;
        end
        P_stack(:, r) = abs(fft(yar)).^2 / (2*pi*N);
    end
    gammaAvgAR = gammaAvgAR / nRealizations;
    P_avg = mean(P_stack, 2);
    sTheory = 1 ./ (2*pi*(1 + ph^2 - 2*ph*cos(wGrid)));

    fprintf('  averaged hat_gamma_k (k=0..7): %s\n', mat2str(round(gammaAvgAR,3)'));

    axAcf = nexttile;
    stem(axAcf, 0:7, gammaAvgAR, 'filled'); hold(axAcf, 'on');
    plot(axAcf, 0:7, gamma0Theory*ph.^(0:7), '--', 'LineWidth', 1.2, 'DisplayName', 'theory');
    title(axAcf, sprintf('AR(1) \\phi=%+.2f: avg autocovariance', ph));
    xlabel(axAcf, 'lag k'); ylabel(axAcf, 'hat\gamma_k');
    if haveViz; viz.styleAxes(axAcf); end

    axPsd = nexttile;
    plot(axPsd, wGrid(1:floor(N/2)), P_avg(1:floor(N/2)), 'LineWidth', 1.2, 'DisplayName', 'averaged periodogram'); hold(axPsd, 'on');
    plot(axPsd, wGrid(1:floor(N/2)), sTheory(1:floor(N/2)), '--', 'LineWidth', 1.5, 'DisplayName', 'theory s_Y(\omega)');
    title(axPsd, sprintf('AR(1) \\phi=%+.2f: spectrum', ph));
    xlabel(axPsd, '\omega (rad/sample)'); ylabel(axPsd, 's(\omega)');
    legend(axPsd, 'Location', 'best');
    if haveViz; viz.styleAxes(axPsd); end
end
saveFigure(fig12, fullfile(resultsDir, 'stage12_ar1_spectrum.png'), haveViz, viz);

%% =====================================================================
%  Compact diagnostic report
%  =====================================================================
reportPath = fullfile(resultsDir, 'report_stochastic.md');
fid = fopen(reportPath, 'w');
fprintf(fid, '# Stage 9-12 Diagnostic Report (exp_synth_stochastic.m)\n\n');
fprintf(fid, '| Experiment | Main diagnostic | Result |\n');
fprintf(fid, '|---|---|---|\n');
fprintf(fid, '| White noise + signal (Stage 9) | avg noise floor vs theory | %.4f vs %.4f |\n', mean(P_avg9(maskFarFromPeaks)), noiseFloorTheory);
fprintf(fid, '| Pure white noise (Stage 10) | avg periodogram mean vs sigma^2/(2pi) | %.5f vs %.5f |\n', mean(P_avg10), flatTheory10);
fprintf(fid, '| MA(1) theta=%.2f (Stage 11) | s(0) theory | %.4f |\n', thetas(1), (1/(2*pi))*(1+thetas(1))^2);
fprintf(fid, '| MA(1) theta=%.2f (Stage 11) | s(0) theory | %.4f |\n', thetas(2), (1/(2*pi))*(1+thetas(2))^2);
fprintf(fid, '| AR(1) phi=%.2f (Stage 12) | gamma_0 theory | %.4f |\n', phis(1), 1/(1-phis(1)^2));
fprintf(fid, '| AR(1) phi=%.2f (Stage 12) | gamma_0 theory | %.4f |\n', phis(2), 1/(1-phis(2)^2));
fclose(fid);
fprintf('\nReport written to %s\n', reportPath);

fprintf('\n=== STAGE 9-12 COMPLETE ===\n');

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
