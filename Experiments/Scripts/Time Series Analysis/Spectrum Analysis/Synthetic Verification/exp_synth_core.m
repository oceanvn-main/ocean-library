% EXP_SYNTH_CORE  Stages 1-6: Fourier basis, variance decomposition, and
% the two periodogram routes, verified on a KNOWN synthetic process.
%
% PURPOSE:
%   Before trusting any spectral estimator on real data (e.g. sunspots, see
%   exp_population_spectrum.m), verify the entire textbook chain on a signal
%   whose true frequencies, amplitudes, and variance decomposition are known
%   in advance:
%       time samples -> Fourier basis -> frequency coefficients
%                     -> variance decomposition -> periodogram
%   and, independently:
%       time samples -> sample autocovariances -> Fourier transform -> periodogram
%   Both routes must agree exactly (up to floating-point error) when the
%   Fourier basis is complete.
%
% DATASET:
%   None — a deterministic signal is constructed in closed form:
%     y[n] = mu + A1*cos(w1*n+phi1) + A2*cos(w2*n+phi2) + A3*cos(w3*n+phi3)
%   with n = 0..N-1, N = 127 (odd, so eq. 6.2.6 applies without a Nyquist
%   special case), and w1,w2,w3 placed exactly on the Fourier grid
%   (j = 9, 27, 49) so the "true" variance contribution of each component
%   is known analytically: V_j = A_j^2 / 2.
%
% ASSUMPTIONS:
%   1. Sampling period Ts = 0.4 s/sample (fs = 2.5 samples/s); all spectra
%      here are computed and plotted against discrete-time angular
%      frequency omega (rad/sample), never mixed with physical Hz.
%   2. The Fourier grid frequencies omega_j = 2*pi*j/N, j=1..(N-1)/2, plus
%      the constant term, form a COMPLETE orthogonal basis of R^N (127
%      columns for N=127 samples): fitting it is a change of coordinates,
%      not a lossy approximation.
%   3. hat_gamma_k = (1/N) * sum_n x[n]*x[n-k] is the BIASED sample
%      autocovariance. It is critical to divide by N, not by the number of
%      overlapping terms (N-k): the periodogram = FT(autocovariance)
%      identity below only holds exactly under the /N normalization.
%
% REPRODUCIBILITY:
%   Fully deterministic (no randomness) — every run reproduces identical
%   numbers to floating-point precision.
%
% See also: exp_synth_leakage.m, exp_synth_stochastic.m, exp_population_spectrum.m

clc; clear; close all;

% Add project root to path to resolve +OceanViz
rootPath = fileparts(fileparts(fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))))));
addpath(rootPath);

resultsDir = fullfile(fileparts(mfilename('fullpath')), 'Results');
if ~exist(resultsDir, 'dir'); mkdir(resultsDir); end

fprintf('=== STAGE 1-6: SYNTHETIC FOURIER / PERIODOGRAM VERIFICATION ===\n\n');

%% --- Core settings ---
N  = 127;                 % odd length: eq. 6.2.6 applies with no Nyquist special case
Ts = 0.4;                 % s/sample
fs = 1 / Ts;               % samples/s
dOmega = 2*pi / N;         % Fourier-grid angular-frequency spacing (rad/sample)

n = (0:N-1)';

mu  = 3.7;
A   = [2.1, 1.35, 0.72];
phi = [0.43, -0.91, 1.27];
jTrue = [9, 27, 49];               % deliberately separated, non-adjacent grid bins
wTrue = 2*pi*jTrue/N;               % rad/sample

y = mu ...
    + A(1)*cos(wTrue(1)*n + phi(1)) ...
    + A(2)*cos(wTrue(2)*n + phi(2)) ...
    + A(3)*cos(wTrue(3)*n + phi(3));

fprintf('N=%d, Ts=%.3f s, fs=%.3f samples/s, dOmega=2*pi/N=%.6f rad/sample\n\n', N, Ts, fs, dOmega);

%% =====================================================================
%  STAGE 1 — Fourier basis reconstructs the sample exactly
%  =====================================================================
fprintf('--- Stage 1: Fourier regression (eq. 6.2.6), full basis ---\n');

J = (N-1)/2;                        % 63 non-DC Fourier-grid frequencies
jGrid = (1:J)';
omegaGrid = 2*pi*jGrid/N;           % [J x 1]

Cmat = cos(n * omegaGrid');         % [N x J]
Smat = sin(n * omegaGrid');         % [N x J]
X = [ones(N,1), Cmat, Smat];        % [N x N] -- complete orthogonal basis of R^N

beta = X \ y;                       % OLS fit of eq. 6.2.6
a0Hat    = beta(1);
alphaHat = beta(2:1+J);
deltaHat = beta(2+J:1+2*J);

yHat = X * beta;
reconErr = max(abs(yHat - y));
fprintf('Reconstruction max |y_hat - y| = %.3e  (expect ~0, machine precision)\n', reconErr);
fprintf('a0_hat = %.10f vs mean(y) = %.10f\n\n', a0Hat, mean(y));

%% =====================================================================
%  STAGE 2 — Orthogonality of the Fourier design columns
%  =====================================================================
fprintf('--- Stage 2: Orthogonality of X (X''X) ---\n');
XtX = X' * X;
offDiagMax = max(max(abs(XtX - diag(diag(XtX)))));
fprintf('max |off-diagonal(X''X)| = %.3e  (expect ~0)\n', offDiagMax);
fprintf('1''1 = %.4f (expect N=%d); c_1''c_1 = %.4f (expect N/2=%.1f); s_1''s_1 = %.4f\n\n', ...
    XtX(1,1), N, XtX(2,2), N/2, XtX(2+J,2+J));

%% =====================================================================
%  STAGE 3 — Sample variance decomposition across frequencies
%  =====================================================================
fprintf('--- Stage 3: Variance decomposition ---\n');
yBar = mean(y);
x = y - yBar;
gamma0Hat = mean(x.^2);             % hat_gamma_0 = (1/N) sum x[n]^2

zComponents = Cmat .* alphaHat' + Smat .* deltaHat';   % [N x J], column j = z_j[n]
V = mean(zComponents.^2, 1)';                          % [J x 1]
Vtheory = 0.5 * (alphaHat.^2 + deltaHat.^2);

fprintf('max |V - V_theory| = %.3e\n', max(abs(V - Vtheory)));
fprintf('gamma0_hat = %.6f, sum(V_j) = %.6f, diff = %.3e\n', gamma0Hat, sum(V), gamma0Hat - sum(V));

expectedTotalV = 0.5*sum(A.^2);
fprintf('Expected total variance from known sinusoids = 0.5*sum(A_j^2) = %.5f\n', expectedTotalV);
fprintf('%-6s %-12s %-12s\n', 'j', 'V_j (hat)', 'A_j^2/2 (true)');
for k = 1:3
    idx = jTrue(k);
    fprintf('%-6d %-12.6f %-12.6f\n', idx, V(idx), A(k)^2/2);
end
fprintf('\n');

%% =====================================================================
%  STAGE 4 — DFT coefficients equal the regression coefficients
%  =====================================================================
fprintf('--- Stage 4: DFT vs regression-coefficient equivalence ---\n');
Xw = fft(x);                        % bin j <-> omega_j = 2*pi*j/N for j=0..N-1

fprintf('%-4s %-10s %-10s %-24s %-24s %-10s\n', 'j', 'alpha_j', 'delta_j', 'X(w_j)', 'predicted N/2*(a-i*d)', '|diff|');
stage4 = struct('j', {}, 'alpha', {}, 'delta', {}, 'Xw', {}, 'pred', {}, 'absdiff', {});
for k = 1:3
    idx = jTrue(k);
    XwJ = Xw(idx+1);                       % +1: MATLAB is 1-indexed, bin idx -> array index idx+1
    pred = (N/2) * (alphaHat(idx) - 1i*deltaHat(idx));
    d = abs(XwJ - pred);
    fprintf('%-4d %-10.4f %-10.4f %-24s %-24s %-10.3e\n', idx, alphaHat(idx), deltaHat(idx), ...
        sprintf('%.3f%+.3fi', real(XwJ), imag(XwJ)), sprintf('%.3f%+.3fi', real(pred), imag(pred)), d);
    stage4(end+1) = struct('j', idx, 'alpha', alphaHat(idx), 'delta', deltaHat(idx), ...
        'Xw', XwJ, 'pred', pred, 'absdiff', d); %#ok<SAGROW>
end
fprintf('\n');

%% =====================================================================
%  STAGE 5 — Periodogram from two independent routes
%  =====================================================================
fprintf('--- Stage 5: Periodogram — autocovariance route vs DFT route ---\n');

% Biased sample autocovariance, lags k = 0..N-1.
% CRITICAL: divide by N (not N-k) -- see header ASSUMPTIONS(3).
gammaHat = zeros(N, 1);
for k = 0:N-1
    gammaHat(k+1) = sum(x(1:N-k) .* x(1+k:N)) / N;
end

sA_grid = zeros(N, 1);
ks = (1:N-1)';
for r = 0:N-1
    w = 2*pi*r/N;
    sA_grid(r+1) = (gammaHat(1) + 2*sum(gammaHat(2:end) .* cos(w*ks))) / (2*pi);
end
sB_grid = (abs(Xw).^2) / (2*pi*N);

maxDiffGrid = max(abs(sA_grid - sB_grid));
fprintf('max |s_A(w_r) - s_B(w_r)| over full Fourier grid = %.3e\n\n', maxDiffGrid);

%% =====================================================================
%  STAGE 6 — Periodogram area equals sample variance
%  =====================================================================
fprintf('--- Stage 6: Periodogram area = sample variance ---\n');
sPositive = sB_grid(2:J+1);              % s(w_j) for j=1..J (positive frequencies only)
areaTerms = 2 * sPositive * dOmega;      % 2*s(w_j)*dOmega

fprintf('max |2*s(w_j)*dOmega - V_j| = %.3e\n', max(abs(areaTerms - V)));
fprintf('sum_j 2*s(w_j)*dOmega = %.6f  vs  gamma0_hat = %.6f  (diff = %.3e)\n\n', ...
    sum(areaTerms), gamma0Hat, sum(areaTerms) - gamma0Hat);

%% =====================================================================
%  Visualization
%  =====================================================================
viz = [];
try
    viz = OceanViz.Core('Theme', 'Dark');
    haveViz = true;
catch
    haveViz = false;
    fprintf('[Info] OceanViz unavailable, using fallback plotting.\n');
end

% --- Figure 1: reconstruction check ---
if haveViz
    fig1 = viz.createFigure('Stage 1 - Fourier Reconstruction');
else
    fig1 = figure('Name', 'Stage 1 - Fourier Reconstruction');
end
ax = gca;
plot(ax, n, y, 'o', 'DisplayName', 'y[n] (original)'); hold(ax, 'on');
plot(ax, n, yHat, '-', 'LineWidth', 1.2, 'DisplayName', 'reconstructed');
xlabel(ax, 'n (sample index)'); ylabel(ax, 'y[n]');
title(ax, sprintf('Stage 1: Fourier reconstruction (max |err| = %.2e)', reconErr));
legend(ax, 'Location', 'best');
if haveViz; viz.styleAxes(ax); end
saveFigure(fig1, fullfile(resultsDir, 'stage1_reconstruction.png'), haveViz, viz);

% --- Figure 2: X'X orthogonality image ---
if haveViz
    fig2 = viz.createFigure('Stage 2 - Orthogonality (X^T X)');
else
    fig2 = figure('Name', 'Stage 2 - Orthogonality (X^T X)');
end
ax = gca;
imagesc(ax, XtX); axis(ax, 'image'); colorbar(ax);
title(ax, 'Stage 2: X^T X (near-diagonal => orthogonal Fourier directions)');
xlabel(ax, 'basis column index'); ylabel(ax, 'basis column index');
if haveViz; viz.styleAxes(ax); end
saveFigure(fig2, fullfile(resultsDir, 'stage2_orthogonality.png'), haveViz, viz);

% --- Figure 3: variance decomposition bar chart ---
if haveViz
    fig3 = viz.createFigure('Stage 3 - Variance Decomposition');
else
    fig3 = figure('Name', 'Stage 3 - Variance Decomposition');
end
ax = gca;
bar(ax, jGrid, V);
xlabel(ax, 'Fourier-grid index j'); ylabel(ax, 'V_j (variance contribution)');
title(ax, sprintf('Stage 3: V_j per frequency (sum = %.4f, gamma0\\_hat = %.4f)', sum(V), gamma0Hat));
if haveViz; viz.styleAxes(ax); end
saveFigure(fig3, fullfile(resultsDir, 'stage3_variance_decomposition.png'), haveViz, viz);

% --- Figure 4: two periodogram routes overlay ---
if haveViz
    fig4 = viz.createFigure('Stage 5 - Periodogram Routes');
else
    fig4 = figure('Name', 'Stage 5 - Periodogram Routes');
end
ax = gca;
wAxis = 2*pi*(0:N-1)/N;
plot(ax, wAxis, sA_grid, 'o', 'DisplayName', 'Route A: FT(autocovariance)'); hold(ax, 'on');
plot(ax, wAxis, sB_grid, '-', 'LineWidth', 1.2, 'DisplayName', 'Route B: |X(\omega)|^2/(2\pi N)');
xlabel(ax, '\omega (rad/sample)'); ylabel(ax, 's(\omega)');
title(ax, sprintf('Stage 5: max route diff = %.2e', maxDiffGrid));
xlim(ax, [0, pi]);
legend(ax, 'Location', 'best');
if haveViz; viz.styleAxes(ax); end
saveFigure(fig4, fullfile(resultsDir, 'stage5_periodogram_routes.png'), haveViz, viz);

%% =====================================================================
%  Compact diagnostic report
%  =====================================================================
reportPath = fullfile(resultsDir, 'report_core.md');
fid = fopen(reportPath, 'w');
fprintf(fid, '# Stage 1-6 Diagnostic Report (exp_synth_core.m)\n\n');
fprintf(fid, '| Experiment | Main diagnostic | Result |\n');
fprintf(fid, '|---|---|---|\n');
fprintf(fid, '| Fourier reconstruction | max\\|y_hat-y\\| | %.3e |\n', reconErr);
fprintf(fid, '| Orthogonality | max off-diag(X''X) | %.3e |\n', offDiagMax);
fprintf(fid, '| Variance decomposition | gamma0_hat vs sum(V_j) | %.6f vs %.6f |\n', gamma0Hat, sum(V));
fprintf(fid, '| DFT equivalence | max\\|X(w_j) - N/2(a-id)\\| (j=9,27,49) | %.3e |\n', max([stage4.absdiff]));
fprintf(fid, '| Two periodogram routes | max\\|s_A - s_B\\| on grid | %.3e |\n', maxDiffGrid);
fprintf(fid, '| Variance/area relation | sum(2 s(w_j) dOmega) vs gamma0_hat | %.6f vs %.6f |\n', sum(areaTerms), gamma0Hat);
fclose(fid);
fprintf('Report written to %s\n', reportPath);

fprintf('\n=== STAGE 1-6 COMPLETE ===\n');

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
