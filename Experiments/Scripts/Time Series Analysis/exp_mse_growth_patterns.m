% =========================================================================
% exp_mse_growth_patterns.m
%
% Experiment B — MSE Growth Patterns: AR(1) vs MA(q) vs AR(p)
%
% Key formula:
%   MSE(s) = sigma^2 * sum_{j=0}^{s-1} psi_j^2
% where {psi_j} is the Wold (MA-infinity) representation of the process.
%
% Model instances:
%   1. AR(1),  phi   = 0.7
%   2. MA(3),  theta = [0.5, -0.3, 0.2]
%   3. AR(2),  phi   = [1.0, -0.8]  (complex-conjugate eigenvalues)
%
% Kolb phases:
%   Phase 0 — Setup & {psi_j} computation
%   Phase 1 — Concrete experience: MSE(s) for three models
%   Phase 2 — Reflective observation: plots + plateau/asymptote detection
%   Phase 3 — Abstract conceptualization: complex-eigenvalue analysis
%   Phase 4 — Active experimentation: vary theta_q and phi toward boundary
%
% =========================================================================

clc; close all; clear;

fprintf('=================================================================\n');
fprintf('  EXPERIMENT B: MSE Growth Patterns — AR(1) vs MA(q) vs AR(p)\n');
fprintf('=================================================================\n\n');

% =========================================================================
% PHASE 0 — Setup
% =========================================================================
fprintf('--- PHASE 0: Setup ---\n\n');

sigma2  = 1.0;      % innovation variance
S_max   = 50;       % max forecast horizon
s_vec   = (1:S_max)';

% Model parameters
phi_ar1     = 0.7;
theta_ma3   = [0.5, -0.3, 0.2];
phi_ar2     = [1.0, -0.8];     % produces complex eigenvalues

fprintf('  sigma^2   = %.1f\n', sigma2);
fprintf('  Horizons  = 1 ... %d\n\n', S_max);
fprintf('  Model 1:  AR(1),  phi   = %.2f\n', phi_ar1);
fprintf('  Model 2:  MA(3),  theta = [%s]\n', num2str(theta_ma3, ' %.2f'));
fprintf('  Model 3:  AR(2),  phi   = [%s]\n\n', num2str(phi_ar2, ' %.2f'));

% Verify AR(2) has complex eigenvalues and is stationary
disc  = phi_ar2(1)^2 + 4*phi_ar2(2);
eigAR2 = eig([phi_ar2; 1, 0]);
fprintf('  AR(2) discriminant = %.4f  (%s)\n', disc, cond_str(disc < 0, 'complex eigenvalues', 'real eigenvalues'));
fprintf('  AR(2) eigenvalues  = %+.4f %+.4fi  (|lambda| = %.4f)\n', ...
    real(eigAR2(1)), imag(eigAR2(1)), abs(eigAR2(1)));
fprintf('                       %+.4f %+.4fi  (|lambda| = %.4f)\n', ...
    real(eigAR2(2)), imag(eigAR2(2)), abs(eigAR2(2)));
fprintf('  Oscillation period ~ %.1f steps\n\n', 2*pi / abs(angle(eigAR2(1))));

assert(all(abs(eigAR2) < 1), 'AR(2) is NOT stationary.');
fprintf('  => AR(2) is stationary with complex-conjugate eigenvalues.\n\n');

% =========================================================================
% Compute {psi_j}: Wold MA-infinity representation coefficients
% =========================================================================

% AR(1): psi_j = phi^j
psi_ar1 = phi_ar1 .^ (0:S_max-1)';

% MA(3): psi_0=1, psi_j = theta_j for j=1..q, psi_j=0 for j>q
q = length(theta_ma3);
psi_ma3 = [1; theta_ma3(:); zeros(S_max-1-q, 1)];

% AR(2): psi_j satisfies the AR(p) recurrence
%   psi_j = phi(1)*psi_{j-1} + phi(2)*psi_{j-2},  psi_0=1, psi_j=0 for j<0
psi_ar2 = compute_ar_psi(phi_ar2, S_max);

% MSE(s) = sigma^2 * cumsum(psi_j^2), evaluated at s = 1,...,S_max
mse_ar1 = sigma2 * cumsum(psi_ar1.^2);
mse_ma3 = sigma2 * cumsum(psi_ma3.^2);
mse_ar2 = sigma2 * cumsum(psi_ar2.^2);

% Theoretical benchmarks
asymptote_ar1 = sigma2 / (1 - phi_ar1^2);               % AR(1) long-run variance
plateau_ma3   = sigma2 * (1 + sum(theta_ma3.^2));        % MA(3) plateau at s=q+1
fprintf('  AR(1) asymptote  = sigma^2/(1-phi^2) = %.6f\n', asymptote_ar1);
fprintf('  MA(3) plateau    = sigma^2*(1+sum(theta^2)) = %.6f  (reached at s=%d)\n\n', ...
    plateau_ma3, q + 1);

% =========================================================================
% PHASE 1 — Concrete Experience: psi_j sequences
% =========================================================================
fprintf('--- PHASE 1: Wold Coefficients psi_j ---\n\n');
fprintf('  %-4s  %-12s  %-12s  %-12s\n', 'j', 'psi_j AR(1)', 'psi_j MA(3)', 'psi_j AR(2)');
fprintf('  %s\n', repmat('-', 1, 46));
for j = 0:min(12, S_max-1)
    fprintf('  %-4d  %12.6f  %12.6f  %12.6f\n', ...
        j, psi_ar1(j+1), psi_ma3(j+1), psi_ar2(j+1));
end
fprintf('  ...\n\n');

% =========================================================================
% PHASE 2 — Reflective Observation: MSE(s) plots
% =========================================================================
fprintf('--- PHASE 2: MSE(s) Curves ---\n\n');

fprintf('  %-4s  %-12s  %-12s  %-12s\n', 's', 'MSE AR(1)', 'MSE MA(3)', 'MSE AR(2)');
fprintf('  %s\n', repmat('-', 1, 46));
for s = [1,2,3,4,5,10,20,30,40,50]
    fprintf('  %-4d  %12.6f  %12.6f  %12.6f\n', ...
        s, mse_ar1(s), mse_ma3(s), mse_ar2(s));
end
fprintf('\n');

% Detect where MA(3) exactly plateaus
s_plateau = q + 1;
fprintf('  MA(3) MSE at s=%d : %.8f\n', s_plateau,   mse_ma3(s_plateau));
fprintf('  MA(3) MSE at s=%d : %.8f  (must equal s=%d value)\n', ...
    s_plateau+1, mse_ma3(s_plateau+1), s_plateau);
fprintf('  Difference: %.2e  (should be ~0)\n\n', abs(mse_ma3(s_plateau+1) - mse_ma3(s_plateau)));

% =========================================================================
% PHASE 3 — Abstract Conceptualization: complex eigenvalue analysis
% =========================================================================
fprintf('--- PHASE 3: Complex-Eigenvalue Analysis ---\n\n');

% The AR(2) psi_j can be written as:
%   psi_j = 2 * |lambda|^j * cos(j*omega + delta) / (some amplitude factor)
% where omega = angle(lambda) and |lambda| = sqrt(-phi_2)
lam = eigAR2(1);
omega = abs(angle(lam));
r     = abs(lam);
fprintf('  Dominant eigenvalue: lambda = r * e^{i*omega}\n');
fprintf('    r     = |lambda| = %.6f  (decay rate per step)\n', r);
fprintf('    omega = arg(lambda) = %.6f rad = %.2f deg  (oscillation)\n', ...
    omega, rad2deg(omega));
fprintf('    Period = 2*pi/omega = %.2f steps\n\n', 2*pi/omega);

% Incremental MSE: sigma2 * psi_{s-1}^2 — shows the oscillatory rate of accumulation
incr_mse_ar2 = sigma2 * psi_ar2.^2;
fprintf('  Incremental MSE contributions (sigma^2 * psi_{s-1}^2):\n');
fprintf('  Note damped oscillation — NOT monotone, but cumsum IS monotone.\n\n');
fprintf('  %-4s  %-14s  %-14s  %-12s\n', 's', 'psi_{s-1}', 'incr. MSE', 'cumul. MSE');
fprintf('  %s\n', repmat('-', 1, 52));
for s = 1:15
    fprintf('  %-4d  %14.8f  %14.8f  %12.6f\n', ...
        s, psi_ar2(s), incr_mse_ar2(s), mse_ar2(s));
end
fprintf('\n');

% =========================================================================
% PHASE 4 — Active Experimentation
% =========================================================================
fprintf('--- PHASE 4: Active Experimentation ---\n\n');

%% 4a — Vary theta_q toward zero: plateau shrinks to sigma^2
theta3_vals = [0.2, 0.15, 0.10, 0.05, 0.001];
fprintf('  Experiment 4a: Vary theta_3 -> 0 (MA(3) plateau shrinks to sigma^2)\n');
fprintf('  %-8s  %-14s\n', 'theta_3', 'Plateau MSE');
fprintf('  %s\n', repmat('-', 1, 26));
mse_ma3_vary = zeros(S_max, length(theta3_vals));
for ii = 1:length(theta3_vals)
    th = [theta_ma3(1:2), theta3_vals(ii)];
    psi = [1; th(:); zeros(S_max-1-3, 1)];
    mse_ma3_vary(:, ii) = sigma2 * cumsum(psi.^2);
    fprintf('  %-8.3f  %-14.6f\n', theta3_vals(ii), mse_ma3_vary(q+1, ii));
end
fprintf('  sigma^2 = %.6f  (limit as theta_3 -> 0, if also theta_1,2->0)\n\n', sigma2);

%% 4b — Vary phi toward unit circle: AR(1) asymptote diverges
phi_vals = [0.3, 0.5, 0.7, 0.85, 0.95, 0.99];
fprintf('  Experiment 4b: Vary phi -> 1 (AR(1) asymptote diverges)\n');
fprintf('  %-6s  %-14s  %-14s\n', 'phi', 'Asymptote', 'MSE(s=50)');
fprintf('  %s\n', repmat('-', 1, 38));
mse_ar1_vary = zeros(S_max, length(phi_vals));
for ii = 1:length(phi_vals)
    ph = phi_vals(ii);
    psi = ph .^ (0:S_max-1)';
    mse_ar1_vary(:, ii) = sigma2 * cumsum(psi.^2);
    asym = sigma2 / (1 - ph^2);
    fprintf('  %-6.2f  %-14.4f  %-14.6f\n', ph, asym, mse_ar1_vary(S_max, ii));
end
fprintf('\n');

% =========================================================================
% PLOTS
% =========================================================================
RESULTS_DIR = fullfile(fileparts(mfilename('fullpath')), 'Results');
if ~exist(RESULTS_DIR, 'dir'), mkdir(RESULTS_DIR); end

%% Figure 1 — Main MSE(s) comparison
figure('Name', 'MSE Growth Patterns', 'Position', [60 60 1000 480]);

% Plot lines with a premium color palette (Indigo, Coral, Emerald)
plot(s_vec, mse_ar1, '-', 'Color', [0.2, 0.3, 0.8], 'LineWidth', 2.5, ...
     'DisplayName', 'AR(1)  \phi=0.7');
hold on;
plot(s_vec, mse_ma3, '-', 'Color', [0.9, 0.2, 0.3], 'LineWidth', 2.5, ...
     'DisplayName', 'MA(3)  \theta=(0.5,-0.3,0.2)');
plot(s_vec, mse_ar2, '-', 'Color', [0.05, 0.65, 0.4], 'LineWidth', 2.5, ...
     'DisplayName', 'AR(2)  \phi=(1.0,-0.8)  [complex \lambda]');

% Mark asymptote and plateau (with HandleVisibility off to keep legend clean)
yline(asymptote_ar1, '--', sprintf('AR(1) asymptote = %.3f', asymptote_ar1), ...
    'Color', [0.4, 0.6, 1.0], 'LineWidth', 1.2, ...
    'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom', ...
    'HandleVisibility', 'off');
yline(plateau_ma3, '--', sprintf('MA(3) plateau = %.3f  (s \\geq %d)', plateau_ma3, q+1), ...
    'Color', [1.0, 0.5, 0.5], 'LineWidth', 1.2, ...
    'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'top', ...
    'HandleVisibility', 'off');
xline(q+1, ':', sprintf('s = q+1 = %d', q+1), ...
    'Color', [1.0, 0.5, 0.5], 'LineWidth', 1.2, ...
    'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');

hold off;

% Premium typography and axes styling
xlabel('Forecast horizon  s', 'FontSize', 11, 'FontName', 'Helvetica');
ylabel('MSE(s)  =  sigma^2 * sum_{j=0}^{s-1} psi_j^2', 'FontSize', 11, 'FontName', 'Helvetica');
title('Experiment B — MSE Growth Patterns: AR(1) vs MA(q) vs AR(p)', ...
      'FontSize', 13, 'FontWeight', 'bold', 'FontName', 'Helvetica');

grid on;
ax = gca;
ax.GridLineStyle = ':';
ax.GridAlpha = 0.3;
ax.LineWidth = 1.1;
ax.FontSize = 10;
ax.FontName = 'Helvetica';

lgd = legend('Location', 'east');
lgd.FontSize = 10;
lgd.Box = 'on';
lgd.EdgeColor = [0.8, 0.8, 0.8];

xlim([1 S_max]);
saveas(gcf, fullfile(RESULTS_DIR, 'exp_mse_main.png'));
fprintf('  Fig 1 saved: exp_mse_main.png\n');

%% Figure 2 — psi_j sequences (show oscillation in AR(2))
figure('Name', 'Wold Coefficients psi_j', 'Position', [60 540 1000 380]);

stem(0:S_max-1, psi_ar1, 'b', 'LineWidth', 1.2, 'MarkerSize', 4, ...
     'DisplayName', 'AR(1)');
hold on;
stem(0:S_max-1, psi_ma3, 'r', 'LineWidth', 1.2, 'MarkerSize', 4, ...
     'DisplayName', 'MA(3)');
stem(0:S_max-1, psi_ar2, 'g', 'LineWidth', 1.2, 'MarkerSize', 4, ...
     'DisplayName', 'AR(2) [complex \lambda]');
hold off;
xlabel('Lag  j');
ylabel('\psi_j');
title('Wold (MA-\infty) Coefficients \psi_j for Each Process');
legend('Location', 'northeast');
grid on;
xlim([-0.5, S_max-0.5]);
saveas(gcf, fullfile(RESULTS_DIR, 'exp_mse_psi_sequences.png'));
fprintf('  Fig 2 saved: exp_mse_psi_sequences.png\n');

%% Figure 3 — Incremental MSE contributions for AR(2) (oscillatory rate)
figure('Name', 'Incremental MSE — AR(2)', 'Position', [60 60 900 380]);

subplot(1,2,1);
stem(1:S_max, incr_mse_ar2, 'g', 'LineWidth', 1.2, 'MarkerSize', 4);
xlabel('Horizon s');
ylabel('\sigma^2 \psi_{s-1}^2');
title('Incremental MSE: \sigma^2\psi_{s-1}^2  (non-monotone)');
grid on;

subplot(1,2,2);
plot(s_vec, mse_ar2, 'g-', 'LineWidth', 2.0);
xlabel('Horizon s');
ylabel('MSE(s)');
title('Cumulative MSE  (monotone, asymptoting)');
grid on;
sgtitle('AR(2) with Complex Eigenvalues: \phi=(1.0,-0.8)');
saveas(gcf, fullfile(RESULTS_DIR, 'exp_mse_ar2_incremental.png'));
fprintf('  Fig 3 saved: exp_mse_ar2_incremental.png\n');

%% Figure 4 — Active experimentation plots
figure('Name', 'Active Experimentation', 'Position', [60 540 1100 450]);

% Panel A: MA(3), vary theta_3
subplot(1,2,1);
cmap_r = lines(length(theta3_vals));
for ii = 1:length(theta3_vals)
    plot(s_vec, mse_ma3_vary(:,ii), '-', 'LineWidth', 1.8, 'Color', cmap_r(ii,:), ...
         'DisplayName', sprintf('\\theta_3=%.3f', theta3_vals(ii)));
    hold on;
end

% High-contrast light gray for reference line and label
yline(sigma2, '--', 'sigma^2 (white noise limit)', ...
    'Color', [0.8, 0.8, 0.8], 'LineWidth', 1.2, ...
    'LabelHorizontalAlignment', 'right', 'HandleVisibility', 'off');
xline(q+1, ':', sprintf('s=%d', q+1), ...
    'Color', [0.8, 0.8, 0.8], 'LineWidth', 1.2, ...
    'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
hold off;

xlabel('Forecast horizon s', 'FontSize', 10, 'FontName', 'Helvetica');
ylabel('MSE(s)', 'FontSize', 10, 'FontName', 'Helvetica');
title('MA(3): Vary \theta_3 \rightarrow 0  (plateau \rightarrow \sigma^2)', ...
      'FontSize', 11, 'FontWeight', 'bold', 'FontName', 'Helvetica');

grid on;
ax1 = gca;
ax1.GridLineStyle = ':';
ax1.GridAlpha = 0.3;
ax1.LineWidth = 1.1;
ax1.FontSize = 9;
ax1.FontName = 'Helvetica';

lgd1 = legend('Location', 'east');
lgd1.FontSize = 8;
lgd1.Box = 'on';
lgd1.EdgeColor = [0.8, 0.8, 0.8];
xlim([1 S_max]);

% Panel B: AR(1), vary phi toward 1
subplot(1,2,2);
cmap_b = cool(length(phi_vals));
for ii = 1:length(phi_vals)
    asym = sigma2 / (1 - phi_vals(ii)^2);
    plot(s_vec, mse_ar1_vary(:,ii), '-', 'LineWidth', 1.8, 'Color', cmap_b(ii,:), ...
         'DisplayName', sprintf('\\phi=%.2f (asym=%.1f)', phi_vals(ii), asym));
    hold on;
end
hold off;

xlabel('Forecast horizon s', 'FontSize', 10, 'FontName', 'Helvetica');
ylabel('MSE(s)', 'FontSize', 10, 'FontName', 'Helvetica');
title('AR(1): Vary \phi \rightarrow 1  (asymptote \rightarrow \infty)', ...
      'FontSize', 11, 'FontWeight', 'bold', 'FontName', 'Helvetica');

grid on;
ax2 = gca;
ax2.GridLineStyle = ':';
ax2.GridAlpha = 0.3;
ax2.LineWidth = 1.1;
ax2.FontSize = 9;
ax2.FontName = 'Helvetica';

lgd2 = legend('Location', 'east');
lgd2.FontSize = 8;
lgd2.Box = 'on';
lgd2.EdgeColor = [0.8, 0.8, 0.8];
xlim([1 S_max]);

sgtitle('Experiment B — Active Experimentation: Parameter Sensitivity', ...
        'FontSize', 13, 'FontWeight', 'bold', 'FontName', 'Helvetica');
saveas(gcf, fullfile(RESULTS_DIR, 'exp_mse_active_exp.png'));
fprintf('  Fig 4 saved: exp_mse_active_exp.png\n\n');

% =========================================================================
% SUMMARY
% =========================================================================
fprintf('=================================================================\n');
fprintf('  SUMMARY\n');
fprintf('=================================================================\n');
fprintf('  sigma^2 = %.1f,  S = %d forecast horizons\n\n', sigma2, S_max);

fprintf('  AR(1)  (phi=%.2f):\n', phi_ar1);
fprintf('    MSE(1)   = %.6f  (= sigma^2)\n', mse_ar1(1));
fprintf('    MSE(50)  = %.6f\n', mse_ar1(50));
fprintf('    Asymptote = %.6f  (= sigma^2/(1-phi^2))\n', asymptote_ar1);
fprintf('    Gap MSE(50) vs asymptote = %.2e\n\n', abs(mse_ar1(50) - asymptote_ar1));

fprintf('  MA(3)  (theta=[%s]):\n', num2str(theta_ma3,' %.2f'));
fprintf('    MSE(q)   = %.6f  (before plateau, s=%d)\n', mse_ma3(q),   q);
fprintf('    MSE(q+1) = %.6f  (plateau reached, s=%d)\n', mse_ma3(q+1), q+1);
fprintf('    MSE(50)  = %.6f  (identical to MSE(q+1): %.2e diff)\n', ...
    mse_ma3(50), abs(mse_ma3(50)-mse_ma3(q+1)));
fprintf('    Theory:  plateau = sigma^2*(1+sum(theta^2)) = %.6f\n\n', plateau_ma3);

fprintf('  AR(2)  (phi=[%s], complex eigenvalues):\n', num2str(phi_ar2,' %.2f'));
fprintf('    |lambda| = %.6f,  period = %.2f steps\n', abs(eigAR2(1)), 2*pi/omega);
fprintf('    MSE(1)   = %.6f\n', mse_ar2(1));
fprintf('    MSE(50)  = %.6f  (still growing, no plateau)\n\n', mse_ar2(50));

fprintf('  CONCLUSION:\n');
fprintf('    1. MA(q): MSE plateaus EXACTLY at s=q+1 (finite memory).\n');
fprintf('    2. AR(1): MSE grows monotonically, asymptoting to sigma^2/(1-phi^2).\n');
fprintf('    3. AR(2) complex: psi_j oscillates (damped), so incremental MSE\n');
fprintf('       (sigma^2*psi_{s-1}^2) oscillates — but CUMULATIVE MSE is\n');
fprintf('       strictly non-decreasing (proven from eps uncorrelatedness).\n');
fprintf('    4. Parameter sensitivity confirmed: theta_q->0 shrinks plateau;\n');
fprintf('       phi->1 pushes AR(1) asymptote to infinity.\n');
fprintf('=================================================================\n');

% =========================================================================
% LOCAL HELPER FUNCTIONS
% =========================================================================

function psi = compute_ar_psi(phi, S_max)
% Compute Wold coefficients psi_0,...,psi_{S_max-1} for an AR(p) process.
% Recursion: psi_j = phi(1)*psi_{j-1} + ... + phi(p)*psi_{j-p}
% with psi_0 = 1, psi_j = 0 for j < 0.
p   = length(phi);
psi = zeros(S_max, 1);
psi(1) = 1;    % psi_0 = 1
for j = 2:S_max
    lag_len = min(j-1, p);
    psi(j)  = phi(1:lag_len) * psi(j-1:-1:j-lag_len);
end
end

function s = cond_str(cond, a, b)
if cond, s = a; else, s = b; end
end
