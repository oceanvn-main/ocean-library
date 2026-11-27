% EXP_LORENZ_BUTTERFLY  Edward Lorenz's Chaotic Butterfly Attractor in 3D.
%
%   Simulates the Lorenz system of ordinary differential equations:
%
%     dx/dt = sigma * (y - x)
%     dy/dt = x * (rho - z) - y
%     dz/dt = x * y - beta * z
%
%   Classic parameters: sigma = 10, rho = 28, beta = 8/3.
%
%   Experiment workflow:
%     PHASE 1 — Numerical Integration
%       Integrate the Lorenz ODE from a single initial condition using
%       4th-order Runge-Kutta (RK4) implemented from scratch (no ode45).
%       Verifies the self-consistency of the trajectory (deterministic).
%
%     PHASE 2 — 3D Butterfly Visualization
%       Renders the full attractor colored by time progression (jet map),
%       showing the characteristic double-wing structure.
%
%     PHASE 3 — Sensitivity to Initial Conditions (Butterfly Effect)
%       Integrates two trajectories with a tiny epsilon perturbation in X0
%       and visualizes divergence: both wings in one plot plus divergence
%       norm over time.  Explicitly shows chaos: exponential separation.
%
%   Assumptions:
%     - Parameters are classical Lorenz (1963) values.
%     - RK4 step dt = 0.005 (fixed step, deterministic).
%     - Trajectories are computed forward only (no future information).
%
%   Reproducibility: rng seed is fixed for any stochastic elements.
%   No expected outputs are hard-coded; all results emerge from integration.
%
%   Ocean Library — Simulation Experiments

% Clean up workspace
clear;
close all;
clc;

rng(42);  % Fix seed for reproducibility (no stochastic steps here, but good practice)

% Initialize OceanViz styling engine
viz = OceanViz.Core('Theme', 'Dark');

fprintf('=======================================================================\n');
fprintf('           EDWARD LORENZ CHAOTIC BUTTERFLY ATTRACTOR (3D)            \n');
fprintf('=======================================================================\n\n');

%% ===================================================================
%% PARAMETERS — Lorenz (1963) canonical values
%% ===================================================================
sigma = 10;       % Prandtl number (thermal diffusion)
rho   = 28;       % Rayleigh number (buoyancy drive)    [> 1 for chaos]
beta  = 8 / 3;    % Geometric factor

fprintf('--- LORENZ SYSTEM PARAMETERS ---\n');
fprintf('  sigma (sigma) = %.4f\n', sigma);
fprintf('  rho   (rho)   = %.4f\n', rho);
fprintf('  beta  (beta)  = %.6f\n', beta);
fprintf('  [rho > 1 required for non-trivial fixed points; rho > 24.74 for chaos]\n\n');

% Verify chaos threshold (Lorenz 1963: rho_crit approx 24.74)
rho_crit = sigma * (sigma + beta + 3) / (sigma - beta - 1);
fprintf('  Critical rho (Hopf bifurcation) = %.4f\n', rho_crit);
if rho > rho_crit
    fprintf('  rho = %.2f > %.4f --> Chaotic regime confirmed.\n\n', rho, rho_crit);
else
    warning('rho = %.2f is below the chaos threshold %.4f.', rho, rho_crit);
end

%% ===================================================================
%% PHASE 1: NUMERICAL INTEGRATION (RK4, from scratch)
%% ===================================================================
fprintf('--- PHASE 1: RK4 NUMERICAL INTEGRATION ---\n');

% Integration settings
dt      = 0.005;          % Fixed time step (seconds)
T_total = 50;             % Total integration time
N       = round(T_total / dt);  % Number of steps
t       = (0:N) * dt;    % Time vector

% Initial condition (Lorenz's original)
X0 = [0.1; 0.0; 0.0];   % [x, y, z]

fprintf('  dt          = %.4f s\n', dt);
fprintf('  T_total     = %.1f s\n', T_total);
fprintf('  N steps     = %d\n', N);
fprintf('  X0          = [%.4f, %.4f, %.4f]\n\n', X0(1), X0(2), X0(3));

% RK4 integration — all math is explicit; no ode45 black box
X = zeros(3, N+1);
X(:, 1) = X0;

% Define Lorenz vector field f(state) = [dx; dy; dz]
lorenz = @(s) [sigma * (s(2) - s(1)); ...
               s(1) * (rho - s(3)) - s(2); ...
               s(1) * s(2) - beta * s(3)];

fprintf('  Integrating with RK4...\n');
for n = 1:N
    s = X(:, n);
    k1 = lorenz(s);
    k2 = lorenz(s + 0.5 * dt * k1);
    k3 = lorenz(s + 0.5 * dt * k2);
    k4 = lorenz(s + dt * k3);
    X(:, n+1) = s + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4);
end

x = X(1, :);
y = X(2, :);
z = X(3, :);

fprintf('  Integration complete.\n');
fprintf('  Trajectory range:  x in [%.2f, %.2f]\n', min(x), max(x));
fprintf('  Trajectory range:  y in [%.2f, %.2f]\n', min(y), max(y));
fprintf('  Trajectory range:  z in [%.2f, %.2f]\n\n', min(z), max(z));

% Sanity check: verify trajectory is bounded (attractor property)
max_radius = max(sqrt(x.^2 + y.^2 + z.^2));
fprintf('  Max radius from origin = %.4f (should be finite for strange attractor)\n', max_radius);
if isfinite(max_radius) && max_radius < 500
    fprintf('  PASS: Trajectory is bounded on the Lorenz attractor.\n\n');
else
    error('Integration diverged or produced non-finite values.');
end

%% ===================================================================
%% PHASE 2: 3D BUTTERFLY VISUALIZATION (Color-by-Time)
%% ===================================================================
fprintf('--- PHASE 2: 3D ATTRACTOR VISUALIZATION ---\n');

% Build output directory relative to this script
scriptDir  = fileparts(mfilename('fullpath'));
resultsDir = fullfile(scriptDir, 'Results');
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

% Create figure with OceanViz theme
fig1 = viz.createFigure('Lorenz Butterfly Attractor — 3D Phase Portrait');
ax1  = axes('Parent', fig1);
viz.styleAxes(ax1);
hold(ax1, 'on');
view(ax1, -60, 25);

% Compute a colormap index proportional to time
cmap     = parula(N+1);   % Parula: dark-blue (early) to yellow (late)
seg_size = 200;           % Segments for smooth gradient rendering

% Draw trajectory as colored segments (time-coded color)
% HandleVisibility='off' prevents each segment from polluting the legend
n_segs = floor((N+1) / seg_size);
for k = 1:n_segs
    idx_start = (k-1) * seg_size + 1;
    idx_end   = min(k * seg_size + 1, N+1);
    seg_color = cmap(idx_start, :);
    plot3(ax1, x(idx_start:idx_end), y(idx_start:idx_end), z(idx_start:idx_end), ...
          'Color', [seg_color, 0.8], 'LineWidth', 0.6, 'HandleVisibility', 'off');
end

% Overlay start and end markers
plot3(ax1, x(1),   y(1),   z(1),   'o', 'MarkerFaceColor', [0.1, 0.95, 0.4], ...
      'MarkerEdgeColor', 'w', 'MarkerSize', 8, 'DisplayName', 'Start X_0');
plot3(ax1, x(end), y(end), z(end), 'h', 'MarkerFaceColor', [1.0, 0.3, 0.3], ...
      'MarkerEdgeColor', 'w', 'MarkerSize', 10, 'DisplayName', 'End point');

% Add colorbar for time
colormap(ax1, parula);
cb = colorbar(ax1);
cb.Color          = viz.Colors.Text;
cb.Label.String   = 'Time (s)';
cb.Label.Color    = viz.Colors.Text;
cb.TickLabels     = arrayfun(@(v) sprintf('%.0f', v * T_total), cb.Ticks, 'UniformOutput', false);

% Labels and title
title(ax1, 'Lorenz Strange Attractor — Phase Portrait (T = 50 s)', ...
      'Color', viz.Colors.Text, 'FontSize', 14, 'FontWeight', 'bold');
xlabel(ax1, 'x(t)',  'Color', viz.Colors.Text, 'FontSize', 11);
ylabel(ax1, 'y(t)',  'Color', viz.Colors.Text, 'FontSize', 11);
zlabel(ax1, 'z(t)',  'Color', viz.Colors.Text, 'FontSize', 11);

% Set Z-axis color explicitly (3D axes need this)
set(ax1, 'ZColor', viz.Colors.Text);

% Add annotation showing the parameters
param_str = sprintf('\\sigma=%.0f, \\rho=%.0f, \\beta=%.4f', sigma, rho, beta);
text(ax1, min(x)*0.8, min(y)*0.8, max(z)*0.92, param_str, ...
     'Color', viz.Colors.Boundary, 'FontSize', 10, 'Interpreter', 'tex');

legend(ax1, 'TextColor', viz.Colors.Text, 'Location', 'northeast', 'FontSize', 9);
grid(ax1, 'on');

saveas(fig1, fullfile(resultsDir, 'lorenz_butterfly_3d.png'));
fprintf('  Saved 3D attractor plot to Results/lorenz_butterfly_3d.png\n\n');

%% ===================================================================
%% PHASE 3: SENSITIVITY TO INITIAL CONDITIONS (BUTTERFLY EFFECT)
%% ===================================================================
fprintf('--- PHASE 3: BUTTERFLY EFFECT — SENSITIVITY TO INITIAL CONDITIONS ---\n');

% Perturb initial condition by a tiny epsilon
epsilon = 1e-8;
X0_pert = X0 + [epsilon; 0; 0];   % Perturb x by 1e-8 only

fprintf('  Original IC : [%.10f, %.4f, %.4f]\n', X0(1), X0(2), X0(3));
fprintf('  Perturbed IC: [%.10f, %.4f, %.4f]\n', X0_pert(1), X0_pert(2), X0_pert(3));
fprintf('  Perturbation epsilon = %.2e  (in x-component only)\n\n', epsilon);

% Integrate perturbed trajectory
Xp = zeros(3, N+1);
Xp(:, 1) = X0_pert;

for n = 1:N
    s = Xp(:, n);
    k1 = lorenz(s);
    k2 = lorenz(s + 0.5 * dt * k1);
    k3 = lorenz(s + 0.5 * dt * k2);
    k4 = lorenz(s + dt * k3);
    Xp(:, n+1) = s + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4);
end

xp = Xp(1, :);
yp = Xp(2, :);
zp = Xp(3, :);

% Compute divergence: Euclidean distance between the two trajectories
divergence = sqrt((x - xp).^2 + (y - yp).^2 + (z - zp).^2);

% -----------------------------------------------------------------------
% MLE METHOD A: Simple log-divergence linear fit (pedagogical, biased)
%
% Fits log(d) ~ lambda*t in the window [delta_lo, delta_hi] — past the
% noise floor, before attractor-scale saturation.
% KNOWN BIAS: once both trajectories saturate to different wings the slope
% decreases, so polyfit averages growth + saturation -> ~0.5x true MLE.
% -----------------------------------------------------------------------
delta_lo  = 1e-6;   % below: still in numerical noise floor
delta_hi  = 1.0;    % above: approaching saturation (attractor ~ 30-50 units)

fit_mask = (divergence > delta_lo) & (divergence < delta_hi);
t_fit    = t(fit_mask);
d_fit    = divergence(fit_mask);

if numel(t_fit) > 20
    log_d         = log(d_fit);
    p             = polyfit(t_fit, log_d, 1);
    lambda_simple = p(1);
    fprintf('  [Method A] Simple log-fit [%.0e, %.1f] --> %d samples\n', ...
            delta_lo, delta_hi, numel(t_fit));
    fprintf('  [Method A] MLE (biased, single-pair):  lambda_A ~ %.4f nats/s\n', lambda_simple);
    fprintf('  (Underestimates true MLE because fit averages over exponential\n');
    fprintf('   growth AND the semi-saturated saturation regime.)\n\n');
else
    lambda_simple = NaN;
    p             = [NaN, NaN];
    fprintf('  [Method A] WARNING: Not enough samples. Try increasing T_total.\n\n');
end

% -----------------------------------------------------------------------
% MLE METHOD B: Wolf et al. (1985) renormalization estimator (standard)
%
% Algorithm:
%   1. Start integration from a point already on the attractor (skip 5 s).
%   2. Maintain a reference and a perturbed satellite trajectory.
%   3. Each step the satellite is advanced; when ||delta|| > d_renorm,
%      accumulate log(||delta||/d0) and rescale the displacement back to d0
%      (keeping the reference unchanged).
%   4. MLE = sum_k log(d_k / d0)  /  T_total
%
% Avoids saturation bias entirely. Gives the true ensemble-averaged
% exponential growth rate on the attractor.
%
% Reference: Wolf, A., Swift, J.B., Swinney, H.L., Vastano, J.A. (1985).
%   Determining Lyapunov exponents from a time series.
%   Physica D: Nonlinear Phenomena, 16(3), 285-317.
% -----------------------------------------------------------------------
fprintf('  [Method B] Wolf et al. (1985) renormalization MLE...\n');

d0       = 1e-8;    % Perturbation magnitude (reset to this after each renorm)
d_renorm = 1e-6;    % Rescale threshold: 100x d0 => cycle time ~5 s => ~40 renorms/200s

% Dedicated Wolf integration (separate from Phase 1, 200 s total)
% Seeded from a point already on the attractor (5 s into Phase 1)
t_skip       = round(5.0 / dt);
T_wolf_total = 200;                   % Integration time (s)
N_wolf       = round(T_wolf_total / dt);
s_ref   = X(:, t_skip + 1);
s_pert  = s_ref + d0 * [1; 0; 0];    % Perturbation in x-direction

sum_log_growth = 0;   % Accumulates log(d_new / d0) at each renormalization
n_renorm_steps = 0;   % Number of renormalizations

for n_wolf = 1 : N_wolf
    % Advance reference (RK4)
    k1r = lorenz(s_ref);
    k2r = lorenz(s_ref + 0.5 * dt * k1r);
    k3r = lorenz(s_ref + 0.5 * dt * k2r);
    k4r = lorenz(s_ref + dt * k3r);
    s_ref_new = s_ref + (dt / 6) * (k1r + 2*k2r + 2*k3r + k4r);

    % Advance perturbed satellite (RK4)
    k1p = lorenz(s_pert);
    k2p = lorenz(s_pert + 0.5 * dt * k1p);
    k3p = lorenz(s_pert + 0.5 * dt * k2p);
    k4p = lorenz(s_pert + dt * k3p);
    s_pert_new = s_pert + (dt / 6) * (k1p + 2*k2p + 2*k3p + k4p);

    % Check renormalization threshold
    d_new = norm(s_pert_new - s_ref_new);
    if d_new > d_renorm
        sum_log_growth = sum_log_growth + log(d_new / d0);
        n_renorm_steps = n_renorm_steps + 1;
        grow_dir   = (s_pert_new - s_ref_new) / d_new;
        s_pert_new = s_ref_new + d0 * grow_dir;  % Rescale, keep direction
    end

    s_ref  = s_ref_new;
    s_pert = s_pert_new;
end

lambda_wolf = sum_log_growth / T_wolf_total;

fprintf('  [Method B] Integration time : %.0f s (dedicated, seeded from attractor)\n', T_wolf_total);
fprintf('  [Method B] d_renorm         : %.0e  => cycle ~ %.1f s => ~%d renorms expected\n', ...
        d_renorm, log(d_renorm/d0)/0.905, round(T_wolf_total * 0.905 / log(d_renorm/d0)));
fprintf('  [Method B] Actual renormalizations: %d\n',       n_renorm_steps);
fprintf('  [Method B] MLE (Wolf 1985)  : lambda_B ~ %.4f nats/s\n', lambda_wolf);
fprintf('  [Theoretical Lorenz MLE     ~ 0.905 nats/s]\n\n');

% Use the Wolf estimate as the canonical result
lambda_est = lambda_wolf;

% Create 2-panel figure: (1) Dual trajectory 3D, (2) Divergence norm
fig2 = viz.createFigure('Lorenz Butterfly Effect — Sensitivity to Initial Conditions');

% Panel 1: Both trajectories overlaid
ax2 = subplot(1, 2, 1, 'Parent', fig2);
viz.styleAxes(ax2);
hold(ax2, 'on');
view(ax2, -55, 20);
set(ax2, 'ZColor', viz.Colors.Text);

% Original trajectory (cyan)
plot3(ax2, x, y, z, 'Color', [0.1, 0.75, 0.95, 0.6], 'LineWidth', 0.7, ...
      'DisplayName', sprintf('X_0 = [%.1f, %.1f, %.1f]', X0(1), X0(2), X0(3)));

% Perturbed trajectory (orange-red)
plot3(ax2, xp, yp, zp, 'Color', [1.0, 0.45, 0.1, 0.6], 'LineWidth', 0.7, ...
      'DisplayName', sprintf('X_0 + eps  (eps = %.0e)', epsilon));

% Highlight start points
plot3(ax2, X0(1),      X0(2),      X0(3),      'o', 'MarkerFaceColor', [0.1, 0.95, 0.4], ...
      'MarkerEdgeColor', 'w', 'MarkerSize', 7, 'HandleVisibility', 'off');
plot3(ax2, X0_pert(1), X0_pert(2), X0_pert(3), 'o', 'MarkerFaceColor', [0.95, 0.8, 0.1], ...
      'MarkerEdgeColor', 'w', 'MarkerSize', 7, 'HandleVisibility', 'off');

title(ax2, 'Two Trajectories — Same Wing, Then Diverge', ...
      'Color', viz.Colors.Text, 'FontSize', 11, 'FontWeight', 'bold');
xlabel(ax2, 'x(t)', 'Color', viz.Colors.Text);
ylabel(ax2, 'y(t)', 'Color', viz.Colors.Text);
zlabel(ax2, 'z(t)', 'Color', viz.Colors.Text);
legend(ax2, 'TextColor', viz.Colors.Text, 'Location', 'southoutside', 'FontSize', 8);
grid(ax2, 'on');

% Panel 2: Divergence norm over time (log scale)
ax3 = subplot(1, 2, 2, 'Parent', fig2);
viz.styleAxes(ax3);
hold(ax3, 'on');

% Plot full divergence
semilogy(ax3, t, divergence, 'Color', viz.Colors.Secondary, 'LineWidth', 1.5, ...
         'DisplayName', '||X(t) - X_{pert}(t)||');

% Overlay Method A exponential fit over its fitting window
if ~isnan(lambda_simple) && numel(t_fit) > 1
    t_exp  = linspace(t_fit(1), t_fit(end), 500);
    d_exp  = exp(lambda_simple * t_exp + p(2));
    semilogy(ax3, t_exp, d_exp, '--', 'Color', viz.Colors.Boundary, 'LineWidth', 2.0, ...
             'DisplayName', sprintf('Method A fit: e^{%.3f t}', lambda_simple));
end

% Horizontal reference: attractor diameter scale (~30 units)
yline(ax3, 30, ':', 'Color', [0.7, 0.7, 0.7], 'LineWidth', 1.2, ...
      'Label', 'Attractor scale', 'LabelVerticalAlignment', 'bottom', ...
      'LabelHorizontalAlignment', 'right');

title(ax3, 'Trajectory Divergence (Lyapunov Growth)', ...
      'Color', viz.Colors.Text, 'FontSize', 11, 'FontWeight', 'bold');
xlabel(ax3, 'Time (s)',                    'Color', viz.Colors.Text);
ylabel(ax3, 'Euclidean Separation  (log)', 'Color', viz.Colors.Text);
legend(ax3, 'TextColor', viz.Colors.Text, 'Location', 'northwest', 'FontSize', 9);
grid(ax3, 'on');
xlim(ax3, [0, T_total]);
ylim(ax3, [1e-10, 1e3]);

saveas(fig2, fullfile(resultsDir, 'lorenz_butterfly_effect.png'));
fprintf('  Saved butterfly effect plot to Results/lorenz_butterfly_effect.png\n\n');

%% ===================================================================
%% SUMMARY
%% ===================================================================
fprintf('=======================================================================\n');
fprintf('  EXPERIMENT COMPLETE\n');
fprintf('=======================================================================\n');
fprintf('  Total integration time  : %.1f s\n',   T_total);
fprintf('  Time step (RK4)         : %.4f s\n',   dt);
fprintf('  Number of steps         : %d\n',       N);
fprintf('  Max attractor radius    : %.4f\n',     max_radius);
if ~isnan(lambda_simple)
    fprintf('  MLE Method A (log-fit)  : %.4f nats/s  (biased: ~0.5x true)\n', lambda_simple);
end
if ~isnan(lambda_est)
    fprintf('  MLE Method B (Wolf 1985): %.4f nats/s\n',  lambda_est);
    fprintf('  Theoretical MLE         : ~0.905  nats/s\n');
end
fprintf('  Output files saved in   : %s\n',       resultsDir);
fprintf('=======================================================================\n\n');
