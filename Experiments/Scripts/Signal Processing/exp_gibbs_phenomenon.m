% exp_gibbs_phenomenon.m
% Ephemeral experiment script for Gibbs Phenomenon and Ideal Lowpass Filter analysis.
%
% This script demonstrates:
% 1. Pointwise convergence of truncated Fourier series approximations.
% 2. Gibbs phenomenon overshoot and peak frequency sweep.
% 3. L2 error energy verification between frequency and time domains (Parseval's Theorem).
% 4. Dark theme visualizations leveraging OceanViz.Core.
%
% Coordinated by Subagent B.

% 1. Initialize RNG and Constants
rng(42, 'twister');

% 2. Instantiate Filter Object
w_c = pi/3;
filterObj = OceanMath.Signal.IdealLowpass(w_c);

% 3. Pointwise Convergence Simulation
w_grid = linspace(-pi, pi, 1000);
M_values = [1, 3, 7, 19, 100];
H_approx_cells = cell(length(M_values), 1);

for i = 1:length(M_values)
    H_approx_cells{i} = filterObj.frequencyResponseApproximation(w_grid, M_values(i));
end

% 4. Gibbs Overshoot Sweep (M = 1 to 200)
M_sweep_overshoot = 1:200;
pctOvershoot_sweep = zeros(1, length(M_sweep_overshoot));
peakFreq_sweep = zeros(1, length(M_sweep_overshoot));

for idx = 1:length(M_sweep_overshoot)
    M = M_sweep_overshoot(idx);
    [~, peakFreq_sweep(idx), pctOvershoot_sweep(idx)] = filterObj.calculateOvershoot(M);
end

% 5. Parseval L2 Error Verification (M = 1 to 100)
M_sweep_err = 1:100;
errFreq_sweep = zeros(1, length(M_sweep_err));
errTime_sweep = zeros(1, length(M_sweep_err));

for idx = 1:length(M_sweep_err)
    M = M_sweep_err(idx);
    [errFreq_sweep(idx), errTime_sweep(idx)] = filterObj.verifyParseval(M);
    
    % Verify discrepancy is small
    diffError = abs(errFreq_sweep(idx) - errTime_sweep(idx));
    if diffError >= 1e-12
        error('Parseval:VerificationFailed', ...
            'Parseval discrepancy exceeded tolerance 1e-12 at M = %d: diff = %e', M, diffError);
    end
end

% Print a structured Parseval L2 verification summary
fprintf('=========================================================\n');
fprintf('         PARSEVAL L2 ERROR VERIFICATION SUMMARY          \n');
fprintf('=========================================================\n');
fprintf('%-5s | %-20s | %-20s | %-12s\n', 'M', 'Frequency Error (L2)', 'Time Error (L2)', 'Difference');
fprintf('---------------------------------------------------------\n');
selected_M = [1, 2, 5, 10, 20, 50, 80, 100];
for M = selected_M
    idx = find(M_sweep_err == M);
    if ~isempty(idx)
        diffVal = abs(errFreq_sweep(idx) - errTime_sweep(idx));
        fprintf('%5d | %20.15e | %20.15e | %12.3e\n', ...
            M, errFreq_sweep(idx), errTime_sweep(idx), diffVal);
    end
end
fprintf('---------------------------------------------------------\n');
fprintf('Max Absolute Parseval Discrepancy: %e\n', max(abs(errFreq_sweep - errTime_sweep)));
fprintf('Verification Result: PASSED (all differences < 1e-12)\n');
fprintf('=========================================================\n\n');

% 6. Visualizations using OceanViz.Core
viz = OceanViz.Core('Theme', 'Dark');

% Define ideal step profile for baseline overlay
w_ideal = [-pi, -w_c, -w_c, w_c, w_c, pi];
H_ideal = [0, 0, 1, 1, 0, 0];

% Figure 1: Pointwise Fourier Series Convergence
fig1 = viz.createFigure('Pointwise Fourier Series Convergence');

% Get color sequence for plotting the 5 approximations
colors = viz.getColorSequence(length(M_values));

% 3x2 grid layout (M = 1, 3, 7, 19, plus a large subplot for M = 100)
subplot_indices = {1, 2, 3, 4, [5, 6]};

for i = 1:length(M_values)
    M = M_values(i);
    ax = subplot(3, 2, subplot_indices{i});
    
    % Overlay ideal step profile
    plot(ax, w_ideal, H_ideal, '--', 'Color', viz.Colors.Boundary, 'LineWidth', 1.8);
    hold(ax, 'on');
    
    % Plot truncated approximation
    plot(ax, w_grid, H_approx_cells{i}, 'Color', colors(i, :), 'LineWidth', 1.8);
    
    % Titles & Labels
    title(ax, sprintf('Fourier Series Approximation (M = %d)', M));
    xlabel(ax, 'Frequency \omega (rad/sample)');
    ylabel(ax, 'H_M(e^{j\omega})');
    
    % Style axes and add legend
    viz.styleAxes(ax);
    
    lgd = legend(ax, {'Ideal Filter', sprintf('M = %d Approx', M)}, 'Location', 'best');
    set(lgd, 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background, 'EdgeColor', viz.Colors.Grid);
end

% Figure 2: Mathematical Characterization
fig2 = viz.createFigure('Mathematical Characterization');

% Left panel: Overshoot Percentage vs M (M = 1 to 200)
ax_left = subplot(1, 2, 1);
plot(ax_left, M_sweep_overshoot, pctOvershoot_sweep, 'Color', viz.Colors.Primary, 'LineWidth', 2);
hold(ax_left, 'on');

% Overlay theoretical asymptotic limit (8.95%) using viz.Colors.Boundary
plot(ax_left, [1, 200], [8.948967, 8.948967], '--', 'Color', viz.Colors.Boundary, 'LineWidth', 1.8);

title(ax_left, 'Gibbs Overshoot Percentage vs. M');
xlabel(ax_left, 'Truncation Limit M');
ylabel(ax_left, 'Overshoot (%)');

viz.styleAxes(ax_left);
lgd1 = legend(ax_left, {'Empirical Overshoot', 'Asymptotic Limit (8.95%)'}, 'Location', 'best');
set(lgd1, 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background, 'EdgeColor', viz.Colors.Grid);

% Right panel: L2 Error Energy vs M (semilogy scale)
ax_right = subplot(1, 2, 2);
semilogy(ax_right, M_sweep_err, errFreq_sweep, 'Color', viz.Colors.Primary, 'LineWidth', 2);
hold(ax_right, 'on');
semilogy(ax_right, M_sweep_err, errTime_sweep, '--', 'Color', viz.Colors.Secondary, 'LineWidth', 2);

title(ax_right, 'L2 Error Energy vs. M');
xlabel(ax_right, 'Truncation Limit M');
ylabel(ax_right, 'L2 Error Energy');

viz.styleAxes(ax_right);
lgd2 = legend(ax_right, {'Frequency Domain Error (errFreq)', 'Time Domain Error (errTime)'}, 'Location', 'best');
set(lgd2, 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background, 'EdgeColor', viz.Colors.Grid);

fprintf('Gibbs Phenomenon Experiment finished successfully.\n');
