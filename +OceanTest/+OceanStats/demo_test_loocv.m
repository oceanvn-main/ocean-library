%% Demo & Test: Automatic Bandwidth Selection using Exact LOOCV
% This script tests the 'calcLoocv_general' function from OceanStatsLib.

% 1. CLEANUP & IMPORTS
import OceanStats.NonParametric.* 

% --- 0. Visualization Setup (Dark Mode) ---
fig_bg  = [0.15, 0.15, 0.15];
ax_bg   = [0.2, 0.2, 0.2];
text_c  = [0.9, 0.9, 0.9];
cyan_c  = [0, 1, 1];
mag_c   = [1, 0, 1];
yell_c  = [1, 0.8, 0.2];

%% 1. Generate Synthetic Data
% -------------------------------------------------------------------------
rng(42); % Reproducible
N = 150;
X = sort(rand(N, 1) * 4 * pi); % 0 to 4pi
% True function: Sine wave with varying frequency
Y_true = sin(X) + 0.5 * sin(2.5 * X);
% Add noise
sigma = 0.4;
Y = Y_true + sigma * randn(N, 1);

%% 2. Unit Test: Shortcut vs. Brute Force
% -------------------------------------------------------------------------
h_test = 0.5;
disp('--- RUNNING UNIT TEST ---');

% A. Run Your Library Function (Fast)
tic;
% Since we imported the package at the top, we can now use the short name:
[cv_fast, df_fast, diags] = calcLoocv(X, Y, h_test, 'gaussian', 1);
time_fast = toc;

% B. Run Brute Force Loop (Slow)
tic;
error_sq = 0;
for i = 1:N
    % Leave one out
    X_tr = X([1:i-1, i+1:end]);
    Y_tr = Y([1:i-1, i+1:end]);
    
    % Inline Local Linear fit
    x0 = X(i);
    w = exp(-((X_tr - x0).^2) / (2 * h_test^2));
    Z = [ones(N-1,1), (X_tr - x0)];
    W = diag(w);
    beta = (Z'*W*Z) \ (Z'*W*Y_tr);
    y_pred = beta(1);
    
    error_sq = error_sq + (Y(i) - y_pred)^2;
end
cv_brute = error_sq;
time_slow = toc;

% C. Compare
fprintf('Bandwidth: %.2f\n', h_test);
fprintf('Shortcut CV:   %.6f (Time: %.4fs)\n', cv_fast, time_fast);
fprintf('BruteForce CV: %.6f (Time: %.4fs)\n', cv_brute, time_slow);
assert(abs(cv_fast - cv_brute) < 1e-4, 'Unit Test Failed: Scores do not match!');
fprintf('>> SUCCESS: The shortcut formula is mathematically correct.\n\n');

%% 3. Application: Model Selection (Hyperparameter Tuning)
% -------------------------------------------------------------------------
disp('--- RUNNING MODEL SELECTION ---');

bandwidth_grid = linspace(0.1, 2.0, 30);
cv_curve = zeros(size(bandwidth_grid));
dof_curve = zeros(size(bandwidth_grid));

for k = 1:length(bandwidth_grid)
    h = bandwidth_grid(k);
    [score, dof] = calcLoocv(X, Y, h, 'gaussian', 1);
    
    cv_curve(k) = score;
    dof_curve(k) = dof;
end

% Find the winner
[min_cv, idx_best] = min(cv_curve);
h_best = bandwidth_grid(idx_best);
dof_best = dof_curve(idx_best);

fprintf('Optimal Bandwidth: %.3f\n', h_best);
fprintf('Minimum LOOCV Score: %.3f\n', min_cv);
fprintf('Effective DoF: %.2f\n', dof_best);

%% 4. Visualization
figure('Color', fig_bg, 'Position',[10, 100, 1200, 500]);

%Plo A: The Cross-Validation Curve
subplot(121);
set(gca, 'Color', ax_bg, 'XColor', text_c, 'YColor', text_c);
hold on;
plot(bandwidth_grid, cv_curve, '-o', 'Color', 'w', 'MarkerFaceColor', cyan_c, 'LineWidth', 1.5);
xline(h_best, '--', 'Color', yell_c, 'LineWidth', 2);
text(h_best, max(cv_curve)*0.9, 'Best H', 'Color', yell_c, 'FontSize', 12);

title('Model Selection: CV Error vs Bandwidth', 'Color', text_c);
xlabel('Bandwidth (h)', 'Color', text_c);
ylabel('LOOCV Error (RSS)', 'Color', text_c);
grid on; ax=gca; ax.GridColor=[1 1 1]; ax.GridAlpha=0.1;

%Plot B: The best fit
subplot(122);
set(gca, 'Color', ax_bg, 'XColor', text_c, 'YColor', text_c);
hold on;
plot(X, Y, 'o', 'Color', [.5 .5 .5], 'MarkerSize', 4, 'DisplayName', 'Noisy Data')
plot(X, Y_true, 'Color', mag_c, 'LineWidth', 1, 'DisplayName', 'True Function');

% Reconstruct best fit for plotting
Y_best_fit = zeros(N, 1);
for i=1:N
    w = exp(-((X - X(i)).^2) / (2 * h_best^2));
    Z = [ones(N,1), (X - X(i))];
    beta = (Z'*diag(w)*Z + eye(2)*1e-6) \ (Z'*diag(w)*Y);
    Y_best_fit(i) = beta(1);
end

plot(X, Y_best_fit, 'Color', cyan_c, 'LineWidth', 2.5, 'DisplayName', sprintf('Best Fit (h=%.2f)', h_best));

title(sprintf('Optimal Model (Effective DoF = %.1f)', dof_best), 'Color', text_c);
legend('TextColor', text_c, 'Color', 'none', 'EdgeColor', 'none');
grid on; ax=gca; ax.GridColor=[1 1 1]; ax.GridAlpha=0.1;

sgtitle('OceanStatsLib: Automatic Smoothing Parameter Selection', 'Color', text_c, 'FontSize', 14);