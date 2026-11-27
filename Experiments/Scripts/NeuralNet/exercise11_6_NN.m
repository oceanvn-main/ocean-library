% Exercise 11.6: Visualizing Projection Pursuit Regression Traces
clear; clc; close all;

% --- 1. Objective Data Generation ---
rng(42); 
N = 300; % Slightly more data for a smoother surface fit
X = randn(N, 2);
Z = randn(N, 1);

% True parameters used ONLY to generate the target data
a1 = [3; 3];
a2 = [3; -3];
Y_true = 1./(1 + exp(-X*a1)) + (X*a2).^2; % Pure signal
Y_train = Y_true + 0.30 * Z;              % Noisy target for training

% --- 2. Train Forward-Stagewise PPR (M = 2) ---
% We let the data find the optimal directions and spline shapes.
M = 2; 
p = 0.8; % Smoothing parameter for cubic splines. p=1 is max smoothing. p=0 is min smoothing.
R_train = Y_train;

% Storage for the learned model parameters
W_learned = zeros(2, M);
Splines = cell(M, 1);

options = optimset('Display', 'off');

%f(X) = sum_m=1^M gm (w_m X)
%Notice gm is not known. We are searching for gm
for m = 1:M
    % Objective function to find the best 1D projection angle (theta)
    obj_fun = @(theta) evaluate_projection(X, R_train, theta, p);
    
    % Search for optimal angle between 0 and pi
    theta_opt = fminbnd(obj_fun, 0, pi, options); %Scalar
    w_opt = [cos(theta_opt); sin(theta_opt)]; % Unit vector
    W_learned(:, m) = w_opt;
    
    % Project data onto the optimal direction and fit the spline
    V = X * w_opt;
    [V_sort, sort_idx] = sort(V);
    R_sort = R_train(sort_idx);
    
    % csaps creates the cubic smoothing spline
    sp = csaps(V_sort, R_sort, p);
    Splines{m} = sp;
    
    % Update residuals for the next term
    pred_m = fnval(sp, V')';
    R_train = R_train - pred_m;
end

% --- 3. Generate Dense Grid for Surface Evaluation ---
x1_range = linspace(-2.5, 2.5, 60);
x2_range = linspace(-2.5, 2.5, 60);
[X1_grid, X2_grid] = meshgrid(x1_range, x2_range);
X_grid = [X1_grid(:), X2_grid(:)];

% --- 4. Evaluate Models on the Grid ---
% A. True Signal Surface
Y_grid_true = 1./(1 + exp(-X_grid*a1)) + (X_grid*a2).^2;
Z_true = reshape(Y_grid_true, size(X1_grid));

% B. PPR Term 1 Surface Trace
V1_grid = X_grid * W_learned(:, 1);
Y_grid_term1 = fnval(Splines{1}, V1_grid')';
Z_term1 = reshape(Y_grid_term1, size(X1_grid));

% C. PPR Term 2 Surface Trace
V2_grid = X_grid * W_learned(:, 2);
Y_grid_term2 = fnval(Splines{2}, V2_grid')';
Z_term2 = reshape(Y_grid_term2, size(X1_grid));

% D. Full PPR Approximation Surface (Term 1 + Term 2)
Z_ppr_full = Z_term1 + Z_term2;

% --- 5. Plotting in Dark Mode ---
fig = figure('Name', 'PPR Function Approximation Traces', 'Position', [100, 100, 1000, 800]);
set(fig, 'Color', [0.15 0.15 0.15]);

% Top-Left: True Signal
ax1 = subplot(2, 2, 1);
surf(X1_grid, X2_grid, Z_true, 'EdgeColor', 'none');
title('True Function Surface (No Noise)', 'Color', 'w');
configureDarkModeAxes(ax1);

% Top-Right: Full PPR Approximation
ax2 = subplot(2, 2, 2);
surf(X1_grid, X2_grid, Z_ppr_full, 'EdgeColor', 'none');
title('PPR Final Approximation (M=2)', 'Color', 'w');
configureDarkModeAxes(ax2);

% Bottom-Left: PPR Term 1 Trace
ax3 = subplot(2, 2, 3);
surf(X1_grid, X2_grid, Z_term1, 'EdgeColor', 'none');
title(sprintf('PPR Term 1 Trace (Angle: %.2f rad)', atan2(W_learned(2,1), W_learned(1,1))), 'Color', 'w');
configureDarkModeAxes(ax3);

% Bottom-Right: PPR Term 2 Trace
ax4 = subplot(2, 2, 4);
surf(X1_grid, X2_grid, Z_term2, 'EdgeColor', 'none');
title(sprintf('PPR Term 2 Trace (Angle: %.2f rad)', atan2(W_learned(2,2), W_learned(1,2))), 'Color', 'w');
configureDarkModeAxes(ax4);

colormap(parula);

% --- Helper Functions ---
function mse = evaluate_projection(X, R, theta, p)
    w = [cos(theta); sin(theta)];
    V = X * w;
    [V_sort, sort_idx] = sort(V);
    R_sort = R(sort_idx);
    sp = csaps(V_sort, R_sort, p);
    pred = fnval(sp, V')';
    mse = mean((R - pred).^2);
end

function configureDarkModeAxes(ax)
    view(ax, -35, 45);
    xlabel(ax, 'X_1', 'Color', 'w');
    ylabel(ax, 'X_2', 'Color', 'w');
    zlabel(ax, 'Y', 'Color', 'w');
    set(ax, 'Color', [0.1 0.1 0.1], ...
            'XColor', [0.8 0.8 0.8], ...
            'YColor', [0.8 0.8 0.8], ...
            'ZColor', [0.8 0.8 0.8], ...
            'GridColor', [0.3 0.3 0.3], ...
            'GridAlpha', 0.5);
    grid on;
end