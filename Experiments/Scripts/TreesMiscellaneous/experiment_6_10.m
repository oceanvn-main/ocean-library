%% Experiment 6.10: Unbiased Risk Estimation (Mallows' Cp / SURE)
% This script verifies that Training Error is biased and testing the
% correction formula C_lambda = ASR + 2*sigma^2/N * trace(S).
%
% Theoretical Expectation:
%   1. E[ASR] (Training Error) will be LOWER than True Error.
%   2. E[C_lambda] (Corrected) should match E[PE] (True Error).

clear; clc; close all;

% --- 1. Simulation Setup ---
rng(123);             % Reproducibility
N = 50;               % Number of data points
n_sims = 500;         % Number of Monte Carlo simulations (Universes)
sigma = 0.5;          % Noise standard deviation (Known sigma)
variance = sigma^2;

% Fixed X locations (The "In-Sample" Assumption)
X = linspace(0, 2*pi, N)';

% True Function f(x)
f_true = sin(X) + sin(2*X);

% Define Bandwidth Grid (Hyperparameter Lambda)
h_grid = linspace(0.05, 1.5, 30);
n_h = length(h_grid);

% Storage for results (Average over simulations)
avg_ASR = zeros(n_h, 1);      % Average Squared Residuals (Training Error)
avg_PE = zeros(n_h, 1);       % True Prediction Error (New Data)
avg_Clambda = zeros(n_h, 1);  % The Formula Estimate
avg_DoF = zeros(n_h, 1);      % Effective Degrees of Freedom

fprintf('Running %d Monte Carlo simulations...\n', n_sims);

% --- 2. Pre-calculate Smoothing Matrices (S) ---
% Since X is fixed, the matrix S depends ONLY on bandwidth h, not on Y.
% We can pre-calculate degrees of freedom (trace(S)) for speed.
S_matrices = cell(n_h, 1);
dofs = zeros(n_h, 1);

for k = 1:n_h
    h = h_grid(k);
    
    % Construct Kernel Matrix (Gaussian)
    % K(x, xi)
    dists_sq = (X - X').^2;
    K = exp(-dists_sq / (2 * h^2));
    
    % Normalize rows to sum to 1 (Nadaraya-Watson)
    % S_ij = K_ij / sum(K_i)
    row_sums = sum(K, 2);
    S = K ./ row_sums;
    
    S_matrices{k} = S;
    dofs(k) = trace(S); % The Penalty Term
end
avg_DoF = dofs;

% --- 3. Monte Carlo Loop ---
% We simulate the world n_sims times to approximate Expectation E[...]
for sim = 1:n_sims
    
    % A. Generate Data
    epsilon = randn(N, 1) * sigma;      % Training Noise
    y_train = f_true + epsilon;         % Training Labels
    
    epsilon_new = randn(N, 1) * sigma;  % Future Noise (Independent)
    y_test = f_true + epsilon_new;      % True Future Data (Same X)
    
    % B. Loop over Bandwidths
    for k = 1:n_h
        S = S_matrices{k};
        
        % 1. Fit Model (Linear Smoother: y_hat = S * y)
        y_hat = S * y_train;
        
        % 2. Calculate Training Error (ASR)
        % Mean Squared Error on the data we saw
        asr = mean((y_train - y_hat).^2);
        
        % 3. Calculate True Prediction Error (PE)
        % Mean Squared Error on the data we DIDN'T see
        pe = mean((y_test - y_hat).^2);
        
        % 4. Calculate The Formula (C_lambda)
        % C_lambda = ASR + (2 * sigma^2 / N) * trace(S)
        penalty = (2 * variance / N) * dofs(k);
        c_stat = asr + penalty;
        
        % Accumulate sums (for averaging later)
        avg_ASR(k) = avg_ASR(k) + asr;
        avg_PE(k) = avg_PE(k) + pe;
        avg_Clambda(k) = avg_Clambda(k) + c_stat;
    end
end

% Compute Expectations (Divide by N_sims)
E_ASR = avg_ASR / n_sims;
E_PE = avg_PE / n_sims;
E_Clambda = avg_Clambda / n_sims;

fprintf('Simulation Complete.\n');

% --- 4. Visualization (Dark Mode) ---
fig_bg  = [0.15, 0.15, 0.15];
ax_bg   = [0.2, 0.2, 0.2];
text_c  = [0.9, 0.9, 0.9];

figure('Color', fig_bg, 'Position', [100, 100, 1000, 600]);

% Plot 1: The Curves
axes('Color', ax_bg, 'XColor', text_c, 'YColor', text_c, 'NextPlot', 'add');
grid on;

% A. The Deceptive Training Error
plot(h_grid, E_ASR, 'Color', [0, 1, 1], 'LineWidth', 2, 'DisplayName', 'E[Training Error] (Biased)');

% B. The True Error
plot(h_grid, E_PE, 'Color', [1, 0.2, 0.6], 'LineWidth', 2, 'DisplayName', 'E[True Error] (Target)');

% C. The Formula Estimate
plot(h_grid, E_Clambda, '--', 'Color', [1, 1, 0], 'LineWidth', 2, 'DisplayName', 'E[Formula C_\lambda] (Corrected)');

% D. Mark Optimal Bandwidth
[min_err, idx_opt] = min(E_PE);
xline(h_grid(idx_opt), '--', 'Color', [0.5 0.5 0.5], 'DisplayName', 'Optimal Bandwidth');

title('Exercise 6.10 Verification: Bias Correction', 'Color', text_c, 'FontSize', 14);
xlabel('Bandwidth (Model Flexibility \rightarrow)', 'Color', text_c);
ylabel('Mean Squared Error', 'Color', text_c);
legend('TextColor', text_c, 'Color', ax_bg, 'Location', 'best');

%Add Explanation Annotation
str = { ...
    'Notice:', ...
    '1. Cyan Line (Training) is always lower than Pink (True).', ...
    '   This is the "Optimism Gap".', ...
    '2. Yellow Dashed (Formula) matches Pink perfectly.', ...
    '   The formula successfully corrected the bias!' ...
};

annotation('textbox', [.6, .4, .25, .2], 'String', str, 'Color', text_c, 'EdgeColor', 'w', 'BackgroundColor', ax_bg);

% Plot 2: Degrees of Freedom (Inset)
axes('Position', [0.25, 0.6, 0.2, 0.2], 'Color', ax_bg, 'XColor', text_c, 'YColor', text_c);
plot(h_grid, avg_DoF, 'g-', 'LineWidth', 1.5);
title('Model Complexity (DoF)', 'Color', text_c, 'FontSize', 10);
xlabel('Bandwidth h', 'Color', text_c);
ylabel('Trace(S)', 'Color', text_c);
grid on;
