%% Test Script for Local LDA (Robust Version)
% --- 1. Setup Library Path ---
libRoot = 'C:\Ocean Library\Matlab code practice';
addpath(libRoot);

% Import the class
import OceanStats.Supervised.Discriminant.LocalLDA;

% --- 2. Locate Data Files (Smart Check) ---
% Define possible paths for zip.train
path_standard = fullfile(libRoot, '+OceanStats', 'data', 'zip.train');

% Do the same for zip.test
path_test_std = fullfile(libRoot, '+OceanStats', 'data', 'zip.test');

% --- 3. Load Data ---
data_train = load(path_standard);
data_test  = load(path_test_std);

% ESL Format: Col 1 = Label, Cols 2:end = Features
y_train = data_train(:, 1);
X_train = data_train(:, 2:end);
y_test  = data_test(:, 1);
X_test  = data_test(:, 2:end);

% --- 4. Optimization (Speed) ---
% Use a subset for faster testing (set to Inf to use all)
n_test_subset = 200; 

if n_test_subset < length(y_test)
    fprintf('Testing on subset of %d points...\n', n_test_subset);
    X_test_sub = X_test(1:n_test_subset, :);
    y_test_sub = y_test(1:n_test_subset);
else
    X_test_sub = X_test;
    y_test_sub = y_test;
end

% --- 5. Run Experiment (Ex 6.12) ---
lambdas = [1.0, 3.0, 5.0, 8.0, 15.0];
errors  = zeros(length(lambdas), 1);

fprintf('\nStarting Experiment...\n');
fprintf('--------------------------------------\n');
fprintf('|  Lambda  |  Test Error  | Accuracy |\n');
fprintf('--------------------------------------\n');

for k = 1:length(lambdas)
    lam = lambdas(k);
    
    % Initialize & Fit
    model = LocalLDA('Bandwidth', lam);
    model.fit(X_train, y_train);
    
    % Predict
    y_pred = model.predict(X_test_sub);
    
    % Calculate Error
    acc = mean(y_pred == y_test_sub);
    err = 1 - acc;
    errors(k) = err;
    
    fprintf('|   %4.1f   |    %.4f    |  %.2f%%  |\n', lam, err, acc * 100);
end
fprintf('--------------------------------------\n');

% --- 6. Dark Mode Plotting ---
figure('Color', [0.15, 0.15, 0.15]); % Dark background

% Setup Axes
ax = axes('Color', [0.2, 0.2, 0.2], ...
          'XColor', [0.9, 0.9, 0.9], ...
          'YColor', [0.9, 0.9, 0.9], ...
          'GridColor', [0.3, 0.3, 0.3]);
hold on; grid on;

% Plot Curve
plot(lambdas, errors, '-o', 'Color', [0, 1, 1], 'LineWidth', 2, ...
     'MarkerFaceColor', [0, 1, 1], 'MarkerEdgeColor', 'w');

% Mark Best
[min_err, idx] = min(errors);
plot(lambdas(idx), min_err, 'h', 'MarkerSize', 15, ...
     'MarkerFaceColor', [1, 0.8, 0], 'MarkerEdgeColor', 'w');

title('Local LDA Error vs Bandwidth', 'Color', 'w');
xlabel('Bandwidth', 'Color', 'w');
ylabel('Error Rate', 'Color', 'w');
text(lambdas(idx), min_err + 0.01, sprintf(' Best: %.1f', lambdas(idx)), ...
     'Color', [1, 0.8, 0], 'FontWeight', 'bold');

hold off;