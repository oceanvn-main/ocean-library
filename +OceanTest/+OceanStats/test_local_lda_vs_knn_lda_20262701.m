%% Master Comparison Script: Global LDA vs. KNN vs. Local LDA
% This script performs two major jobs:
% JOB 1: Compare Confusion Matrices on the full dataset (High Dimensional).
% JOB 2: Visualize Decision Boundaries on 2D PCA-projected data.
% --- 0. Setup & Data Loading ---
libRoot = 'C:\Ocean Library\Matlab code practice';
addpath(libRoot);
import OceanStats.Supervised.Discriminant.LocalLDA;

% Smart Data Loader (Handles nested folders)
path_std = fullfile(libRoot, '+OceanStats', 'data', 'zip.train');
path_nested = fullfile(libRoot, '+OceanStats', 'data', 'zip.train', 'zip.train');

if exist(path_std, 'file'), fpath = path_std;
elseif exist(path_nested, 'file'), fpath = path_nested;
else, error('Data not found. Check path.'); end

fprintf('Loading Zipcode Data...\n');
data = load(fpath);

%% --- JOB 1: Confusion Matrix Comparison (Full 256D Data) ---
fprintf('\n--- Starting JOB 1: Confusion Matrix Comparison ---\n');

% 1.1 Prepare Data
n_train = 2000;
n_test  = 500;
X_train = data(1:n_train, 2:end);
y_train = data(1:n_train, 1);
X_test  = data(n_train+1 : n_train+n_test, 2:end);
y_test  = data(n_train+1 : n_train+n_test, 1);

% 1.2 Train & Predict

% Model A: Global LDA (Toolbox - Fast)
fprintf('Training Global LDA...\n');
t = tic;
mdl_lda = fitcdiscr(X_train, y_train, 'DiscrimType', 'linear');
y_pred_lda = predict(mdl_lda, X_test);
fprintf('  -> Done (%.2fs)\n', toc(t));

% Model B: KNN (Toolbox - Fast)
k = 3;
fprintf('Training KNN (k=%d)...\n', k);
t = tic;
mdl_knn = fitcknn(X_train, y_train, 'NumNeighbors', k, 'Standardize', 1);
y_pred_knn = predict(mdl_knn, X_test);
fprintf('  -> Done (%.2fs)\n', toc(t));

% Model C: Local LDA (Custom Class)
lambda = 5.0;
fprintf('Training Local LDA (Custom, lambda=%.1f)...\n', lambda);
t = tic;
mdl_local = LocalLDA('Bandwidth', lambda);
mdl_local.fit(X_train, y_train);
y_pred_local = mdl_local.predict(X_test);
fprintf('  -> Done (%.2fs)\n', toc(t));

% 1.3 Plot Confusion Matrices (Figure 1)
fig1 = figure('Name', 'Job 1: Confusion Matrices', 'Color', [0.15 0.15 0.15], 'Position', [50, 500, 1400, 400]);
t = tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
title(t, 'Job 1: Confusion Matrix Comparison', 'Color', 'w', 'FontSize', 16);

nexttile; plot_custom_cm(y_test, y_pred_lda, 'Global LDA (Linear)');
nexttile; plot_custom_cm(y_test, y_pred_knn, sprintf('KNN (k=%d)', k));
nexttile; plot_custom_cm(y_test, y_pred_local, sprintf('Local LDA (\\lambda=%.1f)', lambda));

%% --- JOB 2: Decision Boundary Visualization (PCA 2D Data) ---
fprintf('\n--- Starting JOB 2: Decision Boundary Visualization ---\n');

% 2.1 Filter Data (Use only 3 classes for clarity: 0, 1, 8)
target_classes = [0, 1, 8];
idx = ismember(data(:, 1), target_classes);
X_raw = data(idx, 2:end);
y_raw = data(idx, 1);

%Project to 2D using PCA
fprintf('Projecting data to 2D (PCA) ...\n');
[~, score] = pca(X_raw);
X_2d = score(:, 1:2);

%Create Mesh grid
resolution = 0.2;
x_min = min(X_2d(:,1))-1; x_max = max(X_2d(:,1))+1;
y_min = min(X_2d(:,2))-1; y_max = max(X_2d(:,2))+1;
[xx, yy] = meshgrid(x_min:resolution:x_max, y_min:resolution:y_max);
X_grid = [xx(:), yy(:)];

%Train and Predict on 2D data
%Model A: Global LDA
fprintf('Calculating Global LDA Boundary...\n');
mdl_lda_2d = fitcdiscr(X_2d, y_raw);
Z_lda = predict(mdl_lda_2d, X_grid);
Z_lda = reshape(Z_lda, size(xx));

%Model B: KNN
fprintf('Calculating KNN Boundary...\n');
mdl_knn_2d = fitcknn(X_2d, y_raw, 'NumNeighbors', 5);
Z_knn = predict(mdl_knn_2d, X_grid);
Z_knn = reshape(Z_knn, size(xx));

%Model C: Local LDA
fprintf('Calculating Local LDA Boundary...\n');
mdl_local_2d = LocalLDA('Bandwidth', 2.0); % Adjusted lambda for PCA scale
mdl_local_2d.fit(X_2d, y_raw);
Z_local = mdl_local_2d.predict(X_grid);
Z_local = reshape(Z_local, size(xx));

%Plot boundaries
fig2 = figure('Name', 'Job 2: Decision Boundaries', 'Color', [0.15 0.15 0.15], 'Position', [50, 50, 1400, 400]);
t2 = tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
title(t2, 'Job 2: Decision Boundary Comparison (PCA Space)', 'Color', 'w', 'FontSize', 16);

ax1 = nexttile; plot_boundary(ax1, xx, yy, Z_lda, X_2d, y_raw, 'Global LDA (Linear)');
ax2 = nexttile; plot_boundary(ax2, xx, yy, Z_knn, X_2d, y_raw, 'KNN (k=5, Non-Linear)');
ax3 = nexttile; plot_boundary(ax3, xx, yy, Z_local, X_2d, y_raw, 'Local LDA (Adaptive)');

linkaxes([ax1, ax2, ax3], 'xy');
fprintf('\nAll jobs completed successfully.\n');

%% Helper Functions
function plot_custom_cm(y_true, y_pred, title_str)
    % Helper to plot a dark-mode confusion matrix
    cm = confusionchart(y_true, y_pred);
    cm.Title = title_str;
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';
    cm.FontColor = [0.9 0.9 0.9]; % White text
    % Note: Do not set Parent.BackgroundColor here to avoid Layout errors
end


function plot_boundary(ax, xx, yy, Z, X, y, title_str)
    % Helper to plot decision boundaries in dark mode
    set(ax, 'Color', [0.2 0.2 0.2], 'XColor', 'w', 'YColor', 'w');
    hold(ax, 'on'); grid(ax, 'on');
    
    % Draw Regions
    [~, h] = contourf(ax, xx, yy, Z);
    set(h, 'LineColor', 'none', 'FaceAlpha', 0.5);
    
    % Draw Points
    gscatter(ax, X(:,1), X(:,2), y, 'cym', 'o+^', 5, 'off');
    
    title(ax, title_str, 'Color', 'w');
    if contains(title_str, 'KNN') % Legend only on middle plot
        legend(ax, 'Location', 'southoutside', 'Orientation', 'horizontal', 'TextColor', 'w');
    end
end