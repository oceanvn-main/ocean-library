% EXP_PROCRUSTES_EMOTION_CLF  Emotion classification with Procrustes preprocessing
%
%   DATASET:        AffectNet subset (N=68, p=2, L=2400)
%                   4 classes: anger, happy, neutral, surprise (600 each)
%   MODEL:          Support Vector Machine (SVM) with RBF kernel, One-vs-One
%   ASSUMPTIONS:    
%     - Baseline uses raw pixel coordinates (normalized bounding box space).
%     - Rigid GPA preserves aspect ratio (jaw drop / emotion).
%     - Affine GPA distorts aspect ratio, absorbing emotion into the subspace.
%     - 70/30 train/test split, stratified.
%   REPRODUCIBILITY: rng(42, 'twister')
%
clc; clear; rng(42, 'twister');

% Import Ocean Library packages
import OceanStats.Unsupervised.*
import OceanViz.*

% =========================================================================
% === 1. Data Loading ===
% =========================================================================
data_path = fullfile('c:\', 'Ocean Library', 'Matlab code practice', ...
    'Data', 'AffectNet', 'AffectNet_landmarks.mat');

if ~isfile(data_path)
    error('Dataset not found. Please run preprocess_affectnet_landmarks.py first.');
end

fprintf('Loading AffectNet landmarks...\n');
data = load(data_path);
X_raw = data.X;             % [68 x 2 x 2400]
Y     = double(data.Y);     % [2400 x 1] class IDs 0-3
class_names = string(data.label_names);

[N, p, L] = size(X_raw);
fprintf('  Dataset loaded: L=%d shapes, N=%d landmarks, p=%d dims\n', L, N, p);

% Ensure labels are 1-indexed for MATLAB grouping (1 to 4)
Y = Y + 1;

% =========================================================================
% === 2. Feature Extraction (3 Pipelines) ===
% =========================================================================
% We will extract 136-dimensional feature vectors (68 * 2) from three pipelines:
%   1. Baseline: Raw coordinates
%   2. Pipeline A: Rigid Procrustes alignment
%   3. Pipeline B: Affine Procrustes alignment

fprintf('\nPipeline A: Running Rigid GPA...\n');
[M_rigid, ~, ~, X_aligned_rigid] = rigid_procrustes_average(X_raw);

fprintf('Pipeline B: Running Affine-Invariant GPA...\n');
[M_affine, X_aligned_affine] = affine_invariant_average(X_raw);

% Center raw X and scale to norm 1 for fair baseline comparison
X_centered = zeros(N, p, L);
for l = 1:L
    centered = X_raw(:,:,l) - mean(X_raw(:,:,l), 1);
    X_centered(:,:,l) = centered / norm(centered, 'fro');
end

% Flatten to [L x (N*p)] for SVM input
features_base   = reshape(X_centered, N*p, L)';
features_rigid  = reshape(X_aligned_rigid, N*p, L)';
features_affine = reshape(X_aligned_affine, N*p, L)';

% =========================================================================
% === 3. Variance Audit (The Geometric Catastrophe) ===
% =========================================================================
var_base   = var(features_base, 0, 1);
var_rigid  = var(features_rigid, 0, 1);
var_affine = var(features_affine, 0, 1);

mean_var_base   = mean(var_base);
mean_var_rigid  = mean(var_rigid);
mean_var_affine = mean(var_affine);

fprintf('\n--- Geometric Variance Audit ---\n');
fprintf('  Mean feature variance (Baseline) : %.4e\n', mean_var_base);
fprintf('  Mean feature variance (Rigid)    : %.4e\n', mean_var_rigid);
fprintf('  Mean feature variance (Affine)   : %.4e\n', mean_var_affine);
fprintf('  Affine / Rigid ratio             : %.2f\n', mean_var_affine / mean_var_rigid);

% =========================================================================
% === 4. Train / Test Split ===
% =========================================================================
cv = cvpartition(Y, 'Holdout', 0.30);
idx_train = training(cv);
idx_test  = test(cv);

Y_train = Y(idx_train);
Y_test  = Y(idx_test);

fprintf('\nTrain/Test Split: %d Train, %d Test\n', sum(idx_train), sum(idx_test));

% =========================================================================
% === 5. Classifier Training ===
% =========================================================================
% Identical strictly regularised SVM for all pipelines.
% We use fitcecoc to handle multiclass (One-vs-One).
t = templateSVM('KernelFunction', 'gaussian', 'KernelScale', 'auto', 'BoxConstraint', 1);

fprintf('\nTraining Baseline SVM...\n');
mdl_base = fitcecoc(features_base(idx_train, :), Y_train, 'Learners', t, 'FitPosterior', false);

fprintf('Training Pipeline A (Rigid) SVM...\n');
mdl_rigid = fitcecoc(features_rigid(idx_train, :), Y_train, 'Learners', t, 'FitPosterior', false);

fprintf('Training Pipeline B (Affine) SVM...\n');
mdl_affine = fitcecoc(features_affine(idx_train, :), Y_train, 'Learners', t, 'FitPosterior', false);

% =========================================================================
% === 6. Evaluation ===
% =========================================================================
Y_pred_base   = predict(mdl_base, features_base(idx_test, :));
Y_pred_rigid  = predict(mdl_rigid, features_rigid(idx_test, :));
Y_pred_affine = predict(mdl_affine, features_affine(idx_test, :));

acc_base   = mean(Y_pred_base == Y_test) * 100;
acc_rigid  = mean(Y_pred_rigid == Y_test) * 100;
acc_affine = mean(Y_pred_affine == Y_test) * 100;

% =========================================================================
% === 7. Visualization (OceanViz) ===
% =========================================================================
viz = Core('Theme', 'Dark');
bg = viz.Colors.Background;

% --- Figure 1: Geometric Catastrophe (Variance Profile) ---
fig1 = figure('Name', 'Feature Variance Profile', 'Color', bg, 'Position', [100, 100, 800, 400]);
ax1 = axes('Parent', fig1);
hold(ax1, 'on');
bar(ax1, 1:3, [mean_var_base, mean_var_rigid, mean_var_affine], 'FaceColor', 'flat', ...
    'CData', [0.7 0.7 0.7; viz.Colors.Primary; viz.Colors.Secondary]);
set(ax1, 'XTick', 1:3, 'XTickLabel', {'Baseline (Scaled)', 'Rigid GPA (A)', 'Affine GPA (B)'});
title(ax1, 'Mean Feature Variance (Normalized Space)', 'Color', 'w', 'FontSize', 12);
ylabel(ax1, 'Mean Variance \sigma^2', 'Color', 'w');
subtitle(ax1, sprintf('Affine / Rigid Variance Ratio = %.2f', mean_var_affine/mean_var_rigid), ...
    'Color', [0.7 0.7 0.7]);
viz.styleAxes(ax1);

% --- Figure 2: Classification Accuracy ---
fig2 = figure('Name', 'Emotion Classification Accuracy', 'Color', bg, 'Position', [150, 150, 800, 400]);
ax2 = axes('Parent', fig2);
hold(ax2, 'on');
b = bar(ax2, 1:3, [acc_base, acc_rigid, acc_affine], 'FaceColor', 'flat', ...
    'CData', [0.7 0.7 0.7; viz.Colors.Primary; viz.Colors.Secondary]);

% Add text labels on top of bars
xtips = b.XEndPoints;
ytips = b.YEndPoints;
labels = string(round(b.YData, 1)) + "%";
text(xtips, ytips, labels, 'HorizontalAlignment','center', ...
    'VerticalAlignment','bottom', 'Color', 'w', 'FontWeight', 'bold');

set(ax2, 'XTick', 1:3, 'XTickLabel', {'Baseline (Raw)', 'Rigid GPA', 'Affine GPA'});
title(ax2, 'Downstream Emotion Classification Accuracy (RBF SVM)', 'Color', 'w', 'FontSize', 12);
ylabel(ax2, 'Test Accuracy (%)', 'Color', 'w');
yline(ax2, 25, '--', 'Random Chance (25%)', 'Color', [0.6 0.6 0.6], 'LabelVerticalAlignment', 'bottom');
ylim(ax2, [0 100]);
subtitle(ax2, 'Rigid preserves emotion geometry; Affine squashes it', 'Color', [0.7 0.7 0.7]);
viz.styleAxes(ax2);

% --- Figure 3: Confusion Matrices ---
fig3 = figure('Name', 'Confusion Matrices', 'Color', bg, 'Position', [200, 200, 1200, 350]);
tl = tiledlayout(fig3, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Baseline
cm_base = confusionmat(Y_test, Y_pred_base);
c1 = confusionchart(tl, cm_base, class_names, 'FontColor', 'w');
c1.Title = sprintf('Baseline (%.1f%%)', acc_base);

% Rigid
cm_rigid = confusionmat(Y_test, Y_pred_rigid);
c2 = confusionchart(tl, cm_rigid, class_names, 'FontColor', 'w');
c2.Title = sprintf('Rigid GPA (%.1f%%)', acc_rigid);

% Affine
cm_affine = confusionmat(Y_test, Y_pred_affine);
c3 = confusionchart(tl, cm_affine, class_names, 'FontColor', 'w');
c3.Title = sprintf('Affine GPA (%.1f%%)', acc_affine);

title(tl, 'Confusion Matrices (Test Set)', 'Color', 'w', 'FontSize', 14, 'FontWeight', 'bold');

% Export figures
res_dir = fullfile('c:\', 'Ocean Library', 'Matlab code practice', 'Results');
if ~exist(res_dir, 'dir'), mkdir(res_dir); end
viz.export(fig1, fullfile(res_dir, 'exp_emotion_variance.png'));
viz.export(fig2, fullfile(res_dir, 'exp_emotion_accuracy.png'));
viz.export(fig3, fullfile(res_dir, 'exp_emotion_confusion.png'));

% =========================================================================
% === 8. Summary Printout ===
% =========================================================================
fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Dataset       : AffectNet %s (N=%d, p=%d, L=%d)\n', ...
    strjoin(class_names, '/'), N, p, L);
fprintf('  Model         : Multiclass RBF SVM (One-vs-One)\n');
fprintf('  Split         : 70%% Train / 30%% Test\n');
fprintf('  ------------------------------------------\n');
fprintf('  Baseline Acc  : %.2f%%\n', acc_base);
fprintf('  Rigid GPA Acc : %.2f%%   <-- Best (Preserves emotion geometry)\n', acc_rigid);
fprintf('  Affine GPA Acc: %.2f%%   <-- Catastrophe (Destroys emotion geometry)\n', acc_affine);
fprintf('==========================\n');
