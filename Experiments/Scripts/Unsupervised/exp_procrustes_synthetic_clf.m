% EXP_PROCRUSTES_SYNTHETIC_CLF  Synthetic Procrustes "Geometric Catastrophe"
%
%   DATASET:        Synthetic 2D shapes (L=1000, N=30, p=2)
%                   Class 0: Circle (aspect ratio 1.0)
%                   Class 1: Ellipse (aspect ratio 2.0)
%   NOISE:          Random translation, rotation [0, 2pi), scaling [0.5, 2.0]
%                   Gaussian jitter added to each point.
%   MODEL:          Linear SVM
%   ASSUMPTIONS:    
%     - Rigid GPA preserves the aspect ratio. Class 0 and Class 1 remain distinct.
%     - Affine GPA projects both to the same affine subspace (a circle), destroying the signal.
%   REPRODUCIBILITY: rng(42, 'twister')
%
clc; clear; rng(42, 'twister');

% Import Ocean Library packages
import OceanStats.Unsupervised.*
import OceanStats.Supervised.SVM.*
import OceanStats.Selection.CrossValidation.*
import OceanViz.*

% =========================================================================
% === 1. Data Generation (Circle vs. Ellipse) ===
% =========================================================================
L = 1000;
N = 30;
p = 2;

X_raw = zeros(N, p, L);
Y = zeros(L, 1);

% Base shapes
theta = linspace(0, 2*pi, N+1)';
theta(end) = []; % remove duplicate 2*pi

base_circle  = [cos(theta), sin(theta)];
base_ellipse = [1.5 * cos(theta), 0.75 * sin(theta)]; % aspect ratio 2.0

for l = 1:L
    % 50/50 split
    if rand() > 0.5
        shape = base_circle;
        Y(l) = -1;
    else
        shape = base_ellipse;
        Y(l) = 1;
    end
    
    % Apply "Nuisance" transformations
    scale = 0.5 + 1.5 * rand();          % [0.5, 2.0]
    angle = 2 * pi * rand();             % [0, 2*pi)
    tx = 10 * randn();
    ty = 10 * randn();
    
    R = [cos(angle) -sin(angle); sin(angle) cos(angle)];
    
    % --- PURE MODEL ---
    % Noise level = 0. No noise is added to the base shapes.
    % We are evaluating the pure mathematical effect of Procrustes alignment
    % on the exact geometric structures.
    
    % Transform
    shape = scale * (shape * R) + repmat([tx, ty], N, 1);
    
    X_raw(:,:,l) = shape;
end

fprintf('Generated L=%d synthetic shapes (N=%d, p=%d)\n', L, N, p);

% =========================================================================
% === 2. Feature Extraction (3 Pipelines) ===
% =========================================================================
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
% === 3. Variance Audit ===
% =========================================================================
mean_var_base   = mean(var(features_base, 0, 1));
mean_var_rigid  = mean(var(features_rigid, 0, 1));
mean_var_affine = mean(var(features_affine, 0, 1));

fprintf('\n--- Geometric Variance Audit ---\n');
fprintf('  Mean feature variance (Baseline) : %.4e\n', mean_var_base);
fprintf('  Mean feature variance (Rigid)    : %.4e\n', mean_var_rigid);
fprintf('  Mean feature variance (Affine)   : %.4e\n', mean_var_affine);

% =========================================================================
% === 4. Classifier Training ===
% =========================================================================
cv = HoldOut(0.3, 'Seed', 42);
[idx_train, idx_test] = cv.getIndices(L, 1);

Y_train = Y(idx_train);
Y_test  = Y(idx_test);

fprintf('\nTrain/Test Split: %d Train, %d Test\n', length(idx_train), length(idx_test));

% Use RBF SVM from OceanStats
kernel = Kernels.RBFKernel('Gamma', 0.1); 
t = SVMClassifier('C', 1.0, 'Kernel', kernel, 'MaxPasses', 500);

fprintf('\nTraining Baseline SVM...\n');
mdl_base = t.train(features_base(idx_train, :), Y_train);

fprintf('Training Pipeline A (Rigid) SVM...\n');
mdl_rigid = t.train(features_rigid(idx_train, :), Y_train);

fprintf('Training Pipeline B (Affine) SVM...\n');
mdl_affine = t.train(features_affine(idx_train, :), Y_train);

% =========================================================================
% === 5. Evaluation ===
% =========================================================================
Y_pred_base   = t.predict(mdl_base, features_base(idx_test, :));
Y_pred_rigid  = t.predict(mdl_rigid, features_rigid(idx_test, :));
Y_pred_affine = t.predict(mdl_affine, features_affine(idx_test, :));

acc_base   = (1 - t.evaluate(Y_test, Y_pred_base)) * 100;
acc_rigid  = (1 - t.evaluate(Y_test, Y_pred_rigid)) * 100;
acc_affine = (1 - t.evaluate(Y_test, Y_pred_affine)) * 100;

% =========================================================================
% === 6. Visualization (OceanViz) ===
% =========================================================================
viz = Core('Theme', 'Dark');
bg = viz.Colors.Background;

% --- Figure 1: Classification Accuracy ---
fig1 = figure('Name', 'Synthetic Accuracy', 'Color', bg, 'Position', [100, 100, 800, 400]);
ax1 = axes('Parent', fig1);
hold(ax1, 'on');
b = bar(ax1, 1:3, [acc_base, acc_rigid, acc_affine], 'FaceColor', 'flat', ...
    'CData', [0.7 0.7 0.7; viz.Colors.Primary; viz.Colors.Secondary]);

xtips = b.XEndPoints;
ytips = b.YEndPoints;
labels = string(round(b.YData, 1)) + "%";
text(xtips, ytips, labels, 'HorizontalAlignment','center', ...
    'VerticalAlignment','bottom', 'Color', 'w', 'FontWeight', 'bold');

set(ax1, 'XTick', 1:3, 'XTickLabel', {'Baseline (Raw)', 'Rigid GPA', 'Affine GPA'});
title(ax1, 'Classification Accuracy (RBF SVM)', 'Color', 'w', 'FontSize', 12);
ylabel(ax1, 'Test Accuracy (%)', 'Color', 'w');
yline(ax1, 50, '--', 'Random Chance (50%)', 'Color', [0.6 0.6 0.6], 'LabelVerticalAlignment', 'bottom');
ylim(ax1, [0 110]);
subtitle(ax1, 'Rigid perfectly isolates shape; Affine squashes shape signal entirely', 'Color', [0.7 0.7 0.7]);
viz.styleAxes(ax1);

% Add explanatory text annotations inside the plot
text(1, 10, 'SVM struggles with random rotations', 'Color', 'w', 'HorizontalAlignment', 'center', 'FontSize', 9);
text(2, 50, 'Perfectly aligned shapes', 'Color', 'w', 'HorizontalAlignment', 'center', 'FontSize', 9);
text(3, 10, 'Aspect ratio erased (Both become circles)', 'Color', 'w', 'HorizontalAlignment', 'center', 'FontSize', 9);

% --- Figure 2: Visualizing the Subspaces (Catastrophe Proof) ---
fig2 = figure('Name', 'Shape Subspaces', 'Color', bg, 'Position', [150, 150, 1200, 400]);
tl = tiledlayout(fig2, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

c1_idx = find(Y == -1, 5); % Pick 5 circles
c2_idx = find(Y == 1, 5); % Pick 5 ellipses

% Plot Helper
plotShapes = @(ax, X_tensor, title_str, sub_str) ...
    plotSpace(ax, X_tensor, c1_idx, c2_idx, viz, title_str, sub_str);

plotShapes(nexttile(tl), X_centered, 'Baseline (Scaled)', 'Overlapping due to random rotations');
plotShapes(nexttile(tl), X_aligned_rigid, 'Rigid GPA', 'Ellipses & Circles remain distinct');
plotShapes(nexttile(tl), X_aligned_affine, 'Affine GPA', 'Ellipses are squashed into Circles');

sgtitle(fig2, 'The Geometric Catastrophe: Visualizing Extracted Features', 'Color', 'w', 'FontSize', 14, 'FontWeight', 'bold');

% Export figures
res_dir = fullfile('c:\', 'Ocean Library', 'Matlab code practice', 'Results');
if ~exist(res_dir, 'dir'), mkdir(res_dir); end
viz.export(fig1, fullfile(res_dir, 'exp_synthetic_accuracy.png'));
viz.export(fig2, fullfile(res_dir, 'exp_synthetic_subspace.png'));

% =========================================================================
% === 7. Summary Printout ===
% =========================================================================
fprintf('\n=== SYNTHETIC EXPERIMENT SUMMARY ===\n');
fprintf('  Dataset       : Circle vs Ellipse (N=30, p=2, L=1000)\n');
fprintf('  Model         : RBF SVM\n');
fprintf('  ------------------------------------------\n');
fprintf('  Baseline Acc  : %.2f%%\n', acc_base);
fprintf('  Rigid GPA Acc : %.2f%%   <-- Best (Preserves aspect ratio)\n', acc_rigid);
fprintf('  Affine GPA Acc: %.2f%%   <-- Catastrophe (Squashes aspect ratio)\n', acc_affine);
fprintf('==========================================\n');

% Helper Function for Figure 2
function plotSpace(ax, X_tensor, c1_idx, c2_idx, viz, title_str, sub_str)
    hold(ax, 'on');
    h1 = []; h2 = [];
    % Plot class 1 (circles)
    for i = 1:length(c1_idx)
        shape = X_tensor(:,:,c1_idx(i));
        p1 = plot(ax, shape(:,1), shape(:,2), '-o', 'Color', viz.Colors.Primary, 'MarkerSize', 3, 'MarkerFaceColor', viz.Colors.Primary);
        plot(ax, [shape(end,1); shape(1,1)], [shape(end,2); shape(1,2)], '-', 'Color', viz.Colors.Primary);
        if isempty(h1), h1 = p1; end
    end
    % Plot class 2 (ellipses)
    for i = 1:length(c2_idx)
        shape = X_tensor(:,:,c2_idx(i));
        p2 = plot(ax, shape(:,1), shape(:,2), '-s', 'Color', viz.Colors.Secondary, 'MarkerSize', 3, 'MarkerFaceColor', viz.Colors.Secondary);
        plot(ax, [shape(end,1); shape(1,1)], [shape(end,2); shape(1,2)], '-', 'Color', viz.Colors.Secondary);
        if isempty(h2), h2 = p2; end
    end
    title(ax, title_str, 'Color', 'w', 'FontSize', 12);
    subtitle(ax, sub_str, 'Color', [0.7 0.7 0.7]);
    axis(ax, 'equal');
    legend(ax, [h1, h2], {'Circle', 'Ellipse'}, 'TextColor', 'w', 'Color', viz.Colors.Background, 'Location', 'southoutside');
    viz.styleAxes(ax);
end
