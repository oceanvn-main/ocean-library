% EXP_KNN_CLASSIFIERS  ESL Chapter 13.3 — k-Nearest Neighbor Classifiers
%
% Reproduces ESL Figures 13.3 and 13.4:
%   Figure 13.3: KNN decision boundaries at K=1 and K=15 (3-class mixture)
%   Figure 13.4: Error curves (train/test/CV) vs K=1..30 (2-class mixture)
%                + 7-NN decision boundary panel
%
% Pipeline: 8-step Ocean Library ML Pipeline
% Author: Ocean Library   Date: 2026-03-26
% Ref: Hastie et al. (2009), ESL Section 13.3, Figures 13.3 and 13.4
clc; clear; close all;
RNG_SEED = 42;

%% === 1. Data Generation ===
fprintf('=== Step 1: Data Generation ===\n');

% 3-class mixture (for Figure 13.3 — boundary panels)
gen3 = OceanData.Generators.ESLMixtureGenerator('N', 300, 'Seed', RNG_SEED, ...
    'NumClasses', 3, 'CentersPerClass', 10);
[X3, y3] = gen3.generate();
fprintf('[3-CLASS] N=%d | K=3\n', size(X3,1));

% 2-class mixture (for Figure 13.4 — error curves)
gen2 = OceanData.Generators.ESLMixtureGenerator('N', 200, 'Seed', RNG_SEED, ...
    'NumClasses', 2, 'CentersPerClass', 10);
[X2, y2] = gen2.generate();
fprintf('[2-CLASS] N=%d | K=2\n', size(X2,1));

%% === 1.5. Train/Test Split ===
fprintf('\n=== Step 1.5: Train/Test Split ===\n');

% 3-class: 62.5% train
rng(RNG_SEED);
sp3 = OceanData.DataArch.DataSplitter('Strategy','holdout','Ratio',0.375);
s3  = sp3.split(X3, y3);
X3_tr = X3(s3.trainIdx,:); y3_tr = y3(s3.trainIdx);
X3_te = X3(s3.testIdx, :); y3_te = y3(s3.testIdx);
fprintf('[3-CLASS] Train=%d | Test=%d\n', sum(s3.trainIdx), sum(s3.testIdx));

% 2-class: 50% train
sp2 = OceanData.DataArch.DataSplitter('Strategy','holdout','Ratio',0.5);
s2  = sp2.split(X2, y2);
X2_tr = X2(s2.trainIdx,:); y2_tr = y2(s2.trainIdx);
X2_te = X2(s2.testIdx, :); y2_te = y2(s2.testIdx);
fprintf('[2-CLASS] Train=%d | Test=%d\n', sum(s2.trainIdx), sum(s2.testIdx));

%% === 2-4. DataLake / DataWarehouse / CachingWarehouse ===
try
    lake = OceanData.DataArch.DataLake('rootDir', './lake/');
    lake.put('raw/esl_ch13_knn/X3_tr', X3_tr, struct('source','esl_3class','seed',RNG_SEED));
    lake.put('raw/esl_ch13_knn/X2_tr', X2_tr, struct('source','esl_2class','seed',RNG_SEED));
    fprintf('[PASS] DataLake saved.\n');
catch, fprintf('[SKIP] DataLake unavailable.\n'); end

%% === 5-6. Feature Engineering ===
% [SKIP] No feature transformation — raw 2D coordinates used directly.
fprintf('\n[Step 5-6] [SKIP] Raw 2D features retained.\n');

%% === 6.5. Model Selection / Hyperparameter Sweep ===
fprintf('\n=== Step 6.5: KNN Error Sweep (K=1..30 on 2-class data) ===\n');

Ks     = 1:30;
N_CV   = 10;   % 10-fold CV folds
nK     = numel(Ks);

trErr  = zeros(nK, 1);
teErr  = zeros(nK, 1);
cvErr  = zeros(nK, 1);
cvSE   = zeros(nK, 1);

% Bayes error on 2-class test set
[~, bayesErr2] = gen2.bayesPredict(X2_te, y2_te);

rng(RNG_SEED);
for ki = 1:nK
    K_val = Ks(ki);
    knn = OceanStats.Supervised.Neighbors.KNNClassifier('K', K_val);
    knn.fit(X2_tr, y2_tr);

    % Training error
    trErr(ki) = mean(knn.predict(X2_tr) ~= y2_tr);

    % Test error
    teErr(ki) = mean(knn.predict(X2_te) ~= y2_te);

    % 10-fold CV on TRAINING data only (no test leakage)
    N_tr = size(X2_tr, 1);
    cvStrategy = OceanStats.Selection.CrossValidation.KFold(N_CV, 'Seed', RNG_SEED);
    foldErr = zeros(N_CV, 1);
    
    for f = 1:N_CV
        [trainMask, valMask] = cvStrategy.getIndices(N_tr, f);
        
        knn_cv = OceanStats.Supervised.Neighbors.KNNClassifier('K', K_val);
        knn_cv.fit(X2_tr(trainMask,:), y2_tr(trainMask));
        
        y_val_pred = knn_cv.predict(X2_tr(valMask,:));
        foldErr(f) = mean(y_val_pred ~= y2_tr(valMask));
    end
    
    cvErr(ki) = mean(foldErr);
    cvSE(ki)  = std(foldErr) / sqrt(N_CV);

    if mod(K_val, 5) == 0 || K_val == 1 || K_val == 7
        fprintf('  K=%2d  Train=%.3f  Test=%.3f  CV=%.3f±%.3f\n', ...
            K_val, trErr(ki), teErr(ki), cvErr(ki), cvSE(ki));
    end
end

% Best K by CV
[~, bestKi] = min(cvErr);
fprintf('\n[Best K by CV] K=%d  (CV=%.3f, Test=%.3f)\n', ...
    Ks(bestKi), cvErr(bestKi), teErr(bestKi));

%% === 7. Final Evaluation (test set, K=7) ===
fprintf('\n=== Step 7: Final Evaluation ===\n');
K_FINAL = 7;
knn7 = OceanStats.Supervised.Neighbors.KNNClassifier('K', K_FINAL);
knn7.fit(X2_tr, y2_tr);
err7_tr = mean(knn7.predict(X2_tr) ~= y2_tr);
err7_te = mean(knn7.predict(X2_te) ~= y2_te);
fprintf('[K=%d] Train=%.3f | Test=%.3f | Bayes=%.3f\n', ...
    K_FINAL, err7_tr, err7_te, bayesErr2);

%% === 8. Visualization ===
fprintf('\n=== Step 8: Visualization ===\n');

% --- Shared grid for decision regions ---
margin = 0.5;
x1r = [min(X3(:,1))-margin, max(X3(:,1))+margin];
x2r = [min(X3(:,2))-margin, max(X3(:,2))+margin];
[xx3, yy3] = meshgrid(linspace(x1r(1),x1r(2),180), linspace(x2r(1),x2r(2),180));
X3_grid = [xx3(:), yy3(:)];

x1r2 = [min(X2(:,1))-margin, max(X2(:,1))+margin];
x2r2 = [min(X2(:,2))-margin, max(X2(:,2))+margin];
[xx2, yy2] = meshgrid(linspace(x1r2(1),x1r2(2),180), linspace(x2r2(1),x2r2(2),180));
X2_grid = [xx2(:), yy2(:)];

% Bayes boundary grids
D3 = gen3.bayesDensity(X3_grid);
[~, zz3_bayes] = max(D3, [], 2);
zz3_bayes = reshape(zz3_bayes, size(xx3));

D2 = gen2.bayesDensity(X2_grid);
[~, zz2_bayes] = max(D2, [], 2);
zz2_bayes = reshape(zz2_bayes, size(xx2));

viz = OceanViz.Core('Theme', 'Dark');

%% ---- Figure 13.3: K=1 and K=15 boundaries (3-class) ----
fig3 = viz.createFigure('ESL 13.3 — KNN Decision Boundaries');
fig3.Position = [50, 50, 1000, 520];

Ks_show = [15, 1];
axPos = {[0.06 0.06 0.42 0.78], [0.55 0.06 0.42 0.78]};

for pi = 1:2
    K_val = Ks_show(pi);
    knn_pi = OceanStats.Supervised.Neighbors.KNNClassifier('K', K_val);
    knn_pi.fit(X3_tr, y3_tr);
    zz = knn_pi.predict(X3_grid);
    zz = reshape(zz, size(xx3));

    ax = axes('Parent', fig3, 'Position', axPos{pi});
    d = struct('X', X3, 'Y', y3, ...
               'Grid', struct('xx',xx3,'yy',yy3,'zz',zz), ...
               'BayesGrid', struct('xx',xx3,'yy',yy3,'zz',zz3_bayes), ...
               'DataPointSize', 10, 'ProtoSize', 0, 'StippleDensity', 70);
    % Note: Prototypes field unused — pass empty to satisfy contract
    d.Prototypes  = zeros(0,2);
    d.ProtoLabels = zeros(0,1);
    viz.plot(ax, 'PrototypeClassifier', d, ...
        'Title', sprintf('%d-Nearest Neighbor', K_val), ...
        'ShowLegend', false);
end

sgtitle(fig3, 'ESL Figure 13.3: KNN Decision Boundaries (3-class mixture)', ...
    'Color', viz.Colors.Text, 'FontSize', 13, 'FontWeight', 'bold');

resultsDir = fullfile(fileparts(mfilename('fullpath')), 'Results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
saveas(fig3, fullfile(resultsDir, 'fig13_3_knn_boundaries.png'));
fprintf('[Step 8] Saved: fig13_3_knn_boundaries.png\n');

%% ---- Figure 13.4: Error curves (upper) + 7-NN boundary (lower) ----
fig4 = viz.createFigure('ESL 13.4 — KNN Error Curves and Boundary');
fig4.Position = [100, 30, 820, 1050];

% --- Upper panel: error curves vs K ---
ax_top = axes('Parent', fig4, 'Position', [0.12 0.56 0.80 0.38]);
hold(ax_top, 'on'); viz.styleAxes(ax_top);

% Invert x-axis: ESL plots 1/K increasing (complexity left-to-right).
% We keep K increasing left-to-right (standard) and match ESL's curve shapes.
pTE  = plot(ax_top, Ks, teErr,  '-',  'Color', [0.25 0.45 0.90], 'LineWidth', 2.0, 'DisplayName', 'Test Error');
pCV  = plot(ax_top, Ks, cvErr,  '-',  'Color', [0.20 0.75 0.35], 'LineWidth', 2.0, 'DisplayName', '10-fold CV');
pTR  = plot(ax_top, Ks, trErr,  '-',  'Color', [0.95 0.60 0.10], 'LineWidth', 2.0, 'DisplayName', 'Training Error');

% CV error bars (±1 SE)
errorbar(ax_top, Ks, cvErr, cvSE, 'Color', [0.20 0.75 0.35], ...
    'LineStyle', 'none', 'CapSize', 3, 'HandleVisibility', 'off');

% Bayes reference line
yline(ax_top, bayesErr2, '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5, ...
    'DisplayName', sprintf('Bayes Error (%.3f)', bayesErr2));

% Mark best-K from CV
scatter(ax_top, Ks(bestKi), cvErr(bestKi), 60, [0.20 0.75 0.35], 'filled', ...
    'HandleVisibility', 'off');

xlabel(ax_top, 'Number of Neighbors (K)', 'Color', viz.Colors.Text);
ylabel(ax_top, 'Misclassification Errors', 'Color', viz.Colors.Text);
title(ax_top, 'ESL Figure 13.4: KNN Error vs. K (2-class mixture)', ...
    'Color', viz.Colors.Text, 'FontSize', 12, 'FontWeight', 'bold');
legend(ax_top, 'Location', 'northeast', 'FontSize', 9, ...
    'EdgeColor', [0.3 0.3 0.3], 'TextColor', viz.Colors.Text, ...
    'Color', viz.Colors.Background);
xlim(ax_top, [0, 31]);
hold(ax_top, 'off');

% --- Lower panel: 7-NN boundary (2-class) ---
ax_bot = axes('Parent', fig4, 'Position', [0.08 0.05 0.86 0.40]);

zz7 = knn7.predict(X2_grid);
zz7 = reshape(zz7, size(xx2));

d7 = struct('X', X2, 'Y', y2, ...
            'Grid',      struct('xx',xx2,'yy',yy2,'zz',zz7), ...
            'BayesGrid', struct('xx',xx2,'yy',yy2,'zz',zz2_bayes), ...
            'Prototypes', zeros(0,2), 'ProtoLabels', zeros(0,1), ...
            'DataPointSize', 12, 'ProtoSize', 0, 'StippleDensity', 75);
viz.plot(ax_bot, 'PrototypeClassifier', d7, ...
    'Title', sprintf('%d-Nearest Neighbors', K_FINAL), ...
    'ShowLegend', false);

% Annotate corner with errors (ESL style)
annotation(fig4, 'textbox', [0.10 0.06 0.28 0.09], ...
    'String', sprintf('Training Error: %.3f\nTest Error:   %.3f\nBayes Error:  %.3f', ...
        err7_tr, err7_te, bayesErr2), ...
    'Color', viz.Colors.Text, 'BackgroundColor', [0.1 0.1 0.1 0.85], ...
    'EdgeColor', [0.4 0.4 0.4], 'FontSize', 9, 'FitBoxToText', 'on');

% (sgtitle omitted — panel titles are sufficient to avoid overlap)

saveas(fig4, fullfile(resultsDir, 'fig13_4_knn_error_curves.png'));
fprintf('[Step 8] Saved: fig13_4_knn_error_curves.png\n');

fprintf('\n=== Done ===\n');
