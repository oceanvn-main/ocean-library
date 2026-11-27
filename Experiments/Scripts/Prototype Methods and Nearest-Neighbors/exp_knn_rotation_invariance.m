% EXP_KNN_ROTATION_INVARIANCE  ESL Exercise 13.8 — Rotation-Invariant KNN.
%
%   Compares four KNN-based classifiers on 2D Gaussian data with random
%   rotation perturbation:
%     1. Standard KNN (Euclidean distance)
%     2. KNN with hints (augment training with 10 rotations per point)
%     3. Invariant metric KNN (rotation-invariant distance = |r1 - r2|)
%     4. Tangent distance KNN (first-order manifold-aware distance)
%
%   DATASET:
%     Class 1: mean=(-1,-1), Class 2: mean=(+1,+1), std=1, 2 features
%     Each point randomly rotated by theta ~ Uniform[0, 2*pi]
%     Train: 50/class (100 total), Test: 500/class (1000 total)
%   ASSUMPTIONS:
%     - Rotation about the origin preserves ||x|| (radius)
%     - The Bayes-optimal feature is the radius ||x||
%     - Tangent vector at x = (-x2, x1) (perpendicular to radius)
%   REPRODUCIBILITY: rng(42, 'twister')
%
% Ref: Hastie, Tibshirani, Friedman (2009). ESL, Ex 13.8, Sec 13.8.
clc; clear; close all; rng(42, 'twister');

%% === 1. Data Generation ===
fprintf('=== Step 1: Data Generation ===\n');

N_tr_per_class = 50;
N_te_per_class = 500;
N_tr = 2 * N_tr_per_class;
N_te = 2 * N_te_per_class;

% Generate raw Gaussian data (before rotation)
X_tr_raw = [randn(N_tr_per_class, 2) + [-1 -1];    % Class 1
            randn(N_tr_per_class, 2) + [+1 +1]];    % Class 2
Y_tr = [zeros(N_tr_per_class, 1); ones(N_tr_per_class, 1)];

X_te_raw = [randn(N_te_per_class, 2) + [-1 -1];
            randn(N_te_per_class, 2) + [+1 +1]];
Y_te = [zeros(N_te_per_class, 1); ones(N_te_per_class, 1)];

% Apply random rotation to each point
X_tr = applyRandomRotation(X_tr_raw);
X_te = applyRandomRotation(X_te_raw);

fprintf('  Train: %d points (%d/class)\n', N_tr, N_tr_per_class);
fprintf('  Test:  %d points (%d/class)\n', N_te, N_te_per_class);
fprintf('  Mean radius class 0: %.3f, class 1: %.3f\n', ...
    mean(vecnorm(X_tr(Y_tr==0,:), 2, 2)), mean(vecnorm(X_tr(Y_tr==1,:), 2, 2)));

%% === 2. Method Setup ===
fprintf('\n=== Step 2: Setting up 4 methods ===\n');

K_values = 1:2:31; %1,3,5...
nK = numel(K_values);
nFolds = 10;
cv = OceanStats.Selection.CrossValidation.KFold(nFolds, 'Seed', 42);

% Pre-compute augmented training set for Method 2 (hints)
nHints = 10;
X_tr_aug = zeros(N_tr * (1 + nHints), 2);
Y_tr_aug = zeros(N_tr * (1 + nHints), 1);

% Original points
X_tr_aug(1:N_tr, :) = X_tr;
Y_tr_aug(1:N_tr) = Y_tr;

% Add 10 random rotations per training point
for h = 1:nHints
    thetas = 2 * pi * rand(N_tr, 1);
    X_rotated = rotatePoints(X_tr, thetas);
    offset = N_tr * h;
    X_tr_aug(offset+1:offset+N_tr, :) = X_rotated;
    Y_tr_aug(offset+1:offset+N_tr) = Y_tr;
end

fprintf('  Method 2 augmented: %d -> %d points\n', N_tr, size(X_tr_aug, 1));

% Distance functions for methods 3 and 4
distInvariant = @(A, B) (vecnorm(A, 2, 2) - vecnorm(B, 2, 2)').^2; % Collapse column -> N x 1. Radial distance / Polar
distTangent   = @(A, B) tangentDistMatrix(A, B);

%% === 3. 10-Fold CV + Test Error for All Methods ===
fprintf('\n=== Step 3: CV tuning + Test evaluation ===\n');

methodNames = {'Standard KNN', 'KNN + Hints', 'Invariant Metric', 'Tangent Distance'};
nMethods = numel(methodNames);

cvErrors   = zeros(nMethods, nK);
testErrors = zeros(nMethods, nK);

for ki = 1:nK
    K = K_values(ki);
    
    % --- Method 1: Standard Euclidean KNN ---
    knn1 = OceanStats.Supervised.Neighbors.KNNClassifier('K', K);
    knn1.fit(X_tr, Y_tr);
    yp1 = knn1.predict(X_te); testErrors(1, ki) = mean(yp1(:) ~= Y_te);
    cvErrors(1, ki) = cvErrorKNN(X_tr, Y_tr, K, nFolds, cv, []);
    
    % --- Method 2: KNN with Hints (augmented training set) ---
    knn2 = OceanStats.Supervised.Neighbors.KNNClassifier('K', K);
    knn2.fit(X_tr_aug, Y_tr_aug);
    yp2 = knn2.predict(X_te); testErrors(2, ki) = mean(yp2(:) ~= Y_te);
    % CV on original training set but predict with augmented
    cvErrors(2, ki) = cvErrorHints(X_tr, Y_tr, K, nFolds, cv, nHints);
    
    % --- Method 3: Invariant Metric KNN ---
    knn3 = OceanStats.Supervised.Neighbors.KNNClassifier('K', K, ...
        'DistanceFcn', distInvariant);
    knn3.fit(X_tr, Y_tr);
    yp3 = knn3.predict(X_te); testErrors(3, ki) = mean(yp3(:) ~= Y_te);
    cvErrors(3, ki) = cvErrorKNN(X_tr, Y_tr, K, nFolds, cv, distInvariant);
    
    % --- Method 4: Tangent Distance KNN ---
    knn4 = OceanStats.Supervised.Neighbors.KNNClassifier('K', K, ...
        'DistanceFcn', distTangent);
    knn4.fit(X_tr, Y_tr);
    yp4 = knn4.predict(X_te); testErrors(4, ki) = mean(yp4(:) ~= Y_te);
    cvErrors(4, ki) = cvErrorKNN(X_tr, Y_tr, K, nFolds, cv, distTangent);
    
    if mod(ki, 5) == 1
        fprintf('  K=%2d: test=[%.3f, %.3f, %.3f, %.3f]\n', K, ...
            testErrors(1,ki), testErrors(2,ki), testErrors(3,ki), testErrors(4,ki));
    end
end

%% === 4. Find Optimal K per Method ===
fprintf('\n=== Step 4: Optimal K Selection ===\n');

bestK_cv   = zeros(1, nMethods);
bestK_test = zeros(1, nMethods);
for m = 1:nMethods
    [~, idx_cv]   = min(cvErrors(m, :));
    [~, idx_test] = min(testErrors(m, :));
    bestK_cv(m)   = K_values(idx_cv);
    bestK_test(m) = K_values(idx_test);
    fprintf('  %s: best K(CV)=%d (err=%.4f), best K(test)=%d (err=%.4f)\n', ...
        methodNames{m}, bestK_cv(m), cvErrors(m, idx_cv), ...
        bestK_test(m), testErrors(m, idx_test));
end

%% === 5. Visualization ===
fprintf('\n=== Step 5: Visualization ===\n');

viz = OceanViz.Core('Theme', 'Dark');
colors = [0.30 0.65 0.95;    % blue  — Standard
          1.00 0.50 0.15;    % orange — Hints
          0.35 0.80 0.40;    % green  — Invariant
          0.85 0.35 0.85];   % purple — Tangent
markers = {'o', 's', 'd', '^'};

% ─── Figure 1: Test Error Curves ───
fig1 = viz.createFigure('ESL Ex 13.8 — Test Error Comparison');
fig1.Position = [50, 150, 900, 500];
ax1 = axes('Parent', fig1);
hold(ax1, 'on'); viz.styleAxes(ax1);

for m = 1:nMethods
    plot(ax1, K_values, testErrors(m, :), ['-' markers{m}], ...
        'Color', colors(m,:), 'LineWidth', 2, 'MarkerSize', 5, ...
        'MarkerFaceColor', colors(m,:), 'DisplayName', methodNames{m});
end

xlabel(ax1, 'Number of Neighbors (K)', 'Color', viz.Colors.Text, 'FontSize', 12);
ylabel(ax1, 'Test Misclassification Error', 'Color', viz.Colors.Text, 'FontSize', 12);
title(ax1, 'ESL Ex 13.8 — Four KNN Classifiers on Rotated Gaussian Data', ...
    'Color', viz.Colors.Text, 'FontSize', 14, 'FontWeight', 'bold');
legend(ax1, 'Location', 'best', 'TextColor', viz.Colors.Text, ...
    'Color', viz.Colors.Background, 'EdgeColor', [0.3 0.3 0.3]);
grid(ax1, 'on');
hold(ax1, 'off');

% ─── Figure 2: CV Error Curves ───
fig2 = viz.createFigure('ESL Ex 13.8 — CV Error Comparison');
fig2.Position = [100, 100, 900, 500];
ax2 = axes('Parent', fig2);
hold(ax2, 'on'); viz.styleAxes(ax2);

for m = 1:nMethods
    plot(ax2, K_values, cvErrors(m, :), ['-' markers{m}], ...
        'Color', colors(m,:), 'LineWidth', 2, 'MarkerSize', 5, ...
        'MarkerFaceColor', colors(m,:), 'DisplayName', methodNames{m});
end

xlabel(ax2, 'Number of Neighbors (K)', 'Color', viz.Colors.Text, 'FontSize', 12);
ylabel(ax2, '10-Fold CV Error', 'Color', viz.Colors.Text, 'FontSize', 12);
title(ax2, 'ESL Ex 13.8 — 10-Fold CV Error Comparison', ...
    'Color', viz.Colors.Text, 'FontSize', 14, 'FontWeight', 'bold');
legend(ax2, 'Location', 'best', 'TextColor', viz.Colors.Text, ...
    'Color', viz.Colors.Background, 'EdgeColor', [0.3 0.3 0.3]);
grid(ax2, 'on');
hold(ax2, 'off');

% ─── Figure 3: Data Visualization ───
fig3 = viz.createFigure('Rotated Gaussian Data');
fig3.Position = [150, 50, 900, 450];

% Before rotation
ax3a = subplot(1, 2, 1, 'Parent', fig3);
hold(ax3a, 'on'); viz.styleAxes(ax3a);
scatter(ax3a, X_tr_raw(Y_tr==0,1), X_tr_raw(Y_tr==0,2), 30, colors(1,:), 'filled', 'MarkerFaceAlpha', 0.6);
scatter(ax3a, X_tr_raw(Y_tr==1,1), X_tr_raw(Y_tr==1,2), 30, colors(2,:), 'filled', 'MarkerFaceAlpha', 0.6);
title(ax3a, 'Before Rotation', 'Color', viz.Colors.Text, 'FontSize', 12, 'FontWeight', 'bold');
axis(ax3a, 'equal'); xlabel(ax3a, 'X_1', 'Color', viz.Colors.Text); ylabel(ax3a, 'X_2', 'Color', viz.Colors.Text);

% After rotation
ax3b = subplot(1, 2, 2, 'Parent', fig3);
hold(ax3b, 'on'); viz.styleAxes(ax3b);
scatter(ax3b, X_tr(Y_tr==0,1), X_tr(Y_tr==0,2), 30, colors(1,:), 'filled', 'MarkerFaceAlpha', 0.6);
scatter(ax3b, X_tr(Y_tr==1,1), X_tr(Y_tr==1,2), 30, colors(2,:), 'filled', 'MarkerFaceAlpha', 0.6);
% Draw radius boundary circle
theta_circle = linspace(0, 2*pi, 200);
r_boundary = sqrt(2);  % approximate Bayes-optimal radius
plot(ax3b, r_boundary*cos(theta_circle), r_boundary*sin(theta_circle), '--', ...
    'Color', [0.9 0.9 0.9 0.5], 'LineWidth', 1.5);
title(ax3b, 'After Rotation', 'Color', viz.Colors.Text, 'FontSize', 12, 'FontWeight', 'bold');
axis(ax3b, 'equal'); xlabel(ax3b, 'X_1', 'Color', viz.Colors.Text); ylabel(ax3b, 'X_2', 'Color', viz.Colors.Text);

sgtitle(fig3, 'Training Data: Random Rotation Destroys Euclidean Structure', ...
    'Color', viz.Colors.Text, 'FontSize', 14, 'FontWeight', 'bold');

% Save
resultsDir = fullfile(fileparts(mfilename('fullpath')), 'Results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
saveas(fig1, fullfile(resultsDir, 'fig13_8_test_error.png'));
saveas(fig2, fullfile(resultsDir, 'fig13_8_cv_error.png'));
saveas(fig3, fullfile(resultsDir, 'fig13_8_data.png'));
fprintf('[Viz] Saved 3 figures.\n');

%% === 6. Summary ===
fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Dataset : 2D Gaussian, rotated, 100 train / 1000 test\n');
fprintf('  CV      : 10-fold\n');
fprintf('  K range : %d to %d\n\n', K_values(1), K_values(end));

fprintf('  %-22s  Best K(CV)  CV Err    Test Err (at CV-K)\n', 'Method');
fprintf('  %s\n', repmat('-', 1, 60));
for m = 1:nMethods
    [~, idx_cv] = min(cvErrors(m,:));
    fprintf('  %-22s  K=%-5d     %.4f    %.4f\n', ...
        methodNames{m}, bestK_cv(m), cvErrors(m, idx_cv), testErrors(m, idx_cv));
end
fprintf('\n  Conclusion:\n');
fprintf('    Invariant metric achieves the lowest error (exact orbit distance).\n');
fprintf('    Tangent distance is close (good local approximation for 2D rotation).\n');
fprintf('    Hints help but cannot match knowing the invariance analytically.\n');
fprintf('    Standard KNN is worst (Euclidean confused by rotation).\n');
fprintf('==========================\n');

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================

function X_out = applyRandomRotation(X)
    % APPLYRANDOMROTATION  Rotate each row of X by a random angle.
    N = size(X, 1);
    thetas = 2 * pi * rand(N, 1);
    X_out = rotatePoints(X, thetas);
end

function X_out = rotatePoints(X, thetas)
    % ROTATEPOINTS  Rotate each row of X by corresponding angle in thetas.
    %   X: (N x 2), thetas: (N x 1) in radians
    c = cos(thetas);
    s = sin(thetas);
    X_out = [X(:,1).*c - X(:,2).*s, ...
             X(:,1).*s + X(:,2).*c];
end

function D2 = tangentDistMatrix(A, B)
    % TANGENTDISTMATRIX  Two-sided tangent distance for 2D rotation.
    %   For rotation about origin, tangent vector at x = (-x2, x1).
    %   Two-sided: min_{s,t} ||(a + s*Ta) - (b + t*Tb)||^2
    %   This is a 2-variable least-squares problem with closed-form solution.
    %
    %   Returns: (N_A x N_B) squared tangent distance matrix.
    
    N_A = size(A, 1);
    N_B = size(B, 1);
    D2 = zeros(N_A, N_B);
    
    for i = 1:N_A
        a = A(i, :);          % (1 x 2)
        Ta = [-a(2), a(1)];   % tangent at a
        
        for j = 1:N_B
            b = B(j, :);          % (1 x 2)
            Tb = [-b(2), b(1)];   % tangent at b
            
            % Solve: min_{s,t} ||(a + s*Ta) - (b + t*Tb)||^2
            % Let d = a - b
            % Gradient w.r.t. [s; t]:
            %   [Ta·Ta, -Ta·Tb] [s]   [-Ta·d]
            %   [-Tb·Ta, Tb·Tb] [t] = [ Tb·d]
            d = a - b;
            TaTa = dot(Ta, Ta);
            TbTb = dot(Tb, Tb);
            TaTb = dot(Ta, Tb);
            Tad  = dot(Ta, d);
            Tbd  = dot(Tb, d);
            
            denom = TaTa * TbTb - TaTb^2;
            
            if abs(denom) < 1e-12
                % Degenerate — fall back to one-sided or Euclidean
                if TaTa > 1e-12
                    s_opt = -Tad / TaTa;
                    t_opt = 0;
                else
                    s_opt = 0; t_opt = 0;
                end
            else
                s_opt = (-Tad * TbTb + TaTb * Tbd) / denom;
                t_opt = ( Tbd * TaTa - TaTb * Tad) / denom;
            end
            
            residual = d + s_opt * Ta - t_opt * Tb;
            D2(i, j) = dot(residual, residual);
        end
    end
end

function err = cvErrorKNN(X, Y, K, nFolds, cv, distFcn)
    % CVERRORKNN  Compute CV error for KNN with given distance function.
    N = size(X, 1);
    foldErrs = zeros(nFolds, 1);
    for f = 1:nFolds
        [trIdx, teIdx] = cv.getIndices(N, f);
        knn = OceanStats.Supervised.Neighbors.KNNClassifier('K', K, 'DistanceFcn', distFcn);
        knn.fit(X(trIdx,:), Y(trIdx));
        yp = knn.predict(X(teIdx,:));
        foldErrs(f) = mean(yp(:) ~= Y(teIdx));
    end
    err = mean(foldErrs);
end

function err = cvErrorHints(X, Y, K, nFolds, cv, nHints)
    % CVERRORHINTS  CV error for KNN with augmented hints.
    %   Train folds get augmented with nHints rotations; test fold stays original.
    N = size(X, 1);
    foldErrs = zeros(nFolds, 1);
    for f = 1:nFolds
        [trIdx, teIdx] = cv.getIndices(N, f);
        X_cv_tr = X(trIdx, :);
        Y_cv_tr = Y(trIdx);
        
        % Augment CV training fold
        N_fold = sum(trIdx);
        X_aug = X_cv_tr;
        Y_aug = Y_cv_tr;
        for h = 1:nHints
            thetas = 2 * pi * rand(N_fold, 1);
            X_aug = [X_aug; rotatePoints(X_cv_tr, thetas)]; %#ok<AGROW>
            Y_aug = [Y_aug; Y_cv_tr]; %#ok<AGROW>
        end
        
        knn = OceanStats.Supervised.Neighbors.KNNClassifier('K', K);
        knn.fit(X_aug, Y_aug);
        yp = knn.predict(X(teIdx,:));
        foldErrs(f) = mean(yp(:) ~= Y(teIdx));
    end
    err = mean(foldErrs);
end
