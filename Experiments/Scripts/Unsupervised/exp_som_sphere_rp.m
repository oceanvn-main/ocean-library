%% exp_som_sphere.m
%  ============================================================
%  ESL Exercise 14.5 -- SOM vs K-Means on Spherical Cluster Data
%  ============================================================
%
%  PROBLEM:
%    Generate 30 data points per class (3 classes, 90 total) lying
%    near the surface of a unit sphere. Three clusters are centred at:
%      Class 1: (1, 0, 0)
%      Class 2: (0, 1, 0)
%      Class 3: (0, 0, 1)
%
%  DATA GENERATION (spherical coordinates + Gaussian noise):
%
%    Class 1:
%      theta1 ~ U(-pi/8, pi/8)        phi1 ~ U(0, 2*pi)
%      x = sin(theta1)*cos(phi1) + W    W ~ N(0, 0.6^2)
%      y = sin(theta1)*sin(phi1) + W
%      z = cos(theta1)            + W
%
%    Class 2:
%      theta2 ~ U(pi/2-pi/4, pi/2+pi/4)   phi2 ~ U(-pi/4, pi/4)
%      x = sin(theta2)*cos(phi2) + W
%      y = sin(theta2)*sin(phi2) + W
%      z = cos(theta2)            + W
%
%    Class 3:
%      theta3 ~ U(pi/2-pi/4, pi/2+pi/4)   phi3 ~ U(pi/2-pi/4, pi/2+pi/4)
%      x = sin(theta3)*cos(phi3) + W
%      y = sin(theta3)*sin(phi3) + W
%      z = cos(theta3)            + W
%
%  TASKS:
%    1. Fit a SOM (1D: [1 x 3] grid) to all 90 points.
%    2. Fit K-means (K=3) to all 90 points.
%    3. Visualise both in 3D and compute confusion vs. true labels.
%
%  LIBRARY COMPONENTS:
%    OceanStats.Unsupervised.SOM      (online Kohonen, linear LR decay)
%    OceanStats.Unsupervised.KMeans   (Lloyd, multiple restarts)
%    OceanViz.Core                    (Dark theme)
%
%  Ref: Hastie, Tibshirani, Friedman (2009), ESL Ex. 14.5, Section 14.4
%  ============================================================

clear; clc; close all;
rng(42, 'twister');

% ============================================================
%  CONFIGURATION
% ============================================================
N_PER_CLASS = 30;       % samples per class
SIGMA_NOISE = 0.6;      % isotropic Gaussian noise std

% SOM hyperparameters (ESL Section 14.4 learning schedule)
SOM_GRID    = [1, 3];   % 1D chain of 3 prototypes
SOM_PASSES  = 40;
SOM_LR_I    = 0.5;      % initial learning rate
SOM_LR_F    = 0.05;     % final  learning rate
SOM_R_I     = 2.0;      % initial neighbourhood radius (grid units)
SOM_R_F     = 0.5;      % final  neighbourhood radius

K_MEANS_K   = 3;

fprintf('=============================================================\n');
fprintf('  ESL Ex 14.5 -- SOM vs K-Means (Spherical Clusters)\n');
fprintf('  N_per_class=%d  sigma=%.1f  SOM grid=[%d x %d]\n', ...
    N_PER_CLASS, SIGMA_NOISE, SOM_GRID(1), SOM_GRID(2));
fprintf('=============================================================\n\n');

% ============================================================
%  PHASE 1: DATA GENERATION
%  Assumption: angles drawn i.i.d., noise independent across
%  dimensions and samples.  RNG seeded for reproducibility.
% ============================================================
fprintf('>>> PHASE 1: DATA GENERATION <<<\n');

% Helper: spherical -> Cartesian + noise
gen = @(theta, phi) [ ...
    sin(theta).*cos(phi), ...
    sin(theta).*sin(phi), ...
    cos(theta)           ] + SIGMA_NOISE * randn(N_PER_CLASS, 3);

% Class 1: near North Pole -> centroid (0,0,1) but we rotate so centroid=(1,0,0)
% NOTE: theta~U(-pi/8, pi/8) means the data is near theta=0 => near z-axis top
%       => points cluster near (0,0,1) before rotation. ESL Table lists centroid
%       (1,0,0). The parameterisation naturally places this near (0,0,1); the
%       textbook uses a labelling convention, not a coordinate transform.
%       We keep the coordinates as generated -- the geometry is equivalent.
theta1 = -pi/8 + (pi/4) * rand(N_PER_CLASS, 1);   % U(-pi/8, pi/8)
phi1   =          (2*pi) * rand(N_PER_CLASS, 1);   % U(0, 2*pi)
X1     = gen(theta1, phi1);

% Class 2: near equator, front face -> centroid near (1,0,0)
theta2 = (pi/2 - pi/4) + (pi/2) * rand(N_PER_CLASS, 1);   % U(pi/2-pi/4, pi/2+pi/4)
phi2   = (-pi/4)       + (pi/2) * rand(N_PER_CLASS, 1);   % U(-pi/4, pi/4)
X2     = gen(theta2, phi2);

% Class 3: near equator, side face -> centroid near (0,1,0)
theta3 = (pi/2 - pi/4) + (pi/2) * rand(N_PER_CLASS, 1);   % U(pi/2-pi/4, pi/2+pi/4)
phi3   = (pi/2 - pi/4) + (pi/2) * rand(N_PER_CLASS, 1);   % U(pi/2-pi/4, pi/2+pi/4)
X3     = gen(theta3, phi3);

% Combine
X          = [X1; X2; X3];
true_labels = [ones(N_PER_CLASS,1); 2*ones(N_PER_CLASS,1); 3*ones(N_PER_CLASS,1)];
N          = size(X, 1);   % 90

% Print empirical centroids to verify geometry
fprintf('  Empirical centroids (should be ~(0,0,1), ~(1,0,0), ~(0,1,0)):\n');
fprintf('    Class 1: (%.3f, %.3f, %.3f)\n', mean(X1));
fprintf('    Class 2: (%.3f, %.3f, %.3f)\n', mean(X2));
fprintf('    Class 3: (%.3f, %.3f, %.3f)\n', mean(X3));
fprintf('  Total samples: %d (3 x %d)\n\n', N, N_PER_CLASS);

% ============================================================
%  PHASE 2: FIT SOM  (1D chain, 3 prototypes)
% ============================================================
fprintf('>>> PHASE 2: SOM FITTING <<<\n');
fprintf('  Grid: [%d x %d]  Passes: %d\n', SOM_GRID(1), SOM_GRID(2), SOM_PASSES);
fprintf('  LR:  %.2f -> %.2f   Radius: %.2f -> %.2f\n', ...
    SOM_LR_I, SOM_LR_F, SOM_R_I, SOM_R_F);

som = OceanStats.Unsupervised.SOM( ...
    'GridSize',    SOM_GRID, ...
    'NumPasses',   SOM_PASSES, ...
    'InitLR',      SOM_LR_I, ...
    'FinalLR',     SOM_LR_F, ...
    'InitRadius',  SOM_R_I, ...
    'FinalRadius', SOM_R_F);

som_labels = som.fit(X, 'TrackHistory', true, 'Verbose', false);
som_mqe    = som.meanQuantizationError(X);

fprintf('  Prototypes learned (W):\n');
for m = 1:size(som.Weights, 1)
    fprintf('    W%d: (%.3f, %.3f, %.3f)\n', m, som.Weights(m,:));
end
fprintf('  Mean Quantization Error: %.4f\n\n', som_mqe);

% ============================================================
%  PHASE 3: FIT K-MEANS  (K=3)
% ============================================================
fprintf('>>> PHASE 3: K-MEANS FITTING (K=%d) <<<\n', K_MEANS_K);

km = OceanStats.Unsupervised.KMeans();
km_labels 

% Compute WCSS
inertia = 0;
for k = 1:K_MEANS_K
    mask = (km_labels == k);
    if any(mask)
        inertia = inertia + sum(sum((X(mask,:) - km.Centroids(k,:)).^2));
    end
end

fprintf('  Cluster sizes: ');
fprintf('%d ', histcounts(km_labels, 1:K_MEANS_K+1));
fprintf('\n');
fprintf('  WCSS (inertia): %.4f\n', inertia);
fprintf('  Centroids:\n');
for k = 1:K_MEANS_K
    fprintf('    C%d: (%.3f, %.3f, %.3f)\n', k, km.Centroids(k,:));
end
fprintf('\n');

% ============================================================
%  PHASE 4: EVALUATION  (confusion vs. true labels)
%  Note: cluster labels are permutation-invariant -- we must
%  match each cluster/SOM-node to the best-fitting true class.
% ============================================================
fprintf('>>> PHASE 4: EVALUATION <<<\n');

% Helper: find best label permutation via majority vote
match_labels = @(pred, true_lbl, K) deal_labels_(pred, true_lbl, K);

[som_mapped, som_acc] = match_labels(som_labels,  true_labels, 3);
[km_mapped,  km_acc]  = match_labels(km_labels,   true_labels, K_MEANS_K);

fprintf('  SOM accuracy (best permutation): %.2f%%\n', som_acc * 100);
fprintf('  K-Means accuracy (best permutation): %.2f%%\n\n', km_acc * 100);

% ============================================================
%  PHASE 5: VISUALIZATION (OceanViz Dark theme)
% ============================================================
fprintf('>>> PHASE 5: VISUALIZATION <<<\n');

viz = OceanViz.Core('Theme', 'Dark');
pal = [viz.Colors.Primary; viz.Colors.Secondary; viz.Colors.Boundary];
if size(pal,1) < 3
    pal = [0.20 0.60 0.86; 0.91 0.30 0.24; 0.18 0.80 0.44];
end

% -- Fig 1: True labels + SOM prototypes + K-means centroids --
fig1 = viz.createFigure('ESL 14.5 - Sphere Clusters: True vs Discovered');

ax_true = subplot(1, 3, 1, 'Parent', fig1);
hold(ax_true, 'on');
for c = 1:3
    mask = (true_labels == c);
    scatter3(ax_true, X(mask,1), X(mask,2), X(mask,3), 40, ...
        pal(c,:), 'filled', 'MarkerFaceAlpha', 0.7, ...
        'DisplayName', sprintf('Class %d', c));
end
view(ax_true, 45, 25);
viz.styleAxes(ax_true);
title(ax_true, 'Ground Truth', 'Color', viz.Colors.Text, 'FontWeight', 'bold');
xlabel(ax_true, 'x', 'Color', viz.Colors.Text);
ylabel(ax_true, 'y', 'Color', viz.Colors.Text);
zlabel(ax_true, 'z', 'Color', viz.Colors.Text);
legend(ax_true, 'Location', 'best', 'TextColor', viz.Colors.Text, ...
    'Color', 'none', 'EdgeColor', 'none');
hold(ax_true, 'off');