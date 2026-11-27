%% exp_mars_viz_demo.m
% MARS vs PolyMARS: The Interlocking Spiral Challenge (3-Class)
% Goal: Prove PolyMARS's ability to trace winding, non-linear boundaries.

addpath(genpath(pwd));
viz = OceanViz.Core('Theme', 'Dark');

%% 1. Generate 3-Class Spiral Data
rng(42);
N_per_class = 400;
N = N_per_class * 3;

% Spiral Parameters
theta = linspace(0.5 * pi, 4 * pi, N_per_class)';
noise = 0.4;

% Class 1: Base Spiral
r1 = theta;
X1 = [r1 .* cos(theta), r1 .* sin(theta)] + noise * randn(N_per_class, 2);
Y1 = ones(N_per_class, 1);

% Class 2: Rotated 120 deg
X2 = [r1 .* cos(theta + 2*pi/3), r1 .* sin(theta + 2*pi/3)] + noise * randn(N_per_class, 2);
Y2 = 2 * ones(N_per_class, 1);

% Class 3: Rotated 240 deg
X3 = [r1 .* cos(theta + 4*pi/3), r1 .* sin(theta + 4*pi/3)] + noise * randn(N_per_class, 2);
Y3 = 3 * ones(N_per_class, 1);

X = [X1; X2; X3];
Y = [Y1; Y2; Y3];

%% 2. Fit Models
% Using High complexity to capture the winding curves
options = struct('MaxTerms', 51, 'MaxInteractionDegree', 3);

fprintf('--- Fitting Standard MARS (Indicator) ---\n');
m_std = OceanStats.Supervised.MARS.MARS('MaxTerms', options.MaxTerms, ...
    'MaxInteractionDegree', options.MaxInteractionDegree);
m_std.fit(X, Y);

fprintf('\n--- Fitting PolyMARS (Multinomial) ---\n');
m_poly = OceanStats.Supervised.MARS.MARS('MaxTerms', options.MaxTerms, ...
    'MaxInteractionDegree', options.MaxInteractionDegree);
m_poly.ForwardPass = OceanStats.Selection.FeatureSelection.PolyMARSForwardSelector();
m_poly.fit(X, Y);

%% 3. Compute High-Res Prediction Grids
lim = max(abs(X(:))) + 1;
[G1, G2] = meshgrid(linspace(-lim, lim, 150)); % Higher Res for Smooth Boundaries
Xg = [G1(:), G2(:)];

% 1. Raw Probability Grid for Intelligent Projections
P_poly_grid = m_poly.predictProba(Xg); % (N_grid x 3)

% Proba for Metrics
P_std_train  = m_std.predictProba(X);
P_poly_train = m_poly.predictProba(X);

% ROC & Calibration (Focusing on the winding Class 1)
[fpr_std, tpr_std, ~, auc_std]    = OceanStats.Metrics.ROC.compute(Y, P_std_train, 1);
[fpr_poly, tpr_poly, ~, auc_poly] = OceanStats.Metrics.ROC.compute(Y, P_poly_train, 1);

[bins_std, freq_std, ece_std]    = OceanStats.Metrics.Calibration.compute(Y==1, P_std_train(:,1));
[bins_poly, freq_poly, ece_poly] = OceanStats.Metrics.Calibration.compute(Y==1, P_poly_train(:,1));

%% 4. Visualization: 2x2 Analytical Suite
% We subsample for plotting to keep the "neon" maps clean
idx_plot = randperm(N, floor(N/3)); % Show only 33% of points
X_plot = X(idx_plot, :);
Y_plot = Y(idx_plot, :);

fig = viz.createFigure('MARS vs PolyMARS: Hard Spiral Challenge');
set(fig, 'Position', [50 50 1200 900]);

% --- Row 1, Col 1: 2D Planar Projection (Object Strategy) ---
ax1 = subplot(2, 2, 1);
data2D = struct('RawX', X_plot(:,1), 'RawY', X_plot(:,2), 'RawC', Y_plot, ...
               'GridX1', G1, 'GridX2', G2, 'ProbMap', P_poly_grid); % RAW TENSOR
% Pluggable Action: OceanViz.Projections.Planar()
viz.plot(ax1, "Surface", data2D, ...
    'Projection', OceanViz.Projections.Planar(), ... 
    'Title', '2D Strategy: Planar', ...
    'XLabel', 'F1', 'YLabel', 'F2');

% --- Row 1, Col 2: 3D Manifold - Probability Mode (Smooth) ---
ax2 = subplot(2, 2, 2);
% Pluggable Action: Manifold with ZMode = 'Probability' (Auto-lifts tensor)
viz.plot(ax2, "Surface", data2D, ...
    'Projection', OceanViz.Projections.Manifold('ZMode', 'Probability'), ...
    'Title', '3D Strategy: Smooth Manifold', ...
    'XLabel', 'F1', 'YLabel', 'F2');

% --- Row 2, Col 1: 3D Manifold - Label Mode (Staircase) ---
ax3 = subplot(2, 2, 3);
% Pluggable Action: Manifold with ZMode = 'Label' (Auto-argmax tensor)
viz.plot(ax3, "Surface", data2D, ...
    'Projection', OceanViz.Projections.Manifold('ZMode', 'Label'), ...
    'Title', '3D Strategy: Label Manifold', ...
    'XLabel', 'F1', 'YLabel', 'F2');

% --- Row 2, Col 2: Calibration Comparison ---
ax4 = subplot(2, 2, 4);
cal_std = struct('bins', bins_std, 'frequencies', freq_std, 'modelName', 'Standard', 'color', viz.Colors.Secondary);
cal_poly = struct('bins', bins_poly, 'frequencies', freq_poly, 'modelName', 'PolyMARS', 'color', viz.Colors.Boundary);

viz.plot(ax4, "Calibration", cal_std, 'Title', 'Calibration Suite');
viz.plot(ax4, "Calibration", cal_poly, 'Title', 'Calibration Suite');

fprintf('\nObject-Oriented Projection Refactor Complete.\n');