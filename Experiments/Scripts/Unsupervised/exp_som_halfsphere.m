% EXP_SOM_HALFSPHERE  Self-Organizing Map on a 3D half-sphere dataset.
%
%   DATASET:        Synthetic, N=90, p=3 (surface of half unit sphere)
%                   Three clusters: Red (0,1,0), Green (0,0,1), Blue (1,0,0)
%                   Red sigma=0.1 (tight), Green/Blue sigma=0.5 (wide)
%   MODEL:          Self-Organizing Map (Kohonen, 1982)
%                   5x5 grid = 25 prototypes
%                   Gaussian neighborhood kernel
%   ASSUMPTIONS:    - Points generated from isotropic Gaussian then projected
%                     onto unit sphere and clipped to z >= 0 (half-sphere).
%                   - Sigma values from ESL Exercise 14.5.
%                   - SOM hyperparameters match ESL Section 14.4 exactly:
%                     R=2, 40 passes, linear decay of alpha and r.
%   REPRODUCIBILITY: rng(42, 'twister') at top of script.
%
%   Ref: Hastie et al. (2009), ESL Section 14.4, Figures 14.15-14.16.

clc; clear; rng(42, 'twister');

%% === 1. Data Generation ===
% Generate 90 data points on the surface of a half-sphere (z >= 0)
% Three clusters centered near (0,1,0), (0,0,1), (1,0,0)

N_per_class = 30;
centers = [0, 1, 0;    % Red cluster center
           0, 0, 1;    % Green cluster center
           1, 0, 0];   % Blue cluster center
sigmas  = [0.1;        % Red:   tight cluster (ESL Ex 14.5)
           0.5;        % Green: wide cluster
           0.5];       % Blue:  wide cluster

K_classes = 3;
X = zeros(N_per_class * K_classes, 3);
Y = zeros(N_per_class * K_classes, 1);

for k = 1:K_classes
    startIdx = (k - 1) * N_per_class + 1;
    endIdx   = k * N_per_class;
    
    % Sample from N(center, sigma^2 * I_3)
    raw = centers(k, :) + sigmas(k) * randn(N_per_class, 3);
    
    % Project onto unit sphere: x <- x / ||x||
    norms = sqrt(sum(raw.^2, 2));
    raw = raw ./ norms;
    
    % Half-sphere constraint: z >= 0
    % If z < 0, flip the z component
    raw(raw(:,3) < 0, 3) = -raw(raw(:,3) < 0, 3);
    
    X(startIdx:endIdx, :) = raw;
    Y(startIdx:endIdx) = k;
end

N = size(X, 1);
p = size(X, 2);

fprintf('Generated %d points in %d classes on a half-sphere.\n', N, K_classes);
fprintf('  Red   (class 1): center=(0,1,0), sigma=%.1f (tight)\n', sigmas(1));
fprintf('  Green (class 2): center=(0,0,1), sigma=%.1f (wide)\n', sigmas(2));
fprintf('  Blue  (class 3): center=(1,0,0), sigma=%.1f (wide)\n', sigmas(3));

%% === 2. Feature Engineering ===
% None required — raw 3D coordinates on the half-sphere surface.

%% === 3. Train / Test Split ===
% N/A — unsupervised learning experiment, full dataset used for SOM training.

%% === 4. Model Configuration ===
% SOM hyperparameters matching ESL Section 14.4:
%   - 5 x 5 grid = 25 prototypes
%   - Initial neighborhood radius R = 2
%     (about 1/3 of prototypes fall within each neighborhood)
%   - 40 passes through 90 observations = 3600 total iterations
%   - alpha and r decay linearly over all 3600 iterations

som = OceanStats.Unsupervised.SOM( ...
    'GridSize',    [5, 5], ...
    'InitRadius',  2, ...
    'FinalRadius', 0.1, ...
    'InitLR',      0.5, ...
    'FinalLR',     0.01, ...
    'NumPasses',   40);

%% === 5. Training ===
% Capture initial assignment BEFORE training for the "before/after" comparison.

% --- Snapshot initial state ---
% We need initial weights for the "initial configuration" panel (Figure 14.16 left).
% To do this cleanly: create a second SOM with same init, fit with 0 passes... 
% OR: manually do a predict with the initial weights.
% Simplest: fit with TrackHistory=true and use History(1).

labels_final = som.fit(X, 'TrackHistory', true, 'Verbose', true);

% Initial assignment: predict using initial weights
somInit = OceanStats.Unsupervised.SOM('GridSize', [5, 5]);
somInit.Weights = som.History(1).Weights;  % copy initial weights
labels_init = somInit.predict(X);

%% === 6. Evaluation ===
% Compute metrics from scratch — never hard-code expected values.

% Mean Quantization Error (MQE): average distance to BMU
mqe = som.meanQuantizationError(X);

% Topographic Error (TE): fraction of samples where 1st & 2nd BMU are not adjacent
te = som.topographicError(X);

% U-matrix for cluster boundary visualization
U = som.umatrix();

fprintf('\n--- SOM Evaluation ---\n');
fprintf('  MQE (Mean Quantization Error) : %.4f\n', mqe);
fprintf('  Topographic Error             : %.4f (%.1f%%)\n', te, te * 100);

%% === 7. Visualization ===
viz = OceanViz.Core('Theme', 'Dark');

% -----------------------------------------------------------------------
% Figure 1: 3D Half-Sphere Data (replicates ESL Figure 14.15)
% -----------------------------------------------------------------------
fig1 = viz.createFigure('Half-Sphere Data');
set(fig1, 'Position', [100, 100, 800, 650]);
ax1 = axes(fig1);
hold(ax1, 'on');
viz.styleAxes(ax1);

palette = viz.getColorSequence(K_classes);
classNames = {'Red (tight)', 'Green (wide)', 'Blue (wide)'};

% Draw the half-sphere surface as translucent mesh
[sx, sy, sz] = sphere(40);
sz(sz < 0) = NaN;  % keep only upper hemisphere
surf(ax1, sx, sy, sz, ...
    'FaceColor', [0.9, 0.85, 0.3], ...
    'FaceAlpha', 0.15, ...
    'EdgeColor', [0.4, 0.4, 0.2], ...
    'EdgeAlpha', 0.15, ...
    'HandleVisibility', 'off');

% Plot data points by class
for k = 1:K_classes
    mask = (Y == k);
    scatter3(ax1, X(mask, 1), X(mask, 2), X(mask, 3), 50, ...
        'MarkerFaceColor', palette(k, :), ...
        'MarkerEdgeColor', palette(k, :) * 0.6, ...
        'MarkerFaceAlpha', 0.85, ...
        'LineWidth', 0.5, ...
        'DisplayName', classNames{k});
end

% Mark cluster centers
scatter3(ax1, centers(:,1), centers(:,2), centers(:,3), 120, ...
    'Marker', 'p', ...
    'MarkerFaceColor', viz.Colors.Boundary, ...
    'MarkerEdgeColor', [0 0 0], ...
    'LineWidth', 1.5, ...
    'DisplayName', 'True centers');

title(ax1, 'Half-Sphere Data — 3 Clusters (ESL Fig 14.15)', ...
    'Color', viz.Colors.Text, 'FontSize', 13, 'FontWeight', 'bold');
xlabel(ax1, 'X_1', 'Color', viz.Colors.Text);
ylabel(ax1, 'X_2', 'Color', viz.Colors.Text);
zlabel(ax1, 'X_3', 'Color', viz.Colors.Text);
set(ax1, 'ZColor', viz.Colors.Text);
view(ax1, -35, 25);
legend(ax1, 'Location', 'best', ...
    'FontSize', 8, ...
    'EdgeColor', [0.3 0.3 0.3], ...
    'TextColor', viz.Colors.Text, ...
    'Color', viz.Colors.Background);
hold(ax1, 'off');

% -----------------------------------------------------------------------
% Figure 2: SOM Grid — Initial vs Final (replicates ESL Figure 14.16)
% -----------------------------------------------------------------------
fig2 = viz.createFigure('SOM Grid — Initial vs Final');
set(fig2, 'Position', [100, 100, 1100, 500]);

% Panel A: Initial configuration
ax2a = subplot(1, 2, 1, 'Parent', fig2);
dataInit = struct();
dataInit.GridSize    = som.GridSize;
dataInit.Labels      = labels_init;
dataInit.ClassLabels = Y;
dataInit.NumClasses  = K_classes;
viz.plot(ax2a, 'SOMGrid', dataInit, 'Title', 'Initial Configuration');

% Panel B: Final configuration
ax2b = subplot(1, 2, 2, 'Parent', fig2);
dataFinal = struct();
dataFinal.GridSize    = som.GridSize;
dataFinal.Labels      = labels_final;
dataFinal.ClassLabels = Y;
dataFinal.NumClasses  = K_classes;
viz.plot(ax2b, 'SOMGrid', dataFinal, 'Title', 'Final Configuration');

sgtitle(fig2, 'Self-Organizing Map — 5x5 Grid (ESL Fig 14.16)', ...
    'Color', viz.Colors.Text, 'FontSize', 14, 'FontWeight', 'bold');

% -----------------------------------------------------------------------
% Figure 3: U-Matrix (topographic distance heatmap)
% -----------------------------------------------------------------------
fig3 = viz.createFigure('SOM U-Matrix');
set(fig3, 'Position', [100, 100, 600, 550]);
ax3 = axes(fig3);
viz.styleAxes(ax3);
hold(ax3, 'on');

imagesc(ax3, U);
colormap(ax3, hot);
cb = colorbar(ax3);
cb.Color = viz.Colors.Text;
cb.Label.String = 'Mean Neighbor Distance';
cb.Label.Color = viz.Colors.Text;

axis(ax3, 'equal', 'tight');
set(ax3, 'YDir', 'normal');
set(ax3, 'XTick', 1:som.GridSize(2), 'YTick', 1:som.GridSize(1));
title(ax3, 'U-Matrix — Cluster Boundaries', ...
    'Color', viz.Colors.Text, 'FontSize', 13, 'FontWeight', 'bold');
xlabel(ax3, 'Grid Column', 'Color', viz.Colors.Text);
ylabel(ax3, 'Grid Row', 'Color', viz.Colors.Text);
hold(ax3, 'off');

% -----------------------------------------------------------------------
% Figure 4: Animated 3D Wiremesh (replicates ESL Figure 14.17 over time)
%   Shows how the SOM lattice unfolds onto the half-sphere across passes.
%   Prototypes are purple nodes; grid edges are drawn as lines connecting
%   horizontally and vertically adjacent prototypes in 3D data space.
% -----------------------------------------------------------------------
fig4 = viz.createFigure('SOM Wiremesh Animation');
set(fig4, 'Position', [100, 100, 900, 700]);
ax4 = axes(fig4);
hold(ax4, 'on');
viz.styleAxes(ax4);

% Background: half-sphere surface
[sx, sy, sz] = sphere(40);
sz(sz < 0) = NaN;
surf(ax4, sx, sy, sz, ...
    'FaceColor', [0.9, 0.85, 0.3], ...
    'FaceAlpha', 0.10, ...
    'EdgeColor', [0.4, 0.4, 0.2], ...
    'EdgeAlpha', 0.08, ...
    'HandleVisibility', 'off');

% Background: data points (jittered by class color)
for k = 1:K_classes
    mask = (Y == k);
    scatter3(ax4, X(mask,1), X(mask,2), X(mask,3), 30, ...
        'MarkerFaceColor', palette(k,:), ...
        'MarkerEdgeColor', 'none', ...
        'MarkerFaceAlpha', 0.6, ...
        'HandleVisibility', 'off');
end

% Prepare grid adjacency edges for the Q1 x Q2 lattice
Q1 = som.GridSize(1);
Q2 = som.GridSize(2);
edges = [];  % each row = [idx1, idx2]
for i = 1:Q1
    for j = 1:Q2
        idx = (j - 1) * Q1 + i;  % linear index matching ndgrid(1:Q1, 1:Q2)
        % Horizontal neighbor: same row, next column
        if j < Q2
            idx_right = j * Q1 + i;
            edges = [edges; idx, idx_right]; %#ok<AGROW>
        end
        % Vertical neighbor: next row, same column
        if i < Q1
            idx_down = (j - 1) * Q1 + (i + 1);
            edges = [edges; idx, idx_down]; %#ok<AGROW>
        end
    end
end
nEdges = size(edges, 1);

% Initialize line and scatter handles for animation
hEdges = gobjects(nEdges, 1);
W0 = som.History(1).Weights;
for e = 1:nEdges
    i1 = edges(e, 1);
    i2 = edges(e, 2);
    hEdges(e) = plot3(ax4, ...
        [W0(i1,1), W0(i2,1)], ...
        [W0(i1,2), W0(i2,2)], ...
        [W0(i1,3), W0(i2,3)], ...
        '-', 'Color', [0.65, 0.65, 0.65, 0.85], 'LineWidth', 1.5, ...
        'HandleVisibility', 'off');
end

% Prototype node markers (purple, matching textbook)
protoColor = [0.55, 0.20, 0.82];
hNodes = scatter3(ax4, W0(:,1), W0(:,2), W0(:,3), 50, ...
    'MarkerFaceColor', protoColor, ...
    'MarkerEdgeColor', protoColor * 0.6, ...
    'MarkerFaceAlpha', 0.9, ...
    'LineWidth', 0.8, ...
    'HandleVisibility', 'off');

% Title handle for pass counter
hTitle = title(ax4, 'SOM Wiremesh — Pass 0 (Initial)', ...
    'Color', viz.Colors.Text, 'FontSize', 13, 'FontWeight', 'bold');
xlabel(ax4, 'X_1', 'Color', viz.Colors.Text);
ylabel(ax4, 'X_2', 'Color', viz.Colors.Text);
zlabel(ax4, 'X_3', 'Color', viz.Colors.Text);
set(ax4, 'ZColor', viz.Colors.Text);
view(ax4, -35, 25);
axis(ax4, 'equal');
xlim(ax4, [-1.3, 1.3]);
ylim(ax4, [-1.3, 1.3]);
zlim(ax4, [-0.3, 1.5]);

drawnow;
pause(1.0);

% --- Animate through passes ---
nHist = numel(som.History);
for frame = 2:nHist
    W = som.History(frame).Weights;
    
    % Update edge positions
    for e = 1:nEdges
        i1 = edges(e, 1);
        i2 = edges(e, 2);
        set(hEdges(e), ...
            'XData', [W(i1,1), W(i2,1)], ...
            'YData', [W(i1,2), W(i2,2)], ...
            'ZData', [W(i1,3), W(i2,3)]);
    end
    
    % Update node positions
    set(hNodes, 'XData', W(:,1), 'YData', W(:,2), 'ZData', W(:,3));
    
    % Update title with pass number
    set(hTitle, 'String', sprintf('SOM Wiremesh — Pass %d / %d', ...
        frame - 1, nHist - 1));
    
    drawnow;
    pause(0.15);  % ~150ms per frame for smooth animation
end

% Final title
set(hTitle, 'String', sprintf('SOM Wiremesh — Final (Pass %d) (ESL Fig 14.17)', ...
    nHist - 1));
hold(ax4, 'off');

% -----------------------------------------------------------------------
% Figure 5: Reconstruction Error vs Iteration (replicates ESL Fig 14.18)
%   Shows sum of squared distances to BMU at each pass.
%   Horizontal line = K-Means baseline (25 centroids) for comparison.
% -----------------------------------------------------------------------
% K-Means baseline: same K=25, for comparison
km = OceanStats.Unsupervised.KMeans();
km.fit(X, prod(som.GridSize));
km_err = 0;
for kk = 1:prod(som.GridSize)
    if any(km.Labels == kk)
        km_err = km_err + sum(sum((X(km.Labels == kk, :) - km.Centroids(kk, :)).^2));
    end
end

fig5 = viz.createFigure('Reconstruction Error');
set(fig5, 'Position', [100, 100, 750, 500]);
ax5 = axes(fig5);
viz.styleAxes(ax5);
hold(ax5, 'on');

% Iteration axis: pass 0 = iter 0, pass k = iter k*N
iterAxis = (0:(numel(som.ErrorHistory)-1)) * N;

plot(ax5, iterAxis, som.ErrorHistory, '-o', ...
    'Color', viz.Colors.Secondary, ...
    'MarkerFaceColor', viz.Colors.Secondary, ...
    'MarkerSize', 5, ...
    'LineWidth', 1.5, ...
    'DisplayName', 'SOM Reconstruction Error');

% K-Means baseline horizontal line
yline(ax5, km_err, '--', ...
    'Color', viz.Colors.Primary, ...
    'LineWidth', 1.8, ...
    'Label', sprintf('K-Means (K=%d)', prod(som.GridSize)), ...
    'LabelColor', viz.Colors.Primary, ...
    'FontSize', 9, ...
    'HandleVisibility', 'off');
% Legend proxy for K-Means line
plot(ax5, NaN, NaN, '--', 'Color', viz.Colors.Primary, ...
    'LineWidth', 1.8, ...
    'DisplayName', sprintf('K-Means baseline (K=%d)', prod(som.GridSize)));

title(ax5, 'Reconstruction Error vs Iteration (ESL Fig 14.18)', ...
    'Color', viz.Colors.Text, 'FontSize', 13, 'FontWeight', 'bold');
xlabel(ax5, 'Iteration', 'Color', viz.Colors.Text);
ylabel(ax5, 'Reconstruction Error', 'Color', viz.Colors.Text);
legend(ax5, 'Location', 'northeast', ...
    'FontSize', 9, ...
    'EdgeColor', [0.3 0.3 0.3], ...
    'TextColor', viz.Colors.Text, ...
    'Color', viz.Colors.Background);
hold(ax5, 'off');

%% === 8. Summary Printout ===
fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Dataset       : Half-Sphere (N=%d, p=%d, K=%d classes)\n', N, p, K_classes);
fprintf('  Model         : Self-Organizing Map (Kohonen, 1982)\n');
fprintf('  Grid          : %d x %d = %d prototypes\n', som.GridSize(1), som.GridSize(2), prod(som.GridSize));
fprintf('  Training      : %d passes x %d samples = %d iterations\n', 40, N, 40*N);
fprintf('  Init Radius   : R = %.1f\n', 2.0);
fprintf('  Final Radius  : R = %.1f\n', 0.1);
fprintf('  Init LR       : alpha = %.2f\n', 0.5);
fprintf('  Final LR      : alpha = %.2f\n', 0.01);
fprintf('  MQE           : %.4f\n', mqe);
fprintf('  Topographic Err: %.4f (%.1f%%)\n', te, te * 100);
fprintf('  Recon Error   : %.4f (SOM), %.4f (K-Means baseline)\n', ...
    som.ErrorHistory(end), km_err);
fprintf('  Cluster spread : Red sigma=0.1, Green/Blue sigma=0.5\n');
fprintf('==========================\n');

