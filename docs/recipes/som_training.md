# SOM Training Recipe

## Quick Start — Train and Visualize a SOM

```matlab
% 1. Create and train
som = OceanStats.Unsupervised.SOM('GridSize', [5, 5], 'NumPasses', 40);
labels = som.fit(X);

% 2. Evaluate quality
mqe = som.meanQuantizationError(X);   % lower is better
te  = som.topographicError(X);        % lower is better (< 0.5 is decent)
fprintf('MQE = %.4f, Topographic Error = %.4f\n', mqe, te);
```

## Visualize SOM Grid Assignments

```matlab
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('SOM Grid');
ax  = axes(fig);

data = struct();
data.GridSize    = som.GridSize;
data.Labels      = labels;       % BMU assignments
data.ClassLabels = Y;            % true class labels for coloring
viz.plot(ax, 'SOMGrid', data, 'Title', 'SOM Final');
```

## Visualize U-Matrix (Cluster Boundaries)

```matlab
fig2 = viz.createFigure('U-Matrix');
ax2  = axes(fig2);
viz.styleAxes(ax2);

U = som.umatrix();
imagesc(ax2, U);
colormap(ax2, hot);
colorbar(ax2);
set(ax2, 'YDir', 'normal');
title(ax2, 'U-Matrix', 'Color', viz.Colors.Text);
```

## Before/After Comparison (ESL Figure 14.16 style)

```matlab
% Train with history tracking
labels = som.fit(X, 'TrackHistory', true);

% Snapshot initial assignment
somInit = OceanStats.Unsupervised.SOM('GridSize', [5, 5]);
somInit.Weights = som.History(1).Weights;
labels_init = somInit.predict(X);

% Side-by-side panels
fig = viz.createFigure('Initial vs Final');
ax_left  = subplot(1, 2, 1);
ax_right = subplot(1, 2, 2);

dInit = struct('GridSize', som.GridSize, 'Labels', labels_init, 'ClassLabels', Y);
dFinal = struct('GridSize', som.GridSize, 'Labels', labels, 'ClassLabels', Y);
viz.plot(ax_left,  'SOMGrid', dInit,  'Title', 'Initial');
viz.plot(ax_right, 'SOMGrid', dFinal, 'Title', 'Final');
```

## Animated 3D Wiremesh (ESL Figure 14.17 style)

```matlab
% Train with history
labels = som.fit(X, 'TrackHistory', true);

% Build grid edge list
Q1 = som.GridSize(1); Q2 = som.GridSize(2);
edges = [];
for i = 1:Q1, for j = 1:Q2
    idx = (j-1)*Q1 + i;
    if j < Q2, edges = [edges; idx, j*Q1+i]; end
    if i < Q1, edges = [edges; idx, (j-1)*Q1+i+1]; end
end, end

% Draw initial wiremesh
fig = viz.createFigure('Wiremesh');
ax = axes(fig); hold(ax,'on'); viz.styleAxes(ax);
W = som.History(1).Weights;
hEdges = gobjects(size(edges,1),1);
for e = 1:size(edges,1)
    hEdges(e) = plot3(ax, W(edges(e,:),1), W(edges(e,:),2), W(edges(e,:),3), ...
        '-', 'Color', [0.65 0.65 0.65 0.85], 'LineWidth', 1.5);
end
hNodes = scatter3(ax, W(:,1), W(:,2), W(:,3), 50, 'filled', ...
    'MarkerFaceColor', [0.55 0.2 0.82]);
view(ax, -35, 25);

% Animate
for f = 2:numel(som.History)
    W = som.History(f).Weights;
    for e = 1:size(edges,1)
        set(hEdges(e), 'XData', W(edges(e,:),1), ...
            'YData', W(edges(e,:),2), 'ZData', W(edges(e,:),3));
    end
    set(hNodes, 'XData', W(:,1), 'YData', W(:,2), 'ZData', W(:,3));
    drawnow; pause(0.15);
end
```

## Reconstruction Error Curve (ESL Figure 14.18 style)

```matlab
% ErrorHistory is automatically computed when TrackHistory=true
iterAxis = (0:(numel(som.ErrorHistory)-1)) * size(X,1);
plot(iterAxis, som.ErrorHistory, '-o', 'LineWidth', 1.5);
xlabel('Iteration'); ylabel('Reconstruction Error');
```

## Tuning Tips

| Parameter | Effect |
|---|---|
| `GridSize` | Larger grids = finer resolution but slower training |
| `InitRadius` | Start large (~1/3 of grid width) for global ordering |
| `FinalRadius` | End small (~0.1) for local fine-tuning |
| `NumPasses` | 20–100 passes typically sufficient for small datasets |
| `InitLR` | 0.3–0.8 typical; larger = faster convergence |

