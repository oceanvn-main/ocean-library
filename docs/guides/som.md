# Self-Organizing Map (SOM)

## Overview

The Self-Organizing Map (SOM), also known as a Kohonen map, is an unsupervised neural network that produces a low-dimensional (typically 2D) discretized representation of the input space. Unlike methods like PCA that preserve global structure, SOMs preserve *topological* properties — nearby prototypes on the grid map to nearby regions in data space.

**Key idea:** A rectangular grid of Q1 x Q2 prototypes is trained via competitive learning. Each training sample "pulls" the nearest prototype (BMU) and its grid-neighbors toward itself, with the influence decaying over both grid distance and training time.

**Location:** `+OceanStats/+Unsupervised/SOM.m`

## Usage

```matlab
% Basic usage
som = OceanStats.Unsupervised.SOM('GridSize', [5, 5], 'NumPasses', 40);
labels = som.fit(X);         % X is (N x p)
labels_new = som.predict(X_new);

% With custom hyperparameters
som = OceanStats.Unsupervised.SOM( ...
    'GridSize',    [10, 10], ...
    'InitRadius',  4, ...
    'FinalRadius', 0.5, ...
    'InitLR',      0.8, ...
    'FinalLR',     0.01, ...
    'NumPasses',   50);
labels = som.fit(X, 'Verbose', true);
```

### Constructor Options

| Option | Default | Description |
|---|---|---|
| `GridSize` | `[5, 5]` | Grid dimensions `[Q1, Q2]` |
| `InitRadius` | `2` | Starting Gaussian neighborhood radius |
| `FinalRadius` | `0.1` | End neighborhood radius |
| `InitLR` | `0.5` | Starting learning rate alpha |
| `FinalLR` | `0.01` | End learning rate |
| `NumPasses` | `40` | Number of full passes through data |

### Fit Options

| Option | Default | Description |
|---|---|---|
| `TrackHistory` | `false` | Record `.History(t).Weights` per pass |
| `InitWeights` | `[]` | Fixed initial prototype weights (M x p) |
| `Verbose` | `false` | Print MQE per pass |
| `Mode` | `'online'` | `'online'` (stochastic Kohonen) or `'batch'` (ESL Eq 14.48, weighted-mean, no learning rate) |

### Properties (after fit)

- `Weights` — (M x p) final prototype weight vectors
- `Labels` — (N x 1) BMU assignments for training data
- `GridSize` — [Q1, Q2]
- `History` — struct array of `.Weights` snapshots (if `TrackHistory=true`)
- `ErrorHistory` — (NumPasses+1 x 1) reconstruction error per pass (if `TrackHistory=true`)

### Methods

- `predict(X)` — assign each row to its BMU
- `getGridCoords()` — returns `[gx, gy]` integer grid positions
- `meanQuantizationError(X)` — average distance to BMU
- `topographicError(X)` — fraction of samples with non-adjacent 1st/2nd BMUs
- `umatrix()` — compute the U-matrix for cluster boundary visualization

## Examples

```matlab
% Train a 5x5 SOM on 3D half-sphere data
rng(42, 'twister');
som = OceanStats.Unsupervised.SOM('GridSize', [5, 5], 'NumPasses', 40);
labels = som.fit(X, 'TrackHistory', true, 'Verbose', true);

% Evaluate
mqe = som.meanQuantizationError(X);
te  = som.topographicError(X);
U   = som.umatrix();

% Visualize with OceanViz
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('SOM Grid');
ax  = axes(fig);
data = struct('GridSize', som.GridSize, 'Labels', labels, 'ClassLabels', Y);
viz.plot(ax, 'SOMGrid', data, 'Title', 'SOM Assignment');
```

## See Also

- `OceanStats.Unsupervised.KMeans` — Lloyd's algorithm
- `OceanStats.Supervised.Neighbors.LVQ1` — supervised competitive learning
- `OceanViz.Strategies.SOMGridStrategy` — SOM grid visualization
- ESL Section 14.4 (Hastie et al., 2009)
