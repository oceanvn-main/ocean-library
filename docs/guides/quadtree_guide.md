# QuadTree Spatial Index

## Overview

`OceanMath.Geometry.QuadTree` is a 2D spatial index that recursively partitions a rectangular region into 4 quadrants (NW, NE, SW, SE). It enables **O(log N)** nearest-neighbor lookups and range queries, replacing brute-force O(N) scans through station arrays.

## When to Use

- **Station lookup**: Finding which station the user clicked near (TrackDesigner)
- **Range queries**: Finding all stations within a radius of a point
- **Bounding-box queries**: Finding all points inside a rectangular region
- **Pre-filtering**: Skipping expensive per-vertex collision checks for curves whose bounding boxes don't overlap with the search region

## Usage

### Construction (Bulk-Build)

```matlab
positions = rand(1000, 2) * 100;   % Nx2 matrix [X, Y]
ids       = (1:1000)';             % Nx1 vector of IDs
qt = OceanMath.Geometry.QuadTree(positions, ids);
```

The tree is built in O(N log N) time. Each leaf holds up to 8 points (configurable via `MAX_LEAF_SIZE`).

### Nearest-Neighbor Query

```matlab
[nearId, dist] = qt.findNearest([50, 50], 10);
% nearId = ID of closest point within radius 10
% nearId = 0 if no point found
```

### Range Query

```matlab
ids = qt.rangeQuery([50, 50], 20);
% ids = column vector of all point IDs within radius 20
```

### Box Query

```matlab
ids = qt.queryBox([10, 10, 40, 40]);
% ids = all points inside the axis-aligned bounding box
% Box format: [xMin, yMin, xMax, yMax]
```

### Diagnostics

```matlab
qt.describe();
% QuadTree:
%   Points:    1000
%   Leaves:    142
%   Max depth: 8
%   Bounds:    [-0.10, -0.10] to [100.10, 100.10]
```

## How It Works

1. **Build**: The constructor computes a tight bounding box around all points, then recursively splits it into 4 equal quadrants. When a quadrant contains ≤ 8 points, it becomes a leaf node storing the raw data.

2. **Search**: For `findNearest`, the algorithm visits child nodes ordered by distance (nearest first). If the closest possible point in a child's bounding box is already farther than the current best, that entire subtree is **pruned** — skipped entirely without checking any of its points.

3. **Rebuild**: The tree is immutable after construction. When stations change, `TrackDesigner.rebuildSpatialIndex_()` constructs a fresh tree. This is efficient because station mutations (add/delete) are rare compared to queries (every mouse click).

## Integration with TrackDesigner

The `TrackDesigner` holds a `SpatialIdx_` property that is automatically rebuilt when stations are added or deleted. The `InteractionHandler.findNearestStation_()` method uses this index for O(log N) lookups with an O(N) vectorized fallback.

## See Also

- `OceanViz.Designer.InteractionHandler` — Uses QuadTree for station click detection
- `OceanViz.Designer.TrackRenderer` — Uses AABB early-rejection for curve collision
- `OceanMath.Geometry.IParametricCurve` — Spline curves indexed by bounding boxes
