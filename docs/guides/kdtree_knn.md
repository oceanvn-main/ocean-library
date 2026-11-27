# KD-Tree & KNN Classifier

## Overview

`OceanAlgo.Spatial.KDTree` is a **generic N-dimensional KD-Tree** that supports k-nearest-neighbour and radius range queries in **O(log N)** average time. It is the foundation for `OceanStats.Supervised.KNNClassifier`.

**Contrast with the existing QuadTree:**

| | `OceanMath.Geometry.QuadTree` | `OceanAlgo.Spatial.KDTree` |
|---|---|---|
| Dimensions | 2D fixed | Arbitrary d |
| k-NN (k > 1) | No | Yes |
| Distance metric | Euclidean only | euclidean / manhattan / chebyshev / custom |
| Dynamic inserts | No | No (immutable) |
| Primary use | TrackDesigner station picking | ML, KNN, density estimation |

---

## KDTree Usage

```matlab
% Build
kd = OceanAlgo.Spatial.KDTree(X);                              % default euclidean
kd = OceanAlgo.Spatial.KDTree(X, 'Metric', 'manhattan', 'LeafSize', 20);

% k-NN
[ids, dists] = kd.findKNearest(q, 5);    % 5 nearest, sorted ascending

% Radius search
[ids, dists] = kd.rangeQuery(q, 3.0);    % all within distance 3

% Diagnostics
kd.describe();
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Metric` | `'euclidean'` | `'euclidean'` \| `'manhattan'` \| `'chebyshev'` \| `@(a,b)->scalar` |
| `LeafSize` | 10 | Max points per leaf before splitting |
| `MaxDepth` | 64 | Recursion cap (safety) |

### Algorithm

1. **Build** — Recursively split on the dimension with the **largest range** (max-spread). Stop at `LeafSize` points per leaf. O(N log N) build time.
2. **k-NN** — Traverse nearest child first. Prune subtrees where `minBoxDist(AABB, q) >= kth_dist`. Returns results sorted ascending by distance.
3. **Range** — Prune subtrees where `minBoxDist(AABB, q) >= radius`.
4. **AABB distance** — Minimum distance from query to axis-aligned bounding box:  
   `delta_i = max(0, lo_i - q_i, q_i - hi_i)`, then L2/L1/Linf of delta.

---

## KNNClassifier Usage

### Classification

```matlab
knn = OceanStats.Supervised.KNNClassifier('K', 5);
knn.fit(X_train, Y_train);
Y_pred = knn.predict(X_test);
P      = knn.predictProba(X_test);   % M x K, rows sum to 1
knn.printSummary();
```

### Regression

```matlab
knn = OceanStats.Supervised.KNNClassifier('K', 5, 'Task', 'regression');
knn.fit(X_train, Y_train);
Y_pred = knn.predict(X_test);
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K` | 5 | Number of neighbours |
| `Metric` | `'euclidean'` | Passed to KDTree |
| `Weights` | `'uniform'` | `'uniform'` (majority vote) or `'distance'` (inv-dist weighted) |
| `Task` | `'auto'` | `'auto'` \| `'classification'` \| `'regression'` |
| `LeafSize` | 10 | KDTree leaf size |

**Task auto-detection:** if Y is integer-valued with ≤ 20 unique values → classification, otherwise regression.

### With Selection Infrastructure

```matlab
% Tune K with 5-fold CV
strat = OceanStats.Selection.Strategies.KNNStrategy('K', 5);
cv    = OceanStats.Selection.CrossValidation.CrossValidator( ...
    strat, OceanStats.Selection.CrossValidation.KFold(5));
err   = cv.evaluate(X, Y);

% Sweep K values
for K_i = 1:20
    strat = OceanStats.Selection.Strategies.KNNStrategy('K', K_i);
    cv_errs(K_i) = cv.evaluate(X, Y);
end
```

---

## Architecture

```
+OceanAlgo/
  +Spatial/
    KDTree.m               ← Generic N-d KD-Tree (NEW)

+OceanStats/
  +Supervised/
    KNNClassifier.m        ← KNN model using KDTree (NEW)
  +Selection/+Strategies/
    KNNStrategy.m          ← IModelStrategy adapter (NEW)
```

---

## See Also

- `OceanMath.Geometry.QuadTree` — 2D spatial index for TrackDesigner
- `OceanStats.Ensemble.RandomForest` — alternative ensemble classifier
- `Experiments/Scripts/Supervised/exp_knn_kdtree.m` — boundary & CV sweep
