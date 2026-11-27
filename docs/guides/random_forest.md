# Random Forest

## Overview

Random Forest (Breiman, 2001) is a parallel ensemble of randomized decision trees that reduces variance without increasing bias. It extends Bagging with one extra trick: at each split candidate, only `mtry` randomly chosen features are considered. This decorrelates the trees so their average converges faster and to a lower error floor.

**Package location:** `+OceanStats/+Ensemble/`

**Supported tasks:** Binary and multi-class classification, regression.

**Key design notes:**
- Parallelized with `parfor` when the Parallel Computing Toolbox is available **and** `NumTrees >= MinTreesForParallel` (default 20). Below that threshold, parallel overhead dominates.
- The random feature injection is done via `RandomFeatureCandidateGenerator` in `+Supervised/+Tree/`, a new implementation of `ICandidateGenerator`. No existing classes were modified (OCP).
- OOB error and feature importance are computed as free by-products of training (no extra data needed).

---

## Usage

### Classification

```matlab
rf = OceanStats.Ensemble.RandomForest( ...
    'NumTrees',   100, ...
    'Criterion',  'gini', ...    % or 'entropy', 'misclassification'
    'ComputeOOB', true, ...
    'Seed',       42);

rf.fit(X_train, Y_train);

Y_pred = rf.predict(X_test);        % hard labels
P      = rf.predictProba(X_test);   % soft class probabilities (N x K)

fprintf('OOB Error: %.4f\n', rf.oobError());
rf.printSummary();
```

### Regression

```matlab
rf = OceanStats.Ensemble.RandomForest( ...
    'NumTrees',   200, ...
    'Criterion',  'regression', ...
    'Mtry',       0, ...             % auto: floor(p/3)
    'MaxDepth',   Inf, ...           % fully grown (RF default)
    'ComputeOOB', true, ...
    'Seed',       1);

rf.fit(X_train, Y_train);
Y_pred = rf.predict(X_test);
fprintf('OOB MSE: %.6f\n', rf.oobError());
```

### Feature Importance

```matlab
rf = OceanStats.Ensemble.RandomForest( ...
    'NumTrees',          100, ...
    'ComputeImportance', true, ...
    'ImportanceMethod',  'MDA', ...  % or 'MDI'
    'Seed',              42);
rf.fit(X, Y);

scores = rf.featureImportance();     % (1 x p) vector
[~, ord] = sort(scores, 'descend');
fprintf('Most important feature: %d\n', ord(1));
```

---

## Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NumTrees` | double | 100 | Number of trees |
| `Mtry` | double | 0 (auto) | Features per split node. Auto: floor(sqrt(p)) for classification, floor(p/3) for regression |
| `MaxDepth` | double | Inf | Max tree depth (Inf = fully grown, RF default) |
| `MinLeafSize` | double | 1 | Min samples per leaf |
| `Criterion` | string | `'gini'` | Split criterion. Classification: `'gini'`, `'entropy'`, `'misclassification'`. Regression: `'regression'` |
| `ComputeOOB` | logical | `true` | Compute OOB error at fit-time |
| `ComputeImportance` | logical | `true` | Compute feature importance at fit-time |
| `ImportanceMethod` | string | `'MDA'` | `'MDA'` (permutation, more reliable) or `'MDI'` (Gini/RSS, faster) |
| `Seed` | double | 0 | RNG seed (0 = no seed) |
| `MinTreesForParallel` | double | 20 | Minimum NumTrees to justify `parfor` overhead |
| `Verbose` | logical | `false` | Print progress during training |

---

## Using with the Selection Infrastructure

```matlab
% Cross-validation
strat = OceanStats.Selection.Strategies.RandomForestStrategy( ...
    'NumTrees', 100, 'Criterion', 'gini');

cv    = OceanStats.Selection.CrossValidation.CrossValidator( ...
    strat, OceanStats.Selection.CrossValidation.KFold(5));
[err, ~] = cv.evaluate(X, Y);

% Bias-Variance audit
audit = OceanStats.Selection.Auditors.BiasVarianceAuditor(strat);
audit.run(X, Y);
```

---

## Importance Methods

### MDA — Mean Decrease Accuracy (permutation importance)

For each tree `b` and each feature `j`:
1. Predict OOB samples → baseline error `e_b`.
2. Permute column `j` in OOB data → permuted error `e_bj`.
3. `delta_bj = e_bj - e_b` (positive = important).

`MDA(j) = mean over b of delta_bj`

- Reliable for correlated features.
- Slightly slower (requires one prediction pass per feature per tree on OOB data).
- Requires `ComputeOOB = true` (enabled by default).

### MDI — Mean Decrease Impurity

Sums weighted impurity reductions across all nodes in all trees for each feature. Faster, but can over-rank high-cardinality features. Requires `Split.ImpurityDecrease` to be stored in tree nodes (will return zero vector if not available).

---

## Extremely Randomized Trees (Extra-Trees)

The **Extra-Trees** variant (Geurts et al., 2006) adds a second layer of randomization:
- No bootstrap (full dataset per tree by default `UseBootstrap=false`)
- Split thresholds are **random** (not optimal) — one value drawn uniformly from the feature range

```matlab
et = OceanStats.Ensemble.ExtraTreesForest( ...
    'NumTrees', 100, ...
    'NumCandidates', 1, ...    % K in Geurts (2006)
    'Criterion',  'gini', ...
    'Seed', 42);
et.fit(X_train, Y_train);
Y_pred = et.predict(X_test);
et.printSummary();
```

With bootstrap and OOB:
```matlab
et = OceanStats.Ensemble.ExtraTreesForest( ...
    'NumTrees', 100, 'UseBootstrap', true, ...
    'ComputeOOB', true, 'ImportanceMethod', 'MDA');
et.fit(X, Y);
fprintf('OOB: %.4f\n', et.oobError());
```

| | Random Forest | Extra-Trees |
|---|---|---|
| Bootstrap | Yes | No (default) |
| Threshold selection | Optimal (exhaustive midpoints) | Random draw |
| OOB error | Yes | Only with `UseBootstrap=true` |
| MDA importance | Yes | Only with `UseBootstrap=true` |
| Default importance | MDA | MDI |
| Speed per tree | O(N log N) | O(mtry) |

---

## Proximity Matrix

Two samples are *proximate* if they land in the same leaf across many trees.

```matlab
% Compute N x N proximity
P   = OceanStats.Ensemble.ProximityMatrix.compute(rf, X_train);

% Outlier detection (high score = rarely co-leaf with others)
out = OceanStats.Ensemble.ProximityMatrix.outlierScores(P);
[~, top_outliers] = sort(out, 'descend');

% MDS scatter in 2D
[coords, stress] = OceanStats.Ensemble.ProximityMatrix.mdsCoords(P, 2);
scatter(coords(:,1), coords(:,2));

% Sorted heatmap
OceanStats.Ensemble.ProximityMatrix.visualize(P, gca(), Y);
```

> **Note:** For N > 1000, a large-N warning is issued. Pass a subsample of X to `compute()` to control cost.

---

## Architecture

```
+OceanStats/
  +Ensemble/
    RandomForest.m               ← Orchestrator (handle)
    ExtraTreesForest.m           ← ExtraTrees orchestrator (handle)
    OOBTracker.m                 ← Vote accumulator (shared)
    ImportanceCalculator.m       ← Static MDA / MDI (shared)
    ProximityMatrix.m            ← Static co-leaf proximity utilities
  +Supervised/
    +Tree/
      RandomFeatureCandidateGenerator.m  ← ICandidateGenerator: RF subspace
      ExtraTreesCandidateGenerator.m     ← ICandidateGenerator: random thresholds
  +Selection/+Strategies/
    RandomForestStrategy.m       ← IModelStrategy adapter for RF
    ExtraTreesStrategy.m         ← IModelStrategy adapter for ExtraTrees
```

---

## See Also

- `OceanStats.Supervised.Tree.DecisionTree` — Base tree
- `OceanStats.Ensemble.Bagging` — Precursor (no feature randomization)
- `OceanStats.Boosting.BoostingEnsemble` — Sequential alternative
- ESL §15 (Hastie, Tibshirani, Friedman) — Theory reference
- Geurts et al. (2006). Extremely Randomized Trees. *Machine Learning*, 63(1).
- `Experiments/Scripts/Supervised/exp_random_forest_esl.m` — RF ESL §15 experiment
- `Experiments/Scripts/Supervised/exp_extra_trees_comparison.m` — RF vs ExtraTrees comparison
- `docs/recipes/random_forest_recipes.md` — Copy-paste recipes
