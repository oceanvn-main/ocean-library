---
description: The official 11-step Ocean Library Data Science Pipeline for experiments (MLOps Medallion Architecture)
---

# Data Science Pipeline Workflow

Use this workflow for explicitly designing and structuring any Machine Learning or Data Science experiment in the Ocean Library. This ensures data provenance, strict anti-leakage train/test splitting, caching, and model evaluation follow a consistent, production-grade MLOps architectural pattern.

Trigger with `/ml-pipeline` or follow the steps below when building new `exp_*.m` scripts.

---

## 🏗️ The 11-Step Pipeline Architecture

Every complete experiment should explicitly annotate or orchestrate these layers (some may be gracefully skipped depending on config, but the structure and `[SKIP]` log must remain):

### Step 1: Parquet Conversion & Ingestion
- Read the raw files (CSV, JSON, MAT) and immediately convert them into structured MATLAB `table` objects.
- Save them into the DataLake natively as `.parquet` files using `lake.put(..., 'asParquet', true)`.
- **Every experiment must proceed by loading from these Parquet files.** This entirely eliminates row-wise vs column-wise mismatch bugs and ensures strict type safety.

### Step 2: Train / Test Split (MANDATORY — Anti-Leakage)
- **IMMEDIATELY** split the data before it enters the database or feature stores.
- Use `OceanData.DataArch.DataSplitter` so that test data is physically decoupled early in the pipeline.
  ```matlab
  splitter = OceanData.DataArch.DataSplitter('Strategy', 'holdout', 'TestRatio', 0.3);
  [idx_train, idx_test] = splitter.split(size(X_all, 1));
  X_train = X_all(idx_train, :);  y_train = y_all(idx_train);
  X_test  = X_all(idx_test, :);   y_test  = y_all(idx_test);
  ```

### Step 3: DataLake (DA-1) — Bronze / Raw Persistence
- *Role:* Immutable storage of raw data before any processing.
- *Action:* Store the Train and Test datasets into physically separate partitions as Parquet files.
  ```matlab
  lake.put('raw/dataset/train/X', trainTable, meta, 'asParquet', true);
  lake.put('raw/dataset/test/X',  testTable,  meta, 'asParquet', true);
  
  % Always proceed by reading the Parquet file into memory
  % T_train = parquetread('lake/raw/dataset/train/X.parquet');
  ```
- *Component:* `OceanData.DataArch.DataLake`

### Step 4: Dataset Versioning (Tier 2 Metadata)
- *Role:* Register the existence of the raw Parquet files in the metadata database so they can be easily queried and audited.
- *Action:* Insert a metadata record into the Postgres `datasets` table (or local disk fallback).
  ```matlab
  tracker = OceanData.DataArch.MetadataRegistry('rootDir', './meta/');
  tracker.registerDataset('dataset', 'v1', 'train', 'lake/raw/train.parquet', 12000);
  ```

### Step 5: FeatureStore (DA-5) — Persistent Engineered Features
- *Role:* Save expensive preprocessing (PCA, standardizations) to disk. Ensures consistency between training features and inference features.
- *Action:* Compute features using a stable key. **Derive all statistical bounds (mean, variance) STRICTLY from `X_train`, then apply them to `X_test`.**
  ```matlab
  feat_key = sprintf('%s_std_seed%d', dataset_name, rng_seed);
  if ~fs.exists(feat_key)
      [X_feat_tr, mu, sigma] = zscore(X_train);
      X_feat_te = (X_test - mu) ./ sigma; % Apply train stats to test!
      fs.put(feat_key, struct('X_tr', X_feat_tr, 'X_te', X_feat_te), struct());
  end
  ```
- *Component:* `OceanData.DataArch.FeatureStore`
- **Warning:** Never include test data during any preprocessing `fit()` step.

### Step 6: FeatureCache (DA-6) — In-Memory Features
- *Role:* Extremely fast retrieval of batched features for Epoch/Minibatch training.
- *Action:* Deposit features into the cache, then retrieve from cache in the training loop:
- *Component:* `OceanData.Adapters.FeatureCache` (backed by Redis/Memcached)

### Step 7: Model Selection (MANDATORY before final training)

All sub-steps in Step 7 operate **exclusively on training data**. The test set must never be touched here.

#### 7a — Feature Selection
- *Role:* Reduce dimensionality, remove irrelevant predictors, prevent overfitting.
- *Choose one strategy based on task:* Filter, Wrapper, or Embedded.
  > **Rule:** Feature selection fit must use **training data only**. Apply the fitted mask/transform to test data passively.

#### 7b — Hyperparameter Tuning + Cross-Validation
- *Role:* Estimate generalization error and select the best hyperparameters (Lambda, tree depth) without using the test set.
- *Action:* Use a **k-fold CV loop** over the training set. Use `OceanTools.crossvalIndices` for reproducible folds.
  > **Rule:** NEVER evaluate CV performance on `X_test`.

#### 7c — Model Comparison / Information Criteria
- *Role:* Select between qualitatively different models using AIC/BIC or McNemar tests.

---

### Step 8: Model Training & Evaluation (Final)
- *Role:* Train the **final model** using the best hyperparameters from Step 7b, on the **full training set**. Evaluate **once** on the held-out test set.
- *Action:* Use `IModelStrategy` with the selected config. 
  > **Golden rule:** The test set is used **exactly once** — here. If you tune using test error, your reported error is optimistic and invalid.

### Step 9: Experiment Tracking (Tier 3 Metadata)
- *Role:* Record the results permanently for comparison.
- *Action:* Log the run metrics, hyperparameter JSON, and artifact paths into the `MetadataRegistry`.
  ```matlab
  tracker.logRun('r_001', 'v1', 'Lasso', 'rmse', 0.12, '{"lambda": 0.01}');
  ```

### Step 10: The Data Warehouse (Gold Layer / OLAP)
- *Role:* Make the aggregated experiment results queryable for Business Intelligence (BI) and Dashboards. The warehouse NEVER touches feature matrices or model training. It only receives results and summaries after everything is done.
- *Action:* Push the high-level summary statistics into the `DataWarehouse`.
  ```matlab
  dw = OceanData.DataArch.DataWarehouse('rootDir', './datawarehouse/');
  dw.insertBatch('model_performance', perfTable);
  ```

### Step 11: Visualization
- *Role:* Communicate results clearly.
- *Action:* Plot **strictly via** `OceanViz.Core`.
  ```matlab
  viz = OceanViz.Core('Theme', 'Dark');
  fig = viz.createFigure('Results');
  viz.plot(ax, 'StrategyName', data);
  viz.styleAxes(ax);
  ```

---

## 📝 Implementation Template for `exp_*.m`

```matlab
% EXP_EXAMPLE  Demonstrates the MLOps 8-step data science pipeline.
% Author: [name]   Date: [date]
clc; clear; rng(42, 'twister');
RNG_SEED = 42;  DATASET = 'example';

%% === 1. Ingestion ===
% gen = OceanData.Generators.MyGenerator('N', 1000);
% [X_all, y_all] = gen.generate();

%% === 2. Train/Test Split (MANDATORY — Anti-Leakage) ===
% splitter = OceanData.DataArch.DataSplitter('Strategy','holdout','TestRatio',0.3);
% [idx_tr, idx_te] = splitter.split(size(X_all,1));
% X_train = X_all(idx_tr,:); y_train = y_all(idx_tr);
% X_test  = X_all(idx_te,:); y_test  = y_all(idx_te);

%% === 3 & 4. Bronze Layer (Parquet DataLake) & Tier 2 Metadata ===
try
    lake = OceanData.DataArch.DataLake('rootDir', './lake/');
    trainT = array2table([X_train, y_train]); testT = array2table([X_test, y_test]);
    lake.put(sprintf('raw/%s/train/X', DATASET), trainT, struct(), 'asParquet', true);
    lake.put(sprintf('raw/%s/test/X', DATASET),  testT,  struct(), 'asParquet', true);
    
    tracker = OceanData.DataArch.MetadataRegistry('rootDir', './meta/');
    tracker.registerDataset(DATASET, 'v1', 'train', 'lake/train.parquet', height(trainT));
    tracker.registerDataset(DATASET, 'v1', 'test',  'lake/test.parquet',  height(testT));
catch, fprintf('[SKIP] Bronze/Metadata tracking unavailable.\n'); end

%% === 5 & 6. Silver Layer (Feature Engineering & FeatureStore Parquet) ===
try
    [X_tr_std, mu, sig] = zscore(X_train);
    X_te_std = (X_test - mu) ./ sig;
    y_tr_std = y_train - mean(y_train); y_te_std = y_test - mean(y_train);
    
    fs = OceanData.DataArch.DataLake('rootDir', './fs/');
    
    % Store the parameters (mu, sig) securely in the metadata so inference pipelines can reuse them!
    meta = struct('mu', mu, 'sigma', sig);
    fs.put(sprintf('%s/train_feat', DATASET), array2table([X_tr_std, y_tr_std]), meta, 'asParquet', true);
    
    tracker.registerDataset(DATASET+"_feat", 'v1', 'train', 'fs/train_feat.parquet', size(X_tr_std,1));
catch, [X_tr_std, mu, sig] = zscore(X_train); y_tr_std = y_train - mean(y_train); end

%% === 7. Model Selection (training data only) ===
% ... [CV loops]

%% === 8. Final Model Training & Evaluation (test set: ONE time only) ===
% rmse = sqrt(mean((y_te_std - X_te_std * B_opt).^2));

%% === 9 & 10. Tier 3 Tracking & Gold Layer (Data Warehouse) ===
try
    tracker.logRun('r_01', 'v1', 'Lasso', 'rmse', rmse, '{"lambda": 0.01}');
    dw = OceanData.DataArch.DataWarehouse('rootDir', './datawarehouse/');
    dw.insertBatch('model_performance', table("r_01", "v1", "Lasso", "rmse", rmse, datetime('now')));
catch, end

%% === 11. Visualization ===
% ...
```
