---
description: The official Ocean Library ML Production Architecture Pipeline Workflow
---

# ML Production Architecture Pipeline

This workflow defines the modern MLOps architecture for the Ocean Library. It strictly separates **Data Storage (Parquet)** from **Metadata/Experiment Tracking (Postgres)** to ensure high performance, reliable versioning, and clean data provenance.

Trigger with `/ml-production-pipeline` or follow these steps when building new `exp_*.m` scripts.

---

## 🏗️ Core Philosophy

1. **Tier 1 (The Data):** Large numerical feature matrices are stored as **Parquet files** on disk. Parquet is columnar, compressed, highly portable, and type-safe.
2. **Tier 2 (Data Metadata):** **PostgresRegistry** logs the *story* of the data (versions, train/test splits, file paths, row counts).
3. **Tier 3 (Experiment Metrics):** **PostgresRegistry** logs the individual model runs.
4. **Tier 4 (Gold OLAP):** **DataWarehouse** (Snowflake/BigQuery equivalent) stores purely aggregated, business-ready metrics for BI Dashboards.

> **Rule of Thumb:**
> - If it is a huge matrix a model trains on $\rightarrow$ **Parquet (DataLake/FeatureStore)**
> - If it is tracking "who did what and when" $\rightarrow$ **Postgres (MetadataRegistry)**
> - If an analyst asks "What is the average accuracy trend this month?" $\rightarrow$ **DataWarehouse (Gold Layer)**

---

## 🚀 The 8-Step Pipeline

### Step 1: Ingestion & Train/Test Split
- Load the raw dataset into memory as a table.
- **IMMEDIATELY** split the data into Train and Test sets to prevent data leakage.
- *Tool:* `OceanData.DataArch.DataSplitter`

### Step 2: DataLake (Bronze Layer)
- *Role:* Immutable storage of raw, unprocessed data.
- *Action:* Save the split datasets to disk as Parquet files.
  ```matlab
  parquetwrite('lake/raw/dataset_train.parquet', trainTable);
  parquetwrite('lake/raw/dataset_test.parquet', testTable);
  ```

### Step 3: Dataset Versioning (Postgres Metadata)
- *Role:* Register the existence of the raw Parquet files in the database so they can be easily queried and audited.
- *Action:* Insert a metadata record into the Postgres `datasets` table.
  ```matlab
  tracker = OceanData.Adapters.ExperimentTracker();
  tracker.logDataset('dataset', 'prostate', 'version', 'v1', 'split', 'train', 'path', 'lake/raw/train.parquet', 'n_rows', 12000);
  ```

### Step 4: Feature Engineering
- *Role:* Compute ML features (Standardization, PCA, Imputation).
- *Action:* Derive statistical bounds (mean, standard deviation) strictly from the training set, then safely apply them to the test set.

### Step 5: FeatureStore (Gold Layer)
- *Role:* Save expensive preprocessing results so they don't have to be recomputed.
- *Action:* Save the newly engineered features as Parquet files.
  ```matlab
  parquetwrite('fs/dataset_train_feat_v1.parquet', trainFeatTable);
  ```
- *Action:* Log this new engineered dataset version into Postgres using the Tracker.

### Step 6: Feature Cache (In-Memory)
- *Role:* Extremely fast data retrieval for epoch/minibatch training loops.
- *Action:* Load from the FeatureStore Parquet file into Redis/RAM via `FeatureCache`.

### Step 7: Model Training
- *Role:* Train the final model and evaluate it on the Test set *exactly once*.
- *Action:* Use an `IModelStrategy` to fit and predict.

### Step 8: Experiment Tracking (Metadata)
- *Role:* Record the results permanently for comparison.
- *Action:* Log the run metrics, hyperparameter JSON, and artifact paths into the `MetadataRegistry` (Postgres).
  ```matlab
  tracker.logRun('r_001', 'v1', 'Lasso', 'rmse', 0.12, '{"lambda": 0.01}');
  ```

### Step 9: The Data Warehouse (Gold Layer / OLAP)
- *Role:* Make the aggregated experiment results queryable for Business Intelligence (BI) and Dashboards.
- *Action:* Push the high-level summary statistics into the `DataWarehouse`.
  ```matlab
  dw = OceanData.DataArch.DataWarehouse();
  dw.insertBatch('model_performance', table("r_001", "v1", "Lasso", "rmse", 0.12, datetime('now'), ...
      'VariableNames', {'run_id', 'version', 'model', 'metric_name', 'metric_value', 'date'}));
      
  % An analyst later queries: "Average RMSE per model"
  % avg = dw.aggregate('model_performance', 'GroupBy', 'model', 'AggFunc', @mean, 'Target', 'metric_value');
  ```

---

## 📝 Implementation Template for `exp_*.m`

```matlab
% EXP_EXAMPLE  Demonstrates the Parquet/Postgres Metadata Pipeline.
clc; clear; rng(42, 'twister');

%% === 1. Ingestion & Split ===
% T_all = readtable('data.csv');
% [idx_tr, idx_te] = split(T_all);
% T_train = T_all(idx_tr, :);  T_test = T_all(idx_te, :);

%% === 2 & 3. DataLake (Parquet) + Versioning (Postgres) ===
lakeTrainPath = fullfile('lake', 'raw', 'train_v1.parquet');
lakeTestPath  = fullfile('lake', 'raw', 'test_v1.parquet');

parquetwrite(lakeTrainPath, T_train);
parquetwrite(lakeTestPath,  T_test);

tracker = OceanData.Adapters.ExperimentTracker();
tracker.logDataset('dataset', 'example', 'version', 'v1', 'split', 'train', 'path', lakeTrainPath, 'n_rows', height(T_train));
tracker.logDataset('dataset', 'example', 'version', 'v1', 'split', 'test',  'path', lakeTestPath,  'n_rows', height(T_test));

%% === 4. Feature Engineering ===
% mu = mean(T_train.X); sig = std(T_train.X);
% T_train.X = (T_train.X - mu) ./ sig;
% T_test.X  = (T_test.X - mu) ./ sig;

%% === 5. FeatureStore (Parquet) ===
fsTrainPath = fullfile('fs', 'train_feat_v1.parquet');
parquetwrite(fsTrainPath, T_train);
tracker.logDataset('dataset', 'example_feat', 'version', 'v1', 'split', 'train', 'path', fsTrainPath, 'n_rows', height(T_train));

%% === 6. Training ===
% mdl = trainModel(T_train);
% rmse = evaluateModel(mdl, T_test);

%% === 7 & 8. Experiment Tracking (Postgres) & Viz ===
paramsJson = jsonencode(struct('lambda', 0.1, 'solver', 'CoordinateDescent'));
tracker.logRun('run_id', 'r_001', 'dataset_version', 'v1', 'model', 'Lasso', 'metric_name', 'rmse', 'metric_value', rmse, 'params_json', paramsJson);

% viz = OceanViz.Core('Theme', 'Dark'); ...
```
