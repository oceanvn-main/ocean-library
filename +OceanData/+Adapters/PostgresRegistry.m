classdef PostgresRegistry < handle
    % POSTGRESREGISTRY  Tier 2 and Tier 3 storage (PostgreSQL Database).
    %
    %   Implements the exact same interface as MetadataRegistry, but securely
    %   writes records directly into a live Postgres database.
    %
    %   USAGE:
    %     dbConfig = OceanData.Adapters.DbConfig();
    %     tracker = OceanData.Adapters.PostgresRegistry(dbConfig);
    %     tracker.connect();
    %
    %     % Tier 2: Register a Dataset
    %     tracker.registerDataset('prostate', 'v1', 'train', 'lake/train_v1.parquet', 67);
    %
    %     % Tier 3: Log a Run
    %     tracker.logRun('run_001', 'prostate_v1', 'Lasso', 'rmse', 0.12, '{"lambda": 0.01}');
    
    properties (SetAccess = private)
        Db OceanData.Adapters.PostgresAdapter
    end
    
    methods
        function obj = PostgresRegistry(config)
            arguments
                config OceanData.Adapters.DbConfig
            end
            obj.Db = OceanData.Adapters.PostgresAdapter(config);
        end
        
        function connect(obj)
            obj.Db.connect();
            obj.initTables_();
        end
        
        function disconnect(obj)
            obj.Db.disconnect();
        end
        
        function registerDataset(obj, name, version, split, path, n_rows)
            % REGISTERDATASET Logs the location of a Parquet feature matrix.
            arguments
                obj
                name (1,1) string
                version (1,1) string
                split (1,1) string
                path (1,1) string
                n_rows (1,1) double
            end
            
            sql = sprintf("INSERT INTO datasets (dataset_name, version, split, path, n_rows, created_at) VALUES ('%s', '%s', '%s', '%s', %d, NOW())", ...
                name, version, split, path, n_rows);
            
            obj.Db.execute(sql);
            fprintf('[PostgresRegistry] Registered %s (%s split: %s) -> %s\n', name, version, split, path);
        end
        
        function info = getDataset(obj, name, version, split)
            % GETDATASET Retrieves the metadata for a specific dataset version.
            arguments
                obj
                name (1,1) string
                version (1,1) string
                split (1,1) string
            end
            
            sql = sprintf("SELECT * FROM datasets WHERE dataset_name = '%s' AND version = '%s' AND split = '%s' ORDER BY created_at DESC LIMIT 1", name, version, split);
            rows = obj.Db.fetch(sql);
            
            if isempty(rows)
                error('PostgresRegistry:NotFound', 'Dataset %s %s %s not found.', name, version, split);
            end
            info = rows(1);
        end
        
        function logRun(obj, run_id, dataset_version, model, metric_name, metric_value, params_json)
            % LOGRUN Logs experiment results to the registry.
            arguments
                obj
                run_id (1,1) string
                dataset_version (1,1) string
                model (1,1) string
                metric_name (1,1) string
                metric_value (1,1) double
                params_json (1,1) string
            end
            
            sql = sprintf("INSERT INTO runs (run_id, dataset_version, model, metric_name, metric_value, params_json, created_at) VALUES ('%s', '%s', '%s', '%s', %f, '%s', NOW())", ...
                run_id, dataset_version, model, metric_name, metric_value, params_json);
            
            obj.Db.execute(sql);
            fprintf('[PostgresRegistry] Logged run: %s (%s: %.4f)\n', run_id, metric_name, metric_value);
        end
    end
    
    methods (Access = private)
        function initTables_(obj)
            sql_datasets = "CREATE TABLE IF NOT EXISTS datasets (" + ...
                "dataset_name VARCHAR(255), " + ...
                "version VARCHAR(50), " + ...
                "split VARCHAR(50), " + ...
                "path TEXT, " + ...
                "n_rows INT, " + ...
                "created_at TIMESTAMP)";
            obj.Db.execute(sql_datasets);
            
            sql_runs = "CREATE TABLE IF NOT EXISTS runs (" + ...
                "run_id VARCHAR(255), " + ...
                "dataset_version VARCHAR(255), " + ...
                "model VARCHAR(255), " + ...
                "metric_name VARCHAR(100), " + ...
                "metric_value DOUBLE PRECISION, " + ...
                "params_json TEXT, " + ...
                "created_at TIMESTAMP)";
            obj.Db.execute(sql_runs);
        end
    end
end
