classdef MetadataRegistry < handle
    % METADATAREGISTRY  Tier 2 and Tier 3 storage (Local File Fallback).
    %
    %   Replaces the legacy DataWarehouse. This registry STRICTLY stores 
    %   metadata, versioning, and experiment metrics. It NEVER stores 
    %   raw feature matrices (which belong in the Parquet DataLake).
    %
    %   This is the local disk fallback. In production, you would use
    %   OceanData.Adapters.PostgresRegistry which implements this same
    %   interface against a PostgreSQL database.
    %
    %   USAGE:
    %     registry = OceanData.DataArch.MetadataRegistry('rootDir', './meta/');
    %     
    %     % Tier 2: Register a Dataset
    %     registry.registerDataset('prostate', 'v1', 'train', 'lake/train_v1.parquet', 67);
    %
    %     % Tier 3: Log a Run
    %     registry.logRun('run_001', 'prostate_v1', 'Lasso', 'rmse', 0.12, '{"lambda": 0.01}');
    
    properties (SetAccess = private)
        RootDir (1,1) string
        DatasetsFile (1,1) string
        RunsFile (1,1) string
    end
    
    methods
        function obj = MetadataRegistry(opts)
            arguments
                opts.rootDir (1,1) string = "./metadata/"
            end
            obj.RootDir = opts.rootDir;
            if ~isfolder(obj.RootDir), mkdir(obj.RootDir); end
            
            obj.DatasetsFile = fullfile(obj.RootDir, 'datasets.mat');
            obj.RunsFile = fullfile(obj.RootDir, 'runs.mat');
            
            % Initialize files if they don't exist
            if ~isfile(obj.DatasetsFile)
                datasets = table('Size', [0 6], ...
                    'VariableTypes', {'string', 'string', 'string', 'string', 'double', 'datetime'}, ...
                    'VariableNames', {'dataset_name', 'version', 'split', 'path', 'n_rows', 'created_at'});
                save(obj.DatasetsFile, 'datasets');
            end
            
            if ~isfile(obj.RunsFile)
                runs = table('Size', [0 7], ...
                    'VariableTypes', {'string', 'string', 'string', 'string', 'double', 'string', 'datetime'}, ...
                    'VariableNames', {'run_id', 'dataset_version', 'model', 'metric_name', 'metric_value', 'params_json', 'created_at'});
                save(obj.RunsFile, 'runs');
            end
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
            
            loaded = load(obj.DatasetsFile);
            datasets = loaded.datasets;
            
            newRow = {name, version, split, path, n_rows, datetime('now')};
            datasets = [datasets; newRow]; %#ok<AGROW>
            
            save(obj.DatasetsFile, 'datasets');
            fprintf('[MetadataRegistry] Registered dataset: %s (%s) -> %s\n', name, split, path);
        end
        
        function info = getDataset(obj, name, version, split)
            % GETDATASET Retrieves the metadata for a specific dataset version.
            arguments
                obj
                name (1,1) string
                version (1,1) string
                split (1,1) string
            end
            
            loaded = load(obj.DatasetsFile);
            ds = loaded.datasets;
            
            idx = (ds.dataset_name == name) & (ds.version == version) & (ds.split == split);
            if ~any(idx)
                error('MetadataRegistry:NotFound', 'Dataset %s %s %s not found.', name, version, split);
            end
            
            info = table2struct(ds(idx, :));
            info = info(end); % Return the latest if there are duplicates
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
            
            loaded = load(obj.RunsFile);
            runs = loaded.runs;
            
            newRow = {run_id, dataset_version, model, metric_name, metric_value, params_json, datetime('now')};
            runs = [runs; newRow]; %#ok<AGROW>
            
            save(obj.RunsFile, 'runs');
            fprintf('[MetadataRegistry] Logged run: %s (%s: %.4f)\n', run_id, metric_name, metric_value);
        end
    end
end
