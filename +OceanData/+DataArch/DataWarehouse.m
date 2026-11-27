classdef DataWarehouse < handle
    % DATAWAREHOUSE  Gold Layer Analytical Storage (OLAP Mock).
    %
    %   Mimics a columnar analytical data warehouse like Snowflake or BigQuery.
    %   Unlike Postgres (which is OLTP and handles transactions/metadata), 
    %   this warehouse is designed strictly for analytical aggregations 
    %   (GROUP BY, SUM, AVG) over structured, historical business data.
    %
    %   It sits at the Gold Layer of the medallion architecture. You do NOT
    %   store raw feature matrices here. You store aggregated summaries like
    %   model_performance, regional_sales, or user_churn.
    %
    %   USAGE:
    %     dw = OceanData.DataArch.DataWarehouse('rootDir', './warehouse/');
    %     
    %     % Insert aggregated data
    %     dw.insertBatch('model_performance', resultsTable);
    %
    %     % Perform OLAP analytical queries
    %     avgRmseByModel = dw.aggregate('model_performance', 'GroupBy', 'model', 'AggFunc', @mean, 'Target', 'rmse');
    
    properties (SetAccess = private)
        RootDir (1,1) string
    end
    
    methods
        function obj = DataWarehouse(opts)
            arguments
                opts.rootDir (1,1) string = "./warehouse/"
            end
            obj.RootDir = opts.rootDir;
            if ~isfolder(obj.RootDir), mkdir(obj.RootDir); end
        end
        
        function insertBatch(obj, tableName, dataTable)
            % INSERTBATCH Appends a batch of aggregated data to a warehouse table.
            arguments
                obj
                tableName (1,1) string
                dataTable table
            end
            
            tablePath = fullfile(obj.RootDir, tableName + ".parquet");
            
            if isfile(tablePath)
                existing = parquetread(tablePath);
                combined = [existing; dataTable];
                parquetwrite(tablePath, combined);
            else
                parquetwrite(tablePath, dataTable);
            end
            fprintf('[DataWarehouse] Ingested %d rows into Gold table: %s\n', height(dataTable), tableName);
        end
        
        function result = aggregate(obj, tableName, opts)
            % AGGREGATE Mimics an OLAP 'SELECT Target, AggFunc(Target) GROUP BY GroupBy' query.
            arguments
                obj
                tableName (1,1) string
                opts.GroupBy (1,1) string
                opts.Target (1,1) string
                opts.AggFunc (1,1) function_handle = @mean
            end
            
            tablePath = fullfile(obj.RootDir, tableName + ".parquet");
            if ~isfile(tablePath)
                error('DataWarehouse:TableNotFound', 'Gold table %s does not exist.', tableName);
            end
            
            % Load columnar data (Parquet is fast for this)
            data = parquetread(tablePath);
            
            % Mimic SQL GROUP BY
            [groups, ~, idx] = unique(data.(opts.GroupBy));
            aggs = zeros(length(groups), 1);
            
            for i = 1:length(groups)
                aggs(i) = opts.AggFunc(data.(opts.Target)(idx == i));
            end
            
            result = table(groups, aggs, 'VariableNames', {char(opts.GroupBy), 'Aggregated_' + opts.Target});
        end
    end
end
