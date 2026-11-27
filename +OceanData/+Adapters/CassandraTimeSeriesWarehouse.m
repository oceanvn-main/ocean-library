classdef CassandraTimeSeriesWarehouse < handle
    % CASSANDRATIMESERIESWAREHOUSE  Time-series optimized Cassandra backend.
    %
    %   Unlike the generic DataWarehouse, this class treats Cassandra correctly
    %   as a wide-column store using partition keys and clustering keys.
    %   It does NOT implement IDataStore because pure key-value semantics
    %   (like get/put/delete) do not apply to time-series tables efficiently.
    %
    %   SCHEMA:
    %     series_id  text      (PARTITION KEY - for fast lookups)
    %     ts         timestamp (CLUSTERING KEY - for time range queries)
    %     value      double    (The measurement)
    %     meta       text      (Optional JSON metadata blob)
    %
    %   USAGE:
    %     tsw = OceanData.Adapters.CassandraTimeSeriesWarehouse(cfg);
    %     tsw.connect();
    %     tsw.createTable('sensors');
    %     tsw.append('sensors', 'temp_1', datetime('now'), 22.5);
    %     
    %     % Batch append (very fast)
    %     tVec = [datetime('yesterday'); datetime('today')];
    %     xVec = [21.0; 22.1];
    %     tsw.appendBatch('sensors', 'temp_1', tVec, xVec);
    %
    %     % Query by series ID
    %     T = tsw.query('sensors', 'temp_1');
    %
    %     % Query by time range
    %     T = tsw.queryRange('sensors', 'temp_1', tVec(1), tVec(2));
    %
    %   See also: OceanData.Adapters.CassandraAdapter

    properties (SetAccess = private)
        Adapter_    OceanData.Adapters.CassandraAdapter
        keyspace_   string
    end

    methods
        function obj = CassandraTimeSeriesWarehouse(config)
            % CASSANDRATIMESERIESWAREHOUSE  Construct time-series warehouse.
            arguments
                config OceanData.Adapters.DbConfig
            end
            if ~strcmp(config.Vendor, 'Cassandra')
                error('CassandraTimeSeriesWarehouse:InvalidConfig', ...
                      'Requires a Cassandra DbConfig.');
            end
            obj.keyspace_ = config.Database;
            obj.Adapter_  = OceanData.Adapters.CassandraAdapter(config);
        end

        function connect(obj)
            % CONNECT  Open connection to Cassandra.
            obj.Adapter_.connect();
        end

        function disconnect(obj)
            % DISCONNECT  Close connection.
            obj.Adapter_.disconnect();
        end

        function createTable(obj, name)
            % CREATETABLE  Create the standard time-series schema.
            %   Schema: (series_id, ts) forms the Primary Key. Rows are 
            %   physically stored in ascending time order per series_id.
            arguments
                obj
                name (1,1) string
            end
            fullTable = sprintf('%s.%s', obj.keyspace_, name);
            cql = sprintf([ ...
                'CREATE TABLE IF NOT EXISTS %s (', ...
                '  series_id text, ', ...
                '  ts timestamp, ', ...
                '  value double, ', ...
                '  meta text, ', ...
                '  PRIMARY KEY (series_id, ts)', ...
                ') WITH CLUSTERING ORDER BY (ts ASC)'], fullTable);
            obj.Adapter_.execute(cql);
            fprintf('[CassandraTimeSeriesWarehouse] TS Table ready: %s\n', name);
        end

        function append(obj, table, seriesId, timeStamp, value, meta)
            % APPEND  Insert a single time-series data point.
            arguments
                obj
                table     (1,1) string
                seriesId  (1,1) string
                timeStamp (1,1) datetime
                value     (1,1) double
                meta      (1,1) string   = "{}"
            end
            fullTable = sprintf('%s.%s', obj.keyspace_, table);
            dtIso = char(datetime(timeStamp, 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSS''Z''', 'TimeZone','UTC'));
            cql = sprintf('INSERT INTO %s (series_id, ts, value, meta) VALUES (''%s'', ''%s'', %g, ''%s'')', ...
                          fullTable, seriesId, dtIso, value, meta);
            obj.Adapter_.execute(cql);
        end

        function appendBatch(obj, table, seriesId, timeStampVec, valueVec, metaCell)
            % APPENDBATCH  Bulk insert points for a single series.
            %   Uses a prepared statement with concurrent execution.
            arguments
                obj
                table        (1,1) string
                seriesId     (1,1) string
                timeStampVec (:,1) datetime
                valueVec     (:,1) double
                metaCell     (:,1) cell     = {}
            end
            n = numel(timeStampVec);
            if n ~= numel(valueVec)
                error('CassandraTimeSeriesWarehouse:DimMismatch', ...
                      'timeStampVec and valueVec must have the same length.');
            end
            if isempty(metaCell)
                metaCell = repmat({"{}"}, n, 1);
            end
            
            % Convert datetime to ISO-8601 strings
            dtIso = cell(n, 1);
            for i = 1:n
                dtIso{i} = char(datetime(timeStampVec(i), 'Format', ...
                                 'yyyy-MM-dd''T''HH:mm:ss.SSS''Z''', 'TimeZone','UTC'));
            end
            
            tsJson   = jsonencode(dtIso);
            valJson  = jsonencode(valueVec);
            metaJson = jsonencode(metaCell);
            
            try
                obj.Adapter_.Bridge_.append_batch_ts( ...
                    char(obj.keyspace_), char(table), char(seriesId), ...
                    tsJson, valJson, metaJson);
            catch ex
                error('CassandraTimeSeriesWarehouse:appendBatch', ...
                      'Batch append failed: %s', ex.message);
            end
        end

        function T = query(obj, table, seriesId)
            % QUERY  Fetch all rows for a given series.
            arguments
                obj
                table    (1,1) string
                seriesId (1,1) string
            end
            fullTable = sprintf('%s.%s', obj.keyspace_, table);
            cql = sprintf('SELECT * FROM %s WHERE series_id = ''%s''', fullTable, seriesId);
            T = obj.Adapter_.fetch(cql);
            T = obj.formatResultTable_(T);
        end

        function T = queryRange(obj, table, seriesId, tStart, tEnd)
            % QUERYRANGE  Fetch rows for a series within a time window.
            %   tStart and tEnd are inclusive datetime bounds.
            arguments
                obj
                table    (1,1) string
                seriesId (1,1) string
                tStart   (1,1) datetime
                tEnd     (1,1) datetime
            end
            t0Iso = char(datetime(tStart, 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSS''Z''', 'TimeZone','UTC'));
            t1Iso = char(datetime(tEnd,   'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSS''Z''', 'TimeZone','UTC'));
            
            try
                jsonStr = char(obj.Adapter_.Bridge_.query_range( ...
                    char(obj.keyspace_), char(table), char(seriesId), t0Iso, t1Iso));
                rows = jsondecode(jsonStr);
                if isempty(rows)
                    T = table();
                else
                    T = struct2table(rows);
                end
            catch ex
                error('CassandraTimeSeriesWarehouse:queryRange', ...
                      'Query range failed: %s', ex.message);
            end
            T = obj.formatResultTable_(T);
        end

        function deleteSeries(obj, table, seriesId)
            % DELETESERIES  Delete all data for a specific series.
            %   Uses a partition key deletion which is very fast in Cassandra.
            arguments
                obj
                table    (1,1) string
                seriesId (1,1) string
            end
            fullTable = sprintf('%s.%s', obj.keyspace_, table);
            cql = sprintf('DELETE FROM %s WHERE series_id = ''%s''', fullTable, seriesId);
            obj.Adapter_.execute(cql);
        end
    end

    methods (Access = private)
        function T = formatResultTable_(~, T)
            % Ensure 'ts' is converted from ISO-8601 string back to datetime
            if isempty(T) || ~ismember('ts', T.Properties.VariableNames)
                return;
            end
            
            % 'ts' comes back as cell array of char or string
            tsStr = T.ts;
            if iscell(tsStr)
                tsStr = string(tsStr);
            end
            
            % Python datetimes from Cassandra may or may not have 'Z' 
            % Using generic datetime conversion
            try
                dtConfig = {'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSZ', 'TimeZone', 'UTC'};
                T.ts = datetime(tsStr, dtConfig{:});
            catch
                % Fallback if fractional seconds differ
                % e.g '2023-01-01T12:00:00+00:00'
                T.ts = datetime(tsStr, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ssZ', 'TimeZone', 'UTC');
            end
            
            % Reorder columns nicely
            cols = T.Properties.VariableNames;
            pref = {'series_id', 'ts', 'value', 'meta'};
            actualPref = pref(ismember(pref, cols));
            other = setdiff(cols, actualPref, 'stable');
            T = T(:, [actualPref, other]);
        end
    end
end
