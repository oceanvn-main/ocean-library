classdef DataLakeReader < OceanPipeline.Ingestion.IDataReader
    % DATALAKEREADER  Read a DataLake partition into the standard struct.
    %   It implements the classic Adapter Pattern. The DataLake has its own API (lake.get(key) returns an entry struct), 
    %   but the pipeline only speaks IDataReader (which requires result.data and result.meta). 
    %   This class translates between the two without modifying either.
    %   Thin wrapper around DataLake.get .mat that adapts the lake entry
    %   to the IDataReader output format  .
    %
    %   USAGE:
    %     lake   = OceanData.DataArch.DataLake('rootDir', './lake/');
    %     reader = OceanPipeline.Ingestion.DataLakeReader(lake, 'raw/spam/2026');
    %     result = reader.read();
    %     X      = result.data;
    %
    %   The lake entry is expected to contain a numeric matrix in entry.data.
    %   If entry.data is a MATLAB table, numeric columns are extracted.
    %
    %   See also: OceanData.DataArch.DataLake
    %             OceanPipeline.Ingestion.IDataReader

    properties (SetAccess = private)
        Lake    OceanData.DataArch.DataLake
        Key     (1,1) string
    end

    methods
        function obj = DataLakeReader(lake, key)
            arguments
                lake OceanData.DataArch.DataLake
                key  (1,1) string
            end
            obj.Lake = lake;
            obj.Key  = key;
        end

        % -------------------------------------------------------------- %
        function result = read(obj)
            if ~obj.Lake.exists(obj.Key)
                error('OceanPipeline:Ingestion:DataLakeReader:KeyNotFound', ...
                    'DataLake key not found: "%s"', obj.Key);
            end

            entry = obj.Lake.get(obj.Key);
            raw   = entry.data;

            % Handle table entries
            if istable(raw)
                numericVars = varfun(@isnumeric, raw, 'OutputFormat','uniform');
                header      = raw.Properties.VariableNames(numericVars);
                data        = table2array(raw(:, numericVars));
            else
                data   = double(raw);
                header = {};
            end

            % Inherit any meta the lake stored (e.g., source CSV name)
            baseMeta = OceanPipeline.Ingestion.IDataReader.buildMeta( ...
                data, sprintf('datalake://%s', obj.Key), header);

            % Merge lake's original meta (don't lose provenance)
            lakeMeta = entry.meta;
            fields   = fieldnames(lakeMeta);
            for i = 1:numel(fields)
                baseMeta.(fields{i}) = lakeMeta.(fields{i});
            end
            % Restore our canonical fields (don't let lake meta overwrite)
            baseMeta.nSamples  = size(data, 1);
            baseMeta.nFeatures = size(data, 2);
            baseMeta.source    = sprintf('datalake://%s', obj.Key);

            result = struct('data', data, 'meta', baseMeta);
        end

        % -------------------------------------------------------------- %
        function src = getSource(obj)
            src = sprintf('datalake://%s', obj.Key);
        end

        % -------------------------------------------------------------- %
        function tf = isReady(obj)
            tf = obj.Lake.exists(obj.Key);
        end
    end
end
