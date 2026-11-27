classdef CsvFileReader < OceanPipeline.Ingestion.IDataReader
    % CSVFILEREADER  Read a CSV file into a numeric matrix.
    %
    %   Handles:
    %     - Header row (auto-detected or specified)
    %     - Type inference: numeric columns → double, text columns dropped
    %       unless encoder is provided
    %     - NaN for missing values ('NA', '', 'NaN', '?', 'null')
    %
    %   USAGE:
    %     reader = OceanPipeline.Ingestion.CsvFileReader('./Data/spam.csv');
    %     result = reader.read();
    %     X      = result.data;      % N × p double
    %     cols   = result.meta.header;
    %
    %     % Specify which columns to keep (by name)
    %     reader = OceanPipeline.Ingestion.CsvFileReader('./Data/spam.csv', ...
    %                  'Columns', {'age','income','label'});
    %
    %   See also: OceanPipeline.Ingestion.IDataReader

    properties (SetAccess = private)
        FilePath   (1,1) string
        Columns    cell       % column names to keep (empty = all numeric)
        HasHeader  (1,1) logical = true
    end

    methods
        function obj = CsvFileReader(filePath, opts)
            arguments
                filePath   (1,1) string
                opts.Columns  cell    = {}
                opts.HasHeader(1,1) logical = true
            end
            obj.FilePath  = filePath;
            obj.Columns   = opts.Columns;
            obj.HasHeader = opts.HasHeader;
        end

        % -------------------------------------------------------------- %
        function result = read(obj)
            % READ  Load CSV and return standard {data, meta} struct.
            if ~isfile(obj.FilePath)
                error('OceanPipeline:Ingestion:CsvFileReader:FileNotFound', ...
                    'File not found: %s', obj.FilePath);
            end

            % Read into MATLAB table (handles header, type inference)
            opts_rd = detectImportOptions(obj.FilePath, ...
                'NumHeaderLines', double(obj.HasHeader));
            % Treat common NA tokens as missing
            opts_rd.MissingRule = 'fill';
            opts_rd = setvaropts(opts_rd, opts_rd.VariableNames, ...
                'FillValue', NaN, 'TreatAsMissing', {'NA','na','?','null',''});
            T = readtable(obj.FilePath, opts_rd);

            % Select columns if specified
            if ~isempty(obj.Columns)
                present = intersect(obj.Columns, T.Properties.VariableNames);
                T = T(:, present);
            end

            % Keep only numeric columns → double matrix
            numericVars = varfun(@isnumeric, T, 'OutputFormat','uniform');
            T_num       = T(:, numericVars);
            header      = T_num.Properties.VariableNames;
            data        = table2array(T_num);          % double

            meta = OceanPipeline.Ingestion.IDataReader.buildMeta( ...
                data, obj.FilePath, header);
            result = struct('data', data, 'meta', meta);
        end

        % -------------------------------------------------------------- %
        function src = getSource(obj)
            src = obj.FilePath;
        end

        % -------------------------------------------------------------- %
        function tf = isReady(obj)
            tf = isfile(obj.FilePath);
        end
    end
end
