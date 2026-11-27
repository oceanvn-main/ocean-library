classdef MatFileReader < OceanPipeline.Ingestion.IDataReader
    % MATFILEREADER  Read a .mat file into a numeric matrix.
    %
    %   Supports two modes:
    %     1. Named variable: load a specific variable by name (e.g., 'X')
    %     2. Auto-detect:    if only one numeric variable exists, load it
    %
    %   If the variable is a MATLAB table, it is converted to a double matrix
    %   (numeric columns only).
    %
    %   USAGE:
    %     % Load variable 'X' from file
    %     reader = OceanPipeline.Ingestion.MatFileReader('./Data/spam.mat', ...
    %                  'Variable', 'X');
    %     result = reader.read();
    %
    %     % Auto-detect the numeric variable
    %     reader = OceanPipeline.Ingestion.MatFileReader('./Data/spam.mat');
    %     result = reader.read();
    %
    %   See also: OceanPipeline.Ingestion.IDataReader

    properties (SetAccess = private)
        FilePath  (1,1) string
        Variable  (1,1) string = ""   % "" = auto-detect
    end

    methods
        function obj = MatFileReader(filePath, opts)
            arguments
                filePath  (1,1) string
                opts.Variable (1,1) string = ""
            end
            obj.FilePath = filePath;
            obj.Variable = opts.Variable;
        end

        % -------------------------------------------------------------- %
        function result = read(obj)
            if ~isfile(obj.FilePath)
                error('OceanPipeline:Ingestion:MatFileReader:FileNotFound', ...
                    'File not found: %s', obj.FilePath);
            end

            info = whos('-file', obj.FilePath);

            if obj.Variable ~= ""
                varName = char(obj.Variable);
            else
                % Auto-detect: first numeric variable
                numVarIdx = find(arrayfun(@(s) ...
                    any(strcmp(s.class, ...
                         {'double','single','int8','int16','int32','int64', ...
                          'uint8','uint16','uint32','uint64'})), info), 1);
                if isempty(numVarIdx)
                    error('OceanPipeline:Ingestion:MatFileReader:NoNumericVar', ...
                        'No numeric variable found in: %s', obj.FilePath);
                end
                varName = info(numVarIdx).name;
            end

            S    = load(obj.FilePath, varName);
            raw  = S.(varName);

            % Handle MATLAB table input
            if istable(raw)
                numericVars = varfun(@isnumeric, raw, 'OutputFormat','uniform');
                header      = raw.Properties.VariableNames(numericVars);
                data        = table2array(raw(:, numericVars));
            else
                data   = double(raw);
                header = {};
            end
            
            
            meta   = OceanPipeline.Ingestion.IDataReader.buildMeta( ...
                data, sprintf('%s::%s', obj.FilePath, varName), header);
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
