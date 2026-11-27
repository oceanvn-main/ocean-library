classdef IDataReader < handle
    % IDATAREADER  Abstract interface for all data ingestion readers.
    %
    %   Every concrete reader must implement read() and getSource().
    %   All readers produce the SAME output struct shape so downstream
    %   pipeline stages can process data blindly regardless of source.
    %
    %   OUTPUT CONTRACT (struct returned by read()):
    %     result.data          — (N × p) double matrix
    %     result.meta          — struct with at minimum:
    %       .nSamples          N
    %       .nFeatures         p
    %       .source            string identifier (file path, key, etc.)
    %       .header            (1×p) cell of column name strings (may be empty)
    %       .timestamp         datetime of this read
    %
    %   USAGE PATTERN:
    %     reader = OceanPipeline.Ingestion.CsvFileReader('./Data/spam.csv');
    %     result = reader.read();
    %     X      = result.data;
    %     cols   = result.meta.header;
    %     reader.describe();
    %
    %   See also: OceanPipeline.Ingestion.CsvFileReader
    %             OceanPipeline.Ingestion.DataLakeReader
    %             OceanPipeline.Ingestion.SyntheticReader

    methods (Abstract)
        result = read(obj)
        %   READ  Load data from the source and return the standard struct.

        src = getSource(obj)
        %   GETSOURCE  Return a string identifying the data source.
    end

    methods
        function describe(obj)
            % DESCRIBE  Print a human-readable summary of this reader.
            fprintf('[%s] source: %s\n', class(obj), char(obj.getSource()));
        end

        function tf = isReady(obj)
            % ISREADY  True if the source is accessible right now.
            %   Default: always true. Override in subclasses if needed.
            tf = true;
        end
    end

    methods (Access = protected, Static)
        function meta = buildMeta(data, source, header)
            % BUILDMETA  Construct the standard meta struct.
            arguments
                data   double
                source (1,1) string
                header cell = {}
            end
            meta.nSamples  = size(data, 1);
            meta.nFeatures = size(data, 2);
            meta.source    = char(source);
            meta.header    = header;
            meta.timestamp = datetime('now');
        end
    end
end
