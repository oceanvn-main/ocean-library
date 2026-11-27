classdef SyntheticReader < OceanPipeline.Ingestion.IDataReader
    % SYNTHETICREADER  Generate synthetic data via OceanStats data generators.
    %
    %   Wraps any IDataGenerator (ClassificationGenerator, RegressionGenerator,
    %   etc.) or OceanStats.Data.DataFactory into the IDataReader interface.
    %
    %   Two construction modes:
    %     1. Generator object  — pass any IDataGenerator handle
    %     2. DataFactory name  — pass a string key and DataFactory.load() is called
    %
    %   USAGE:
    %     % Mode 1: inject a generator
    %     gen    = OceanStats.Data.Generators.ClassificationGenerator( ...
    %                  'nSamples', 300, 'nFeatures', 10);
    %     reader = OceanPipeline.Ingestion.SyntheticReader(gen);
    %     result = reader.read();
    %
    %     % Mode 2: DataFactory name
    %     reader = OceanPipeline.Ingestion.SyntheticReader('spam');
    %     result = reader.read();   % uses DataFactory.load('spam')
    %
    %   See also: OceanStats.Data.DataFactory
    %             OceanStats.Data.Generators.*
    %             OceanPipeline.Ingestion.IDataReader

    properties (SetAccess = private)
        Source_     % IDataGenerator handle, OR string factory key
        Mode_       (1,1) string  % 'generator' | 'factory'
    end

    methods
        function obj = SyntheticReader(source)
            arguments
                source   % IDataGenerator or string
            end
            if ischar(source) || isstring(source)
                obj.Mode_   = 'factory';
                obj.Source_ = string(source);
            else
                obj.Mode_   = 'generator';
                obj.Source_ = source;
            end
        end

        % -------------------------------------------------------------- %
        function result = read(obj)
            switch obj.Mode_
                case 'factory'
                    % DataFactory.load returns struct with .X and .Y
                    ds   = OceanStats.Data.DataFactory.load(char(obj.Source_));
                    data = ds.X;
                    if isfield(ds, 'Y')
                        data = [data, ds.Y(:)];  % append label as last column
                    end
                    srcStr = sprintf('factory://%s', obj.Source_);
                    header = {};

                case 'generator'
                    % IDataGenerator.generate() returns struct with .X (and .Y)
                    ds   = obj.Source_.generate();
                    data = ds.X;
                    if isfield(ds, 'Y')
                        data = [data, ds.Y(:)];
                    end
                    srcStr = sprintf('generator://%s', class(obj.Source_));
                    header = {};
            end

            data   = double(data);
            meta   = OceanPipeline.Ingestion.IDataReader.buildMeta( ...
                data, srcStr, header);
            result = struct('data', data, 'meta', meta);
        end

        % -------------------------------------------------------------- %
        function src = getSource(obj)
            if obj.Mode_ == "factory"
                src = sprintf('factory://%s', obj.Source_);
            else
                src = sprintf('generator://%s', class(obj.Source_));
            end
        end

        % -------------------------------------------------------------- %
        function tf = isReady(~)
            tf = true;  % synthetic data is always available
        end
    end
end
