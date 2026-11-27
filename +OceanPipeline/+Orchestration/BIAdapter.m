classdef BIAdapter < handle
    % BIADAPTER  Business Intelligence adapter — pipeline output to BI formats.
    %
    %   STUB: This class reserves the DA-6 contract for BI export.
    %   All methods throw NotImplemented until DA-7 is built.
    %
    %   PLANNED FUNCTIONALITY:
    %     - export(result, path, format)  — CSV, Parquet, XLSX
    %     - push(result, dest)            — DataWarehouse table insert
    %     - schedule(pipeline, cronExpr)  — time-based pipeline trigger
    %
    %   USAGE (when implemented):
    %     bi = OceanPipeline.Orchestration.BIAdapter(pipeline);
    %     bi.export(result, './Reports/weekly.csv', 'csv');
    %
    %   See also: OceanPipeline.Orchestration.DataPipeline

    properties (SetAccess = private)
        Pipeline  OceanPipeline.Orchestration.DataPipeline
    end

    methods
        function obj = BIAdapter(pipeline)
            arguments
                pipeline OceanPipeline.Orchestration.DataPipeline
            end
            obj.Pipeline = pipeline;
        end

        % ---------------------------------------------------------------- %
        function export(~, ~, ~, ~) %#ok<INUSD>
            % EXPORT  Export pipeline result to a BI format. [STUB]
            error('OceanPipeline:Orchestration:BIAdapter:NotImplemented', ...
                'BIAdapter.export() is not yet implemented (DA-7).');
        end

        % ---------------------------------------------------------------- %
        function push(~, ~, ~) %#ok<INUSD>
            % PUSH  Push clean data into a DataWarehouse table. [STUB]
            error('OceanPipeline:Orchestration:BIAdapter:NotImplemented', ...
                'BIAdapter.push() is not yet implemented (DA-7).');
        end

        % ---------------------------------------------------------------- %
        function schedule(~, ~) %#ok<INUSD>
            % SCHEDULE  Register a cron-style pipeline trigger. [STUB]
            error('OceanPipeline.Orchestration:BIAdapter:NotImplemented', ...
                'BIAdapter.schedule() is not yet implemented (DA-7).');
        end

        % ---------------------------------------------------------------- %
        function describe(obj)
            fprintf('[BIAdapter] Pipeline: %s  [STUB — DA-7]\n', ...
                class(obj.Pipeline));
        end
    end
end
