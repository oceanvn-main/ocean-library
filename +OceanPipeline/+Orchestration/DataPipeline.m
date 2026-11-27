classdef DataPipeline < handle
    % DATAPIPELINE  Chain one reader + N transforms into a single reusable pipeline.
    %
    %   ANTI-LEAKAGE DISCIPLINE:
    %     fit(data)      — learns all transform states from training data ONLY
    %     run(data)      — applies fitted states sequentially (no learning)
    %
    %   USAGE:
    %     pipeline = OceanPipeline.Orchestration.DataPipeline( ...
    %         OceanPipeline.Ingestion.CsvFileReader('spam.csv'),   ...
    %         OceanPipeline.Transforms.MissingValueImputer(),      ...
    %         OceanPipeline.Transforms.OutlierClipper('Mode','iqr') ...
    %     );
    %
    %     pipeline.fit(X_train);          % learn from training data only
    %     result_train = pipeline.run(X_train);
    %     result_test  = pipeline.run(X_test);   % same fitted states
    %
    %     % OR: use reader to load data automatically
    %     result = pipeline.runFromSource();
    %
    %   See also: OceanPipeline.Orchestration.MLDataAdapter
    %             OceanPipeline.Transforms.IDataTransform
    %             OceanPipeline.Ingestion.IDataReader

    properties (SetAccess = private)
        Reader      % IDataReader instance (optional)
        Transforms  cell      % (1 x T) cell of IDataTransform instances
        States_     cell      % (1 x T) cell of fitted states
        IsFitted    (1,1) logical = false
        StepLog     cell      % execution log for diagnostics
    end

    methods
        function obj = DataPipeline(varargin)
            % DATAPIPELINE  Constructor — accepts (reader?, t1, t2, ...) in any order.
            %   Reader is optional. Transforms can be passed in any number.
            obj.Transforms = {};
            obj.States_    = {};
            obj.StepLog    = {};
            obj.Reader     = [];

            for i = 1:numel(varargin)
                arg = varargin{i};
                if isa(arg, 'OceanPipeline.Ingestion.IDataReader')
                    obj.Reader = arg;
                elseif isa(arg, 'OceanPipeline.Transforms.IDataTransform')
                    obj.Transforms{end+1} = arg;
                else
                    error('OceanPipeline:Orchestration:DataPipeline:BadArg', ...
                        'Argument %d is neither an IDataReader nor an IDataTransform.', i);
                end
            end
        end

        % ---------------------------------------------------------------- %
        function fit(obj, data)
            % FIT  Learn all transform states from training data.
            %   Calls transform.fit(data) sequentially, updating data after each.
            %   MUST be called before run(). Data must be (N x p) double.
            arguments
                obj
                data double
            end
            obj.States_ = cell(1, numel(obj.Transforms));
            obj.StepLog = {};

            current = data;
            for t = 1:numel(obj.Transforms)
                tr            = obj.Transforms{t};
                state         = tr.fit(current);
                obj.States_{t} = state;

                % Apply to keep data consistent for next transform's fit()
                [current, ~]  = obj.safeApply_(tr, current, state);

                obj.StepLog{end+1} = sprintf('fit: %s  [%d x %d → %d x %d]', ...
                    class(tr), size(data,1), size(data,2), ...
                    size(current,1), size(current,2));
            end
            obj.IsFitted = true;
        end

        % ---------------------------------------------------------------- %
        function result = run(obj, data)
            % RUN  Apply all fitted transforms to data sequentially.
            %   Returns struct(data, meta) compatible with IDataReader output.
            arguments
                obj
                data double
            end
            if ~obj.IsFitted
                error('OceanPipeline:Orchestration:DataPipeline:NotFitted', ...
                    'Pipeline has not been fitted. Call fit(X_train) first.');
            end

            current = data;
            reports = {};  % validation reports (from DataValidator steps)

            for t = 1:numel(obj.Transforms)
                [current, report] = obj.safeApply_(obj.Transforms{t}, current, obj.States_{t});
                if ~isempty(report)
                    reports{end+1} = report; %#ok<AGROW>
                end
            end

            result.data            = current;
            result.meta.nSamples   = size(current, 1);
            result.meta.nFeatures  = size(current, 2);
            result.meta.timestamp  = datetime('now');
            result.meta.reports    = reports;   % any ValidationReports from DataValidator steps
        end

        % ---------------------------------------------------------------- %
        function result = runFromSource(obj)
            % RUNFROMSOURCE  Read from the attached reader, then run transforms.
            if isempty(obj.Reader)
                error('OceanPipeline:Orchestration:DataPipeline:NoReader', ...
                    'No reader was provided. Use run(data) instead.');
            end
            if ~obj.Reader.isReady()
                error('OceanPipeline:Orchestration:DataPipeline:SourceNotReady', ...
                    'Reader source is not ready: %s', obj.Reader.getSource());
            end

            raw    = obj.Reader.read();
            result = obj.run(raw.data);
            % Merge reader meta with pipeline meta
            fields = fieldnames(raw.meta);
            for i = 1:numel(fields)
                if ~isfield(result.meta, fields{i})
                    result.meta.(fields{i}) = raw.meta.(fields{i});
                end
            end
        end

        % ---------------------------------------------------------------- %
        function describe(obj)
            % DESCRIBE  Print the pipeline structure and fit status.
            fprintf('\n=== DataPipeline ===\n');
            if ~isempty(obj.Reader)
                fprintf('  Reader   : %s  [%s]\n', class(obj.Reader), obj.Reader.getSource());
            else
                fprintf('  Reader   : (none — uses run(data))\n');
            end
            fprintf('  Steps    : %d transform(s)\n', numel(obj.Transforms));
            for t = 1:numel(obj.Transforms)
                fprintf('    [%d] %s\n', t, class(obj.Transforms{t}));
            end
            fprintf('  Fitted   : %s\n\n', mat2str(obj.IsFitted));
        end
    end

    % ================================================================== %
    methods (Access = private)
        function [out, report] = safeApply_(~, tr, data, state)
            % Route apply() correctly for DataValidator (two outputs) vs others (one output)
            report = [];
            if isa(tr, 'OceanPipeline.Transforms.DataValidator')
                [out, report] = tr.apply(data, state);
            else
                out = tr.apply(data, state);
            end
        end
    end
end
