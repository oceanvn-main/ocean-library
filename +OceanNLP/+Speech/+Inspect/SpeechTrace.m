classdef SpeechTrace < handle
    % SPEECHTRACE  Bounded capture of the real intermediates of a speech pipeline run.
    %
    %   t  = OceanNLP.Speech.Inspect.SpeechTrace()
    %   t  = OceanNLP.Speech.Inspect.SpeechTrace('MaxArtifactBytes', 8e6, 'MaxTotalBytes', 64e6, ...
    %          'Enabled', true)
    %   id = t.record(name, stage, value, inputIds)          % -> "artifact-<k>" ("" when disabled)
    %   id = t.record(..., 'Summary', "what this step did")
    %   r  = t.reader()                                      % OceanNLP.Inspect.MemoryReader
    %
    %   Every record() call adds ONE step (Id "step-<k>", Name, Stage, InputIds,
    %   OutputIds = the new artifact id, Summary, Status, Sequence) and ONE artifact
    %   (Id "artifact-<k>", Kind = class(value), Name, Value, Summary, Status,
    %   ByteCount) -- the same step/artifact schema as OceanNLP.Inspect.PipelineAdapter,
    %   so OceanViz.App.NLP.ProcessExplorer and every MemoryReader consumer read it.
    %
    %   MEMORY BUDGET (same rule as PipelineAdapter): with B = byte size of value,
    %     B > MaxArtifactBytes  or  B > MaxTotalBytes - BytesUsed  ->  Value = [],
    %     Status = "omitted", Summary says so; type (Kind), size (Size field in the
    %     Summary) and ByteCount are still kept, and the step still records lineage.
    %   Otherwise Value is stored and BytesUsed += B.
    %   B is counted deeply: numeric/char/logical/string arrays by whos; struct and
    %   cell arrays as the sum of their contents; objects as the sum of their
    %   properties (whos reports only a handle's 8-byte reference).
    %
    %   DISABLED TRACE ('Enabled', false): record() returns "" and touches nothing
    %   (no state, no RNG, no allocation of the value); reader() returns an empty
    %   reader. Callers may test t.Enabled to skip computing inspection-only values.
    %
    %   The trace only stores what it is given; it never computes anything itself,
    %   so attaching it cannot change a pipeline's outputs or the RNG state.

    properties (SetAccess = immutable)
        Enabled (1,1) logical
        MaxArtifactBytes (1,1) double
        MaxTotalBytes (1,1) double
    end

    properties (SetAccess = private)
        BytesUsed (1,1) double = 0
    end

    properties (Access = private)
        Steps struct
        Artifacts struct
        Count (1,1) double = 0
    end

    methods
        function obj = SpeechTrace(opts)
            arguments
                opts.MaxArtifactBytes (1,1) double {mustBeNonnegative} = 8e6
                opts.MaxTotalBytes (1,1) double {mustBeNonnegative} = 64e6
                opts.Enabled (1,1) logical = true
            end
            obj.Enabled = opts.Enabled;
            obj.MaxArtifactBytes = opts.MaxArtifactBytes;
            obj.MaxTotalBytes = opts.MaxTotalBytes;
            obj.Steps = OceanNLP.Speech.Inspect.SpeechTrace.emptySteps();
            obj.Artifacts = OceanNLP.Speech.Inspect.SpeechTrace.emptyArtifacts();
        end

        function id = record(obj, name, stage, value, inputIds, opts)
            % RECORD  Store VALUE (within budget) as the output of step NAME in STAGE.
            %   inputIds: artifact ids this value was computed from ("" entries ignored).
            arguments
                obj
                name (1,1) string
                stage (1,1) string
                value
                inputIds string = strings(1, 0)
                opts.Summary (1,1) string = ""
            end
            id = "";
            if ~obj.Enabled
                return;
            end
            inputIds = reshape(inputIds(inputIds ~= ""), 1, []);
            if ~isempty(inputIds)
                known = string([obj.Artifacts.Id]);
                unknown = inputIds(~ismember(inputIds, known));
                if ~isempty(unknown)
                    error('OceanNLP:Speech:Inspect:SpeechTrace:UnknownInput', ...
                        'Input artifact(s) %s were not recorded in this trace.', join(unknown, ', '));
                end
            end
            k = obj.Count + 1;
            bytes = OceanNLP.Speech.Inspect.SpeechTrace.byteCount(value);
            kind = string(class(value));
            shape = strjoin(string(size(value)), "×");
            summary = kind + " · " + shape + " · " + OceanNLP.Speech.Inspect.SpeechTrace.formatBytes(bytes);
            if bytes > obj.MaxArtifactBytes || bytes > obj.MaxTotalBytes - obj.BytesUsed
                stored = [];
                status = "omitted";
                summary = summary + " · value omitted by memory budget";
            else
                stored = value;
                status = "recorded";
                obj.BytesUsed = obj.BytesUsed + bytes;
            end
            id = "artifact-" + k;
            obj.Artifacts(k) = struct('Id', id, 'Kind', kind, 'Name', name, 'Value', {stored}, ...
                'Summary', summary, 'Status', status, 'ByteCount', bytes);
            stepSummary = opts.Summary;
            if stepSummary == ""
                stepSummary = "Produced " + summary;
            else
                stepSummary = stepSummary + " → " + summary;
            end
            obj.Steps(k) = struct('Id', "step-" + k, 'Name', name, 'Stage', stage, ...
                'InputIds', inputIds, 'OutputIds', id, 'Summary', stepSummary, ...
                'Status', status, 'Sequence', k);
            obj.Count = k;
        end

        function r = reader(obj)
            % READER  Read-only MemoryReader over everything recorded so far.
            r = OceanNLP.Inspect.MemoryReader(obj.Steps, obj.Artifacts);
        end

        function n = numRecords(obj)
            n = obj.Count;
        end
    end

    methods (Static)
        function b = byteCount(value, depth)
            % BYTECOUNT  Deep in-memory size of VALUE in bytes (see class help).
            %   Object/struct nesting deeper than 16 levels (e.g. handle cycles) is
            %   counted as the 8-byte reference only.
            arguments
                value
                depth (1,1) double = 0
            end
            if depth > 16 && (isstruct(value) || iscell(value) || isobject(value))
                b = 8;
                return;
            end
            if isstruct(value)
                b = 0;
                f = fieldnames(value);
                for i = 1:numel(value)
                    for j = 1:numel(f)
                        b = b + OceanNLP.Speech.Inspect.SpeechTrace.byteCount(value(i).(f{j}), depth + 1);
                    end
                end
            elseif iscell(value)
                b = 0;
                for i = 1:numel(value)
                    b = b + OceanNLP.Speech.Inspect.SpeechTrace.byteCount(value{i}, depth + 1);
                end
            elseif isobject(value) && ~isstring(value) && ~isa(value, 'OceanDL.Autograd.Tensor')
                b = 0;
                p = properties(value);
                for i = 1:numel(value)
                    for j = 1:numel(p)
                        b = b + OceanNLP.Speech.Inspect.SpeechTrace.byteCount(value(i).(p{j}), depth + 1);
                    end
                end
            elseif isa(value, 'OceanDL.Autograd.Tensor')
                b = OceanNLP.Speech.Inspect.SpeechTrace.byteCount(OceanDL.Autograd.Tensor.raw(value));
            else
                info = whos('value');
                b = info.bytes;
            end
        end
    end

    methods (Static, Access = private)
        function s = emptySteps()
            s = struct('Id', {}, 'Name', {}, 'Stage', {}, 'InputIds', {}, 'OutputIds', {}, ...
                'Summary', {}, 'Status', {}, 'Sequence', {});
        end

        function a = emptyArtifacts()
            a = struct('Id', {}, 'Kind', {}, 'Name', {}, 'Value', {}, 'Summary', {}, ...
                'Status', {}, 'ByteCount', {});
        end

        function s = formatBytes(b)
            if b < 1024
                s = sprintf("%d B", b);
            elseif b < 1024^2
                s = sprintf("%.1f KiB", b / 1024);
            else
                s = sprintf("%.2f MiB", b / 1024^2);
            end
        end
    end
end
