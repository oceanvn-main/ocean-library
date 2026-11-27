classdef ModelCatalog < handle
    % MODELCATALOG  Discover speech checkpoints, describe them without loading weights, load lazily.
    %
    %   cat = OceanNLP.Speech.Inference.ModelCatalog('Root', <repo>/Results/Speech/checkpoints)
    %   T   = cat.list()                      recognisers only (one row per loadable checkpoint)
    %   T   = cat.list('IncludeSkipped', true) also the .mat files that are not recognisers,
    %                                         with the reason in column Reason
    %   m   = cat.load(name)                  recogniser (SpeechCheckpoint.load), cached
    %   h   = cat.history(name)               trainer History struct (Step* curves + Epochs)
    %   cat.clearCache();  tf = cat.isLoaded(name)
    %
    %   DISCOVERY: every *.mat under Root (recursively). Name = path relative to Root
    %   without extension, "/"-separated (e.g. "ctc_fsdd_best", "hubert_ft/ctc_it1_seed1").
    %   The variables of each file are listed with whos('-file') (no data read), then:
    %     Format "OceanSpeechCheckpoint" + a recogniser ModelConfig.Type -> loadable row;
    %     Format "OceanHuBERTPretrain"   -> skipped: self-supervised encoder pretraining
    %                                       (no recognition head, not a recogniser);
    %     anything else                  -> skipped: not an Ocean speech checkpoint
    %                                       (e.g. hubert_ft/ft_*.mat experiment summaries).
    %
    %   COLUMNS: Name, File, Type, ModelId, Units, SampleRate, NumParams, BestEpoch,
    %   BestValWER (frozen API) + Loadable, Reason, Encoders, Created, FileMB.
    %     Type/ModelId/Units/SampleRate/Encoders  from ModelConfig (a small MAT variable)
    %     NumParams   the architecture's parameter count: the model is rebuilt from its
    %                 config with RecognizerFactory.fromConfig (fresh weights, global RNG
    %                 untouched) and numParameters() summed -- the WeightValues variable is
    %                 never read (tested equal to the loaded model's count)
    %     BestEpoch, BestValWER  TrainerState.BestEpoch and BestScore(1) (SpeechTrainer's
    %                 model-selection WER on the validation split). TrainerState is one MAT
    %                 variable, so it is read whole; for checkpoints saved with BestWeights it
    %                 contains a copy of weights that is discarded (MAT v7 cannot read part of
    %                 a struct). NaN when absent.
    %   Rows are cached per file modification time; list('Refresh', true) re-reads.

    properties (SetAccess = immutable)
        Root (1,1) string
    end

    properties (Access = private)
        Models                 % containers.Map name -> recogniser
        Rows                   % containers.Map file -> struct(Date, Row)
        Histories              % containers.Map name -> history struct
        ParamCounts            % containers.Map architecture key -> NumParams
    end

    methods
        function obj = ModelCatalog(opts)
            arguments
                opts.Root (1,1) string = OceanNLP.Speech.Inference.ModelCatalog.defaultRoot()
            end
            root = opts.Root;
            if isfolder(root)                       % canonical absolute form (no '..'), as dir() reports
                d = dir(root);
                root = string(d(1).folder);
            end
            obj.Root = root;
            obj.Models = containers.Map('KeyType', 'char', 'ValueType', 'any');
            obj.Rows = containers.Map('KeyType', 'char', 'ValueType', 'any');
            obj.Histories = containers.Map('KeyType', 'char', 'ValueType', 'any');
            obj.ParamCounts = containers.Map('KeyType', 'char', 'ValueType', 'double');
        end

        function T = list(obj, opts)
            arguments
                obj
                opts.IncludeSkipped (1,1) logical = false
                opts.Refresh (1,1) logical = false
            end
            if ~isfolder(obj.Root)
                error('OceanNLP:Speech:Inference:ModelCatalog:NoRoot', 'Checkpoint folder "%s" does not exist.', obj.Root);
            end
            if opts.Refresh
                remove(obj.Rows, keys(obj.Rows));
            end
            files = dir(fullfile(obj.Root, '**', '*.mat'));
            rows = cell(numel(files), 1);
            for k = 1:numel(files)
                f = fullfile(files(k).folder, files(k).name);
                key = char(f);
                if isKey(obj.Rows, key) && obj.Rows(key).Date == files(k).datenum
                    rows{k} = obj.Rows(key).Row;
                else
                    try
                        rows{k} = obj.describe(f, files(k));
                    catch err                    % one unreadable/corrupt .mat must not break the catalog
                        rows{k} = obj.unreadableRow(f, files(k), err);
                    end
                    obj.Rows(key) = struct('Date', files(k).datenum, 'Row', rows{k});
                end
            end
            if isempty(rows)
                T = obj.emptyTable();
                return;
            end
            T = struct2table(vertcat(rows{:}), 'AsArray', true);
            T = sortrows(T, 'Name');
            if ~opts.IncludeSkipped
                T = T(T.Loadable, :);
            end
        end

        function m = load(obj, name)
            % LOAD  The recogniser of checkpoint NAME (cached after the first load).
            arguments
                obj
                name (1,1) string
            end
            key = char(name);
            if isKey(obj.Models, key)
                m = obj.Models(key);
                return;
            end
            row = obj.rowOf(name);
            if ~row.Loadable
                error('OceanNLP:Speech:Inference:ModelCatalog:NotRecognizer', '"%s" is not a recogniser: %s', name, row.Reason);
            end
            ck = OceanNLP.Speech.Training.SpeechCheckpoint.load(row.File);
            m = ck.Model;
            obj.Models(key) = m;
        end

        function h = history(obj, name)
            % HISTORY  Training history of NAME: TrainerState.History for recognisers, the
            %   pretraining History for HuBERT pretraining files, struct('Epochs', ...) for
            %   experiment summaries that store per-epoch rows.
            arguments
                obj
                name (1,1) string
            end
            key = char(name);
            if isKey(obj.Histories, key)
                h = obj.Histories(key);
                return;
            end
            row = obj.rowOf(name);
            vars = string({whos('-file', row.File).name});
            if row.Loadable
                s = load(row.File, 'TrainerState');
                if ~isfield(s.TrainerState, 'History')
                    error('OceanNLP:Speech:Inference:ModelCatalog:NoHistory', '"%s" has no trainer history.', name);
                end
                h = s.TrainerState.History;
            elseif any(vars == "History")
                h = load(row.File, 'History').History;
            elseif any(vars == "Epochs")
                h = struct('Epochs', load(row.File, 'Epochs').Epochs);
            else
                error('OceanNLP:Speech:Inference:ModelCatalog:NoHistory', '"%s" has no training history.', name);
            end
            obj.Histories(key) = h;
        end

        function tf = isLoaded(obj, name)
            arguments
                obj
                name (1,1) string
            end
            tf = isKey(obj.Models, char(name));
        end

        function clearCache(obj)
            remove(obj.Models, keys(obj.Models));
            remove(obj.Histories, keys(obj.Histories));
        end
    end

    methods (Static)
        function r = defaultRoot()
            % DEFAULTROOT  <repo>/Results/Speech/checkpoints, derived from this file's location.
            here = fileparts(mfilename('fullpath'));            % .../+OceanNLP/+Speech/+Inference
            repo = fileparts(fileparts(fileparts(here)));
            r = string(fullfile(repo, 'Results', 'Speech', 'checkpoints'));
        end
    end

    methods (Access = private)
        function row = rowOf(obj, name)
            T = obj.list('IncludeSkipped', true);
            k = find(T.Name == name, 1);
            if isempty(k)
                error('OceanNLP:Speech:Inference:ModelCatalog:UnknownName', 'No checkpoint named "%s" under %s.', name, obj.Root);
            end
            row = table2struct(T(k, :));
        end

        function row = unreadableRow(obj, f, d, err)
            % UNREADABLEROW  Listed (with IncludeSkipped) but never loadable; the reason names the error.
            rel = extractAfter(string(f), strlength(obj.Root));
            rel = regexprep(replace(rel, '\', '/'), '^/+', '');
            row = struct('Name', regexprep(rel, '\.mat$', ''), 'File', string(f), 'Type', "unreadable", ...
                'ModelId', "", 'Units', "", 'SampleRate', NaN, 'NumParams', NaN, 'BestEpoch', NaN, ...
                'BestValWER', NaN, 'Loadable', false, 'Reason', "unreadable file: " + string(err.message), ...
                'Encoders', "", 'Created', "", 'FileMB', d.bytes / 2^20);
        end

        function row = describe(obj, f, d)
            rel = extractAfter(string(f), strlength(obj.Root));
            rel = regexprep(replace(rel, '\', '/'), '^/+', '');
            name = regexprep(rel, '\.mat$', '');
            row = struct('Name', name, 'File', string(f), 'Type', "", 'ModelId', "", 'Units', "", ...
                'SampleRate', NaN, 'NumParams', NaN, 'BestEpoch', NaN, 'BestValWER', NaN, ...
                'Loadable', false, 'Reason', "", 'Encoders', "", 'Created', "", 'FileMB', d.bytes / 2^20);
            vars = string({whos('-file', f).name});
            if ~all(ismember(["Format", "ModelConfig"], vars))
                row.Type = "other";
                row.Reason = "not an Ocean speech checkpoint (variables: " + strjoin(vars, ", ") + ")";
                return;
            end
            s = load(f, 'Format', 'ModelConfig');
            fmt = string(s.Format);
            cfg = s.ModelConfig;
            if isfield(cfg, 'Type'), row.Type = string(cfg.Type); end
            if any(vars == "Created"), row.Created = string(load(f, 'Created').Created); end
            if fmt == "OceanHuBERTPretrain"
                row.Reason = "HuBERT self-supervised pretraining checkpoint (encoder only, no recognition head): not a recogniser";
                if isfield(cfg, 'InputDim'), row.Encoders = "HuBERT"; end
                return;
            end
            if fmt ~= OceanNLP.Speech.Training.SpeechCheckpoint.Format
                row.Reason = "unknown checkpoint format """ + fmt + """";
                return;
            end
            if ~ismember(row.Type, ["CTCRecognizer", "EncoderDecoderRecognizer"])
                row.Reason = "model type """ + row.Type + """ is not a recogniser";
                return;
            end
            row.ModelId = string(cfg.ModelId);
            row.Units = string(cfg.Vocabulary.Unit);
            row.SampleRate = featureRate(cfg.Features);
            encs = cfg.Encoders;
            if isstruct(encs), encs = num2cell(encs); end
            row.Encoders = strjoin(cellfun(@(e) string(e.Type), encs), "+");
            arch = jsonencode(rmfield(cfg, intersect(fieldnames(cfg), {'ModelId', 'Seed'})));   % same architecture -> same count
            if isKey(obj.ParamCounts, arch)
                row.NumParams = obj.ParamCounts(arch);
            else
                model = OceanNLP.Speech.Models.RecognizerFactory.fromConfig(cfg);
                row.NumParams = model.numParameters();
                obj.ParamCounts(arch) = row.NumParams;
            end
            if any(vars == "TrainerState")
                ts = load(f, 'TrainerState').TrainerState;
                if isfield(ts, 'BestEpoch'), row.BestEpoch = double(ts.BestEpoch); end
                if isfield(ts, 'BestScore') && ~isempty(ts.BestScore) && isfinite(ts.BestScore(1))
                    row.BestValWER = double(ts.BestScore(1));
                end
            end
            row.Loadable = true;
        end

        function T = emptyTable(~)
            T = table('Size', [0 14], 'VariableTypes', ["string", "string", "string", "string", "string", ...
                "double", "double", "double", "double", "logical", "string", "string", "string", "double"], ...
                'VariableNames', ["Name", "File", "Type", "ModelId", "Units", "SampleRate", "NumParams", ...
                "BestEpoch", "BestValWER", "Loadable", "Reason", "Encoders", "Created", "FileMB"]);
        end
    end
end

function fs = featureRate(c)
% Sample rate recorded in a feature config (LogMel / MFCC / FeaturePipeline).
fs = NaN;
if isfield(c, 'SampleRate')
    fs = double(c.SampleRate);
elseif isfield(c, 'LogMel')
    fs = featureRate(c.LogMel);
elseif isfield(c, 'Stages') && ~isempty(c.Stages)
    st = c.Stages;
    if iscell(st), st = st{1}; else, st = st(1); end
    fs = featureRate(st);
end
end
