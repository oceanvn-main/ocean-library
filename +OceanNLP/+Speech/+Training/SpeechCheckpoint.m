classdef SpeechCheckpoint
    % SPEECHCHECKPOINT  Save / load a recogniser with its full training state (MAT file).
    %
    %   OceanNLP.Speech.Training.SpeechCheckpoint.save(file, model, trainerState, 'Extra', s)
    %   ck = OceanNLP.Speech.Training.SpeechCheckpoint.load(file)
    %     ck.Model         recogniser rebuilt with Models.RecognizerFactory.fromConfig(ModelConfig)
    %                      and the saved weights loaded (strict, shape-checked, bit-exact)
    %     ck.ModelConfig   model.config(): Type, architecture, Vocabulary stateDict,
    %                      Features config, ModelId
    %     ck.Vocabulary    ck.Model.Vocabulary;   ck.Features  ck.Model.Features
    %     ck.TrainerState  SpeechTrainer.state() at save time (struct() if none):
    %                      Step, Epoch, NextBatch, global RNG state, batcher config,
    %                      history, best score, trainer options, and the per-parameter
    %                      AdamW optimizer objects (moments M, V and step count)
    %     ck.Extra         caller metadata (e.g. dataset audit, experiment config)
    %
    %   FILE LAYOUT (MAT v7): Format "OceanSpeechCheckpoint", Version, Created,
    %   MatlabRelease, ModelConfig, WeightNames (cellstr), WeightValues (cell, the
    %   exact arrays of model.stateDict()), TrainerState, Extra.
    %   Optimizer moments are stored by serialising the optimizer objects themselves:
    %   OceanStats.Optim.AdamWOptimizer keeps M_/V_ private with no accessor, and MAT
    %   serialisation preserves every property bit for bit. A rebuilt model and
    %   restored optimizers therefore continue training exactly as the original
    %   would (Training.SpeechTrainer exact-resume test).

    properties (Constant)
        Format = "OceanSpeechCheckpoint"
        Version = 1
    end

    methods (Static)
        function save(file, model, trainerState, opts)
            arguments
                file (1,1) string
                model (1,1) OceanNLP.Speech.Models.ISpeechRecognizer
                trainerState struct = struct()
                opts.Extra struct = struct()
            end
            sd = model.stateDict();
            names = keys(sd);
            vals = cell(size(names));
            for k = 1:numel(names)
                vals{k} = gather(sd(names{k}));
            end
            ck = struct();
            ck.Format = OceanNLP.Speech.Training.SpeechCheckpoint.Format;
            ck.Version = OceanNLP.Speech.Training.SpeechCheckpoint.Version;
            ck.Created = string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
            ck.MatlabRelease = string(version('-release'));
            ck.ModelConfig = model.config();
            ck.WeightNames = names;
            ck.WeightValues = vals;
            ck.TrainerState = trainerState;
            ck.Extra = opts.Extra;
            folder = fileparts(file);
            if strlength(folder) > 0 && ~isfolder(folder)
                mkdir(folder);
            end
            save(file, '-struct', 'ck', '-v7');
        end

        function ck = load(file)
            arguments
                file (1,1) string
            end
            if ~isfile(file)
                error('OceanNLP:Speech:Training:SpeechCheckpoint:NoFile', 'Checkpoint "%s" does not exist.', file);
            end
            raw = load(file);
            if ~isfield(raw, 'Format') || string(raw.Format) ~= OceanNLP.Speech.Training.SpeechCheckpoint.Format
                error('OceanNLP:Speech:Training:SpeechCheckpoint:Format', '"%s" is not an Ocean speech checkpoint.', file);
            end
            if raw.Version > OceanNLP.Speech.Training.SpeechCheckpoint.Version
                error('OceanNLP:Speech:Training:SpeechCheckpoint:Version', ...
                    'Checkpoint version %d is newer than this reader (%d).', raw.Version, ...
                    OceanNLP.Speech.Training.SpeechCheckpoint.Version);
            end
            model = OceanNLP.Speech.Models.RecognizerFactory.fromConfig(raw.ModelConfig);
            sd = containers.Map(raw.WeightNames, raw.WeightValues, 'UniformValues', false);
            model.loadStateDict(sd);
            ck = struct('Model', model, 'ModelConfig', raw.ModelConfig, 'Vocabulary', model.Vocabulary, ...
                'Features', model.Features, 'TrainerState', raw.TrainerState, 'Extra', raw.Extra, ...
                'Created', raw.Created, 'File', file);
        end
    end
end
