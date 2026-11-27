classdef RecognizerFactory
    % RECOGNIZERFACTORY  Rebuild recognisers, encoders and feature transforms from config() structs.
    %
    %   m = OceanNLP.Speech.Models.RecognizerFactory.fromConfig(cfg)           % cfg.Type dispatch
    %   e = OceanNLP.Speech.Models.RecognizerFactory.encoderFromConfig(cfg)
    %   f = OceanNLP.Speech.Models.RecognizerFactory.featuresFromConfig(cfg)
    %   t = OceanNLP.Speech.Models.RecognizerFactory.types("recognizer"|"encoder"|"feature")
    %   OceanNLP.Speech.Models.RecognizerFactory.register(kind, type, @builder)   % runtime entry
    %
    %   fromConfig rebuilds the ARCHITECTURE, feature transform and vocabulary with
    %   fresh weights; load trained weights with m.loadStateDict(sd) (or use
    %   Training.SpeechCheckpoint.load, which does both). An unknown Type raises
    %   'OceanNLP:Speech:Models:RecognizerFactory:UnknownType' listing the known ones.
    %
    %   EXTENDING (how B5/B6 and later builders add a type):
    %     1. Give the class a static fromConfig(cfg) and a config() whose Type field
    %        names it (and that embeds everything fromConfig needs).
    %     2. Add ONE line to the matching builtin table below, e.g. in
    %        builtinRecognizers():  r.EncoderDecoderRecognizer = @OceanNLP.Speech.Models.EncoderDecoderRecognizer.fromConfig;
    %        in builtinEncoders():  r.ConformerEncoder = @OceanNLP.Speech.Models.Encoders.ConformerEncoder.fromConfig;
    %                               r.HuBERTModel     = @OceanNLP.Speech.Models.Pretrain.HuBERTModel.fromConfig;
    %     (Or, without touching this file -- e.g. for experiments -- call
    %     register("encoder", "MyEncoder", @MyEncoder.fromConfig) once per session;
    %     runtime entries shadow builtins and live until `clear classes`.)
    %   Encoder contract (consumed by CTCRecognizer): an OceanDL.nn.Module with
    %   InputDim, OutputDim, [H, lens] = forward(X, lens), outputLength(lens), config().
    %   Feature types dispatch to each transform's own static fromConfig in
    %   OceanNLP.Speech.Features (LogMel, MFCC, DeltaFeatures, EnergyContour,
    %   PitchContour, FeaturePipeline); each rebuilt transform's config() equals the
    %   saved one (tested).

    methods (Static)
        function m = fromConfig(cfg)
            arguments
                cfg (1,1) struct
            end
            fn = OceanNLP.Speech.Models.RecognizerFactory.lookup("recognizer", cfg);
            m = fn(cfg);
        end

        function e = encoderFromConfig(cfg)
            arguments
                cfg (1,1) struct
            end
            fn = OceanNLP.Speech.Models.RecognizerFactory.lookup("encoder", cfg);
            e = fn(cfg);
        end

        function f = featuresFromConfig(cfg)
            arguments
                cfg (1,1) struct
            end
            fn = OceanNLP.Speech.Models.RecognizerFactory.lookup("feature", cfg);
            f = fn(cfg);
        end

        function register(kind, type, builder)
            arguments
                kind (1,1) string {mustBeMember(kind, ["recognizer", "encoder", "feature"])}
                type (1,1) string {mustBeNonzeroLengthText}
                builder (1,1) function_handle
            end
            OceanNLP.Speech.Models.RecognizerFactory.runtime(kind, type, builder);
        end

        function t = types(kind)
            arguments
                kind (1,1) string {mustBeMember(kind, ["recognizer", "encoder", "feature"])}
            end
            t = sort(string(fieldnames(OceanNLP.Speech.Models.RecognizerFactory.table(kind))))';
        end

        % ---------------- builtin tables (add one line per new type) ----------------
        function r = builtinRecognizers()
            r = struct();
            r.CTCRecognizer = @OceanNLP.Speech.Models.CTCRecognizer.fromConfig;
            r.EncoderDecoderRecognizer = @OceanNLP.Speech.Models.EncoderDecoderRecognizer.fromConfig;
        end

        function r = builtinEncoders()
            r = struct();
            r.ConvEncoder = @OceanNLP.Speech.Models.Encoders.ConvEncoder.fromConfig;
            r.RecurrentEncoder = @OceanNLP.Speech.Models.Encoders.RecurrentEncoder.fromConfig;
            r.HuBERTModel = @OceanNLP.Speech.Models.Pretrain.HuBERTModel.fromConfig;
            r.ConformerEncoder = @OceanNLP.Speech.Models.Encoders.ConformerEncoder.fromConfig;
        end

        function r = builtinFeatures()
            r = struct();
            r.LogMel = @OceanNLP.Speech.Features.LogMel.fromConfig;
            r.MFCC = @OceanNLP.Speech.Features.MFCC.fromConfig;
            r.DeltaFeatures = @OceanNLP.Speech.Features.DeltaFeatures.fromConfig;
            r.EnergyContour = @OceanNLP.Speech.Features.EnergyContour.fromConfig;
            r.PitchContour = @OceanNLP.Speech.Features.PitchContour.fromConfig;
            r.FeaturePipeline = @(c) OceanNLP.Speech.Features.FeaturePipeline.fromConfig(c, ...
                'StageBuilder', @OceanNLP.Speech.Models.RecognizerFactory.featuresFromConfig);
        end
    end

    methods (Static, Access = private)
        function fn = lookup(kind, cfg)
            if ~isfield(cfg, 'Type')
                error('OceanNLP:Speech:Models:RecognizerFactory:NoType', 'A %s config needs a Type field.', kind);
            end
            type = string(cfg.Type);
            t = OceanNLP.Speech.Models.RecognizerFactory.table(kind);
            if ~isfield(t, type)
                error('OceanNLP:Speech:Models:RecognizerFactory:UnknownType', ...
                    'Unknown %s type "%s". Known: %s.', kind, type, strjoin(string(fieldnames(t)), ', '));
            end
            fn = t.(type);
        end

        function t = table(kind)
            switch kind
                case "recognizer", t = OceanNLP.Speech.Models.RecognizerFactory.builtinRecognizers();
                case "encoder",    t = OceanNLP.Speech.Models.RecognizerFactory.builtinEncoders();
                otherwise,         t = OceanNLP.Speech.Models.RecognizerFactory.builtinFeatures();
            end
            extra = OceanNLP.Speech.Models.RecognizerFactory.runtime(kind);
            for f = string(fieldnames(extra))'
                t.(f) = extra.(f);
            end
        end

        function out = runtime(kind, type, builder)
            % Runtime registrations (persistent per MATLAB session).
            persistent reg
            if isempty(reg)
                reg = struct('recognizer', struct(), 'encoder', struct(), 'feature', struct());
            end
            if nargin == 3
                reg.(kind).(type) = builder;
            end
            out = reg.(kind);
        end
    end
end
