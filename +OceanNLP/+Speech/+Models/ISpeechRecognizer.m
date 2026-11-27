classdef (Abstract) ISpeechRecognizer < OceanDL.nn.Module
    % ISPEECHRECOGNIZER  Contract every trainable ASR model implements.
    %
    %   out = m.forward(batch)          batch: struct from Data.SpeechBatcher
    %                                     (X: F x N x T, Lengths: 1 x N, Mask: N x T, Targets ...)
    %                                   out: struct with Logits (K x N x T' Tensor) and
    %                                     Lengths (1 x N valid output steps); models may add
    %                                     fields (EncoderOutput, Attention, ...)
    %   L   = m.loss(batch)             scalar Tensor training objective; padded steps
    %                                   contribute neither loss nor gradient
    %   tr  = m.transcribe(input, ...)  input: Core.Waveform | Core.FeatureSequence | batch
    %                                   -> 1 x N Core.Transcript (Decoder and options recorded)
    %   cfg = m.config()                struct: Type, architecture hyper-parameters,
    %                                   Vocabulary (stateDict), Features (config), ModelId
    %
    %   Inherited from OceanDL.nn.Module: parameters, train/eval, zeroGrad,
    %   stateDict/loadStateDict (weights only), to(...).
    %   Training state (optimizer moments, RNG, epoch) lives in Training.*, not here.
    %   Rebuild from a saved config with Models.RecognizerFactory.fromConfig(cfg).

    properties (SetAccess = protected)
        Features            % Core.IFeatureTransform used by transcribe(Waveform)
        Vocabulary          % Data.SpeechVocabulary (owns Blank/Pad/Bos/Eos indices)
        ModelId (1,1) string = ""
    end

    methods (Abstract)
        out = forward(obj, batch)
        L = loss(obj, batch)
        tr = transcribe(obj, input, varargin)
        cfg = config(obj)
    end

    methods
        function setModelId(obj, id)
            arguments
                obj
                id (1,1) string
            end
            obj.ModelId = id;
        end
    end

    methods (Access = protected)
        function X = padToMinimumFrames(obj, X)
            % PADTOMINIMUMFRAMES  Right-pad the batch time axis with zeros until the
            %   encoders can run (outputLength(T) >= 1). Valid Lengths are unchanged, so an
            %   utterance shorter than the receptive field gets 0 output frames and an
            %   empty transcript -- exactly as when it shares a batch with a longer one
            %   (contract sec. 3: shorter than one frame is empty, not an error).
            T = size(X, 3);
            if obj.outputLength(T) >= 1
                return;
            end
            Tmin = T + 1;
            while obj.outputLength(Tmin) < 1
                Tmin = Tmin + 1;
                if Tmin > 4096
                    error('OceanNLP:Speech:Models:ISpeechRecognizer:ReceptiveField', ...
                        'Encoders produce no output frame for any input up to 4096 frames.');
                end
            end
            [F, N, ~] = size(X, 1:3);
            X = cat(3, X, zeros(F, N, Tmin - T, 'like', X));
        end
    end
end
