classdef PipelineResult < handle
    % PIPELINERESULT  Immutable record of one RecognitionPipeline.run.
    %
    %   Fields (frozen API, Speech Pipeline Studio plan sec. 4.1):
    %     Input              the Waveform as received (after toWaveform; before conditioning)
    %     Conditioned        AudioConditioner output (mono, model rate)
    %     Report             AudioConditioner report struct
    %     Segments           1 x J struct Start, End (s, on Conditioned) from the detector;
    %                        empty when the detector is off
    %     SegmentWaveforms   1 x P Waveform: what was actually transcribed (P = 1 when not
    %                        segmented: the whole conditioned waveform)
    %     Transcript         merged Core.Transcript; Segments(p) = Text + StartTime/EndTime
    %                        of the audio piece p (detector times, never token timing)
    %     SegmentTranscripts 1 x P Core.Transcript, recognizer's own output per piece
    %     Readers            1 x P cell of OceanNLP.Inspect.MemoryReader ({} when no trace)
    %     Evaluation         CorpusEvaluator.evaluate result ([] when no Reference)
    %     Timings            struct, seconds: Input, Conditioning, Detection, Segmentation,
    %                        Recognition, Evaluation, Total, PerSegment (1 x P)
    %     Warnings           string row: conditioner + detector + pipeline warnings
    %     ModelId            recognizer.ModelId
    %     DecodeOptions      struct of the decode options passed through to transcribe
    %   Additions (beyond sec. 4.1):
    %     DetectorInfo       detector info struct (FrameTimes, Energy, Threshold, ...) or []
    %     Segmented          true when the pieces are detector segments
    %     PieceTimes         P x 2 [start end] seconds of each transcribed piece on Conditioned
    %     Reference          the reference text ("" when none)
    %     ModelClass         class name of the recognizer

    properties (SetAccess = immutable)
        Input
        Conditioned
        Report struct
        Segments struct
        SegmentWaveforms
        Transcript
        SegmentTranscripts
        Readers cell
        Evaluation
        Timings struct
        Warnings (1,:) string
        ModelId (1,1) string
        DecodeOptions struct
        DetectorInfo
        Segmented (1,1) logical
        PieceTimes double
        Reference (1,1) string
        ModelClass (1,1) string
    end

    methods
        function obj = PipelineResult(s)
            arguments
                s (1,1) struct
            end
            names = properties(obj);
            for k = 1:numel(names)
                n = names{k};
                if ~isfield(s, n)
                    error('OceanNLP:Speech:Inference:PipelineResult:MissingField', 'PipelineResult needs field %s.', n);
                end
                obj.(n) = s.(n);
            end
        end

        function n = numPieces(obj)
            % NUMPIECES  Number of transcribed audio pieces.
            n = numel(obj.SegmentTranscripts);
        end

        function r = reader(obj, k)
            % READER  MemoryReader of piece k (default 1); error when traced off.
            arguments
                obj
                k (1,1) double {mustBeInteger, mustBePositive} = 1
            end
            if k > numel(obj.Readers)
                error('OceanNLP:Speech:Inference:PipelineResult:NoTrace', ...
                    'No trace reader for piece %d (run with ''Trace'', true).', k);
            end
            r = obj.Readers{k};
        end
    end
end
