classdef RecognitionPipeline < handle
    % RECOGNITIONPIPELINE  condition -> detect speech -> (segment) -> recognise -> merge -> evaluate.
    %
    %   p = OceanNLP.Speech.Inference.RecognitionPipeline(recognizer, 'Conditioner', c, ...
    %         'Detector', d, 'UseDetector', true, 'Segmentation', "auto")   % "auto"|"never"|"always"
    %   r = p.run(input, 'Trace', true, 'Reference', "", 'Decoder', "beam", 'BeamWidth', 8, ...)
    %       input = Core.Waveform | audio file path | struct(Samples, SampleRate[, Source])
    %       every name-value pair other than Trace / Reference / TraceArgs is a decode option
    %       passed unchanged to recognizer.transcribe (or to SpeechProcessAdapter.run, which
    %       replays transcribe stage by stage and returns the identical transcript).
    %       'TraceArgs' (addition): cell of SpeechTrace name-values (memory budgets).
    %   r is an OceanNLP.Speech.Inference.PipelineResult.
    %
    %   STAGES
    %     1. toWaveform(input); integer PCM scaled to [-1, 1).
    %     2. Conditioner.forward -> mono waveform at the recogniser's feature rate
    %        (default AudioConditioner('TargetRate', recognizer.Features.SampleRate)).
    %     3. UseDetector: Detector.detect(conditioned) -> speech segments (seconds).
    %     4. Segmentation: "never"  -> transcribe the whole conditioned waveform (one piece);
    %                      "always" -> transcribe each detected segment (needs the detector);
    %                      "auto"   -> segment when the conditioned audio is longer than the
    %                                  detector's MaxSegment, otherwise one piece.
    %        No speech detected ("auto" / "always") -> no piece is recognised, empty
    %        transcript + warning (the model is not asked to transcribe silence); "never"
    %        always transcribes the whole conditioned waveform.
    %     5. Each piece: recognizer.transcribe(piece, decodeOptions{:}) -- or, with Trace, the
    %        stage-by-stage SpeechProcessAdapter.run (tested equal), one MemoryReader per piece.
    %     6. Merge: Text = non-empty piece texts joined by one space; TokenIds = piece ids
    %        joined by the vocabulary's word-boundary token (char units); Segments(p) =
    %        (Text_p, StartTime_p, EndTime_p) where the times are the PIECE's span on the
    %        conditioned audio -- the detector's segment times (or [0, duration] for one
    %        unsegmented piece), never token timing. With one piece the merged transcript
    %        carries that piece's Score and Candidates unchanged; with several, Score = NaN
    %        and Candidates is empty (an N-best of a concatenation is not defined) -- per-piece
    %        scores stay in SegmentTranscripts.
    %     7. Reference given -> Evaluation.CorpusEvaluator (vocabulary's normaliser).
    %   Timings (s): Input, Conditioning, Detection, Segmentation, Recognition (sum of pieces),
    %   Evaluation, Total, PerSegment. Nothing here draws random numbers; the trace only
    %   observes, so Trace on/off gives identical outputs and leaves the global RNG untouched.

    properties (SetAccess = immutable)
        Recognizer
        Conditioner
        Detector
        UseDetector (1,1) logical
        Segmentation (1,1) string
    end

    methods
        function obj = RecognitionPipeline(recognizer, opts)
            arguments
                recognizer (1,1) OceanNLP.Speech.Models.ISpeechRecognizer
                opts.Conditioner = []
                opts.Detector = []
                opts.UseDetector (1,1) logical = true
                opts.Segmentation (1,1) string {mustBeMember(opts.Segmentation, ["auto", "never", "always"])} = "auto"
            end
            fs = recognizer.Features.SampleRate;
            c = opts.Conditioner;
            if isempty(c)
                c = OceanNLP.Speech.Inference.AudioConditioner('TargetRate', fs);
            elseif ~isa(c, 'OceanNLP.Speech.Inference.AudioConditioner')
                error('OceanNLP:Speech:Inference:RecognitionPipeline:ConditionerType', ...
                    'Conditioner must be an OceanNLP.Speech.Inference.AudioConditioner (got %s).', class(c));
            end
            if c.TargetRate ~= fs
                error('OceanNLP:Speech:Inference:RecognitionPipeline:RateMismatch', ...
                    'Conditioner TargetRate %g Hz differs from the model''s feature rate %g Hz.', c.TargetRate, fs);
            end
            d = opts.Detector;
            if isempty(d)
                d = OceanNLP.Speech.Inference.SpeechActivityDetector();
            elseif ~isa(d, 'OceanNLP.Speech.Inference.SpeechActivityDetector')
                error('OceanNLP:Speech:Inference:RecognitionPipeline:DetectorType', ...
                    'Detector must be an OceanNLP.Speech.Inference.SpeechActivityDetector (got %s).', class(d));
            end
            if opts.Segmentation == "always" && ~opts.UseDetector
                error('OceanNLP:Speech:Inference:RecognitionPipeline:NeedsDetector', ...
                    'Segmentation "always" needs UseDetector = true.');
            end
            obj.Recognizer = recognizer;
            obj.Conditioner = c;
            obj.Detector = d;
            obj.UseDetector = opts.UseDetector;
            obj.Segmentation = opts.Segmentation;
        end

        function r = run(obj, input, varargin)
            arguments
                obj
                input
            end
            arguments (Repeating)
                varargin
            end
            [useTrace, reference, traceArgs, decodeArgs] = parseRunArgs(varargin);
            m = obj.Recognizer;
            tAll = tic;
            tm = struct('Input', 0, 'Conditioning', 0, 'Detection', 0, 'Segmentation', 0, ...
                'Recognition', 0, 'Evaluation', 0, 'Total', 0, 'PerSegment', zeros(1, 0));

            t = tic;
            w = OceanNLP.Speech.Inference.AudioConditioner.toWaveform(input);
            tm.Input = toc(t);

            t = tic;
            [wc, report] = obj.Conditioner.forward(w);
            tm.Conditioning = toc(t);
            warns = report.Warnings;

            segs = struct('Start', {}, 'End', {});
            info = [];
            if obj.UseDetector
                t = tic;
                % the detector's absolute floor applies to the recorded level, before normalisation gain
                [segs, info] = obj.Detector.detect(wc, 'GainDB', report.GainDB);
                tm.Detection = toc(t);
                if strlength(info.Note) > 0
                    warns(end + 1) = info.Note;
                end
                if isempty(segs)
                    warns(end + 1) = "No speech detected by the speech-activity detector";
                end
            end

            t = tic;
            dur = wc.duration();
            switch obj.Segmentation
                case "never"
                    segmented = false;
                case "always"
                    segmented = true;
                otherwise
                    segmented = obj.UseDetector && (dur > obj.Detector.MaxSegment || isempty(segs));
            end
            if segmented
                pieces = obj.Detector.split(wc, segs);
                pieceTimes = [[segs.Start]', [segs.End]'];
                if numel(segs) > 1
                    warns(end + 1) = sprintf("Input %.1f s split into %d speech segments at detected pauses", dur, numel(segs));
                end
            else
                pieces = wc;
                pieceTimes = [0, dur];
                if dur > obj.Detector.MaxSegment
                    warns(end + 1) = sprintf("Long input (%.1f s > %g s) transcribed in one piece (segmentation off)", ...
                        dur, obj.Detector.MaxSegment);
                end
            end
            pieceTimes = reshape(pieceTimes, [], 2);
            tm.Segmentation = toc(t);

            P = numel(pieces);
            trs = OceanNLP.Speech.Core.Transcript.empty(1, 0);
            readers = {};
            tm.PerSegment = zeros(1, P);
            for p = 1:P
                t = tic;
                if useTrace
                    [tr, rdr] = OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(m, pieces(p), ...
                        'Trace', OceanNLP.Speech.Inspect.SpeechTrace(traceArgs{:}), decodeArgs{:});
                    readers{p} = rdr; %#ok<AGROW>
                else
                    tr = m.transcribe(pieces(p), decodeArgs{:});
                end
                trs(p) = tr;
                tm.PerSegment(p) = toc(t);
            end
            tm.Recognition = sum(tm.PerSegment);

            merged = mergeTranscripts(trs, pieceTimes, m);

            evaluation = [];
            if strlength(reference) > 0
                t = tic;
                ev = OceanNLP.Speech.Evaluation.CorpusEvaluator('Normalizer', m.Vocabulary.Normalizer);
                evaluation = ev.evaluate(reference, merged.Text);
                tm.Evaluation = toc(t);
            end
            tm.Total = toc(tAll);

            r = OceanNLP.Speech.Inference.PipelineResult(struct('Input', w, 'Conditioned', wc, ...
                'Report', report, 'Segments', segs, 'SegmentWaveforms', pieces, 'Transcript', merged, ...
                'SegmentTranscripts', trs, 'Readers', {readers}, 'Evaluation', evaluation, 'Timings', tm, ...
                'Warnings', warns, 'ModelId', m.ModelId, 'DecodeOptions', optionStruct(decodeArgs), ...
                'DetectorInfo', info, 'Segmented', segmented, 'PieceTimes', pieceTimes, 'Reference', reference, ...
                'ModelClass', string(class(m))));
        end
    end
end

function [useTrace, reference, traceArgs, decodeArgs] = parseRunArgs(args)
if mod(numel(args), 2) ~= 0
    error('OceanNLP:Speech:Inference:RecognitionPipeline:NameValue', 'Options must be name-value pairs.');
end
useTrace = false;
reference = "";
traceArgs = {};
decodeArgs = {};
for k = 1:2:numel(args)
    name = args{k};
    if ~(ischar(name) || (isstring(name) && isscalar(name)))
        error('OceanNLP:Speech:Inference:RecognitionPipeline:NameValue', 'Option names must be text.');
    end
    name = char(name);
    v = args{k + 1};
    switch lower(name)
        case 'trace'
            if ~(islogical(v) || isnumeric(v)) || ~isscalar(v)
                error('OceanNLP:Speech:Inference:RecognitionPipeline:Trace', 'Trace must be true or false.');
            end
            useTrace = logical(v);
        case 'reference'
            reference = string(v);
            if ~isscalar(reference)
                error('OceanNLP:Speech:Inference:RecognitionPipeline:Reference', 'Reference must be one string.');
            end
            if ismissing(reference), reference = ""; end
        case 'traceargs'
            if ~iscell(v)
                error('OceanNLP:Speech:Inference:RecognitionPipeline:TraceArgs', 'TraceArgs must be a cell of SpeechTrace name-values.');
            end
            traceArgs = v;
        otherwise
            decodeArgs = [decodeArgs, {name, v}]; %#ok<AGROW>
    end
end
end

function merged = mergeTranscripts(trs, pieceTimes, m)
P = numel(trs);
segs = struct('Text', {}, 'StartTime', {}, 'EndTime', {});
for p = 1:P
    segs(p) = struct('Text', trs(p).Text, 'StartTime', pieceTimes(p, 1), 'EndTime', pieceTimes(p, 2));
end
if P == 0
    merged = OceanNLP.Speech.Core.Transcript("", 'Segments', segs, 'ModelId', m.ModelId);
    return;
end
if P == 1
    t = trs(1);
    merged = OceanNLP.Speech.Core.Transcript(t.Text, 'RawText', t.RawText, 'TokenIds', t.TokenIds, ...
        'Segments', segs, 'Language', t.Language, 'Decoder', t.Decoder, 'DecoderOptions', t.DecoderOptions, ...
        'ModelId', t.ModelId, 'Score', t.Score, 'Candidates', t.Candidates);
    return;
end
texts = arrayfun(@(t) t.Text, trs);
raws = arrayfun(@(t) t.RawText, trs);
nonEmpty = strlength(texts) > 0;
v = m.Vocabulary;
sep = [];
if v.Unit == "char"
    sep = find(v.Tokens == v.WordBoundary, 1);
end
ids = [];
for p = find(nonEmpty)
    if ~isempty(ids)
        ids = [ids, sep]; %#ok<AGROW>
    end
    ids = [ids, trs(p).TokenIds]; %#ok<AGROW>
end
merged = OceanNLP.Speech.Core.Transcript(strjoin(texts(nonEmpty), " "), 'RawText', strjoin(raws(nonEmpty), " "), ...
    'TokenIds', ids, 'Segments', segs, 'Language', trs(1).Language, 'Decoder', trs(1).Decoder, ...
    'DecoderOptions', trs(1).DecoderOptions, 'ModelId', trs(1).ModelId, 'Score', NaN);
end

function s = optionStruct(args)
s = struct();
for k = 1:2:numel(args)
    s.(args{k}) = args{k + 1};
end
end
