classdef Transcript < handle
    % TRANSCRIPT  A recogniser's output for one utterance, with how it was produced.
    %
    %   tr = OceanNLP.Speech.Core.Transcript(text, 'TokenIds', ids, 'Decoder', "ctc-beam", ...
    %          'DecoderOptions', opts, 'ModelId', id, 'Score', logScore, 'Candidates', nbest)
    %
    %   Honesty rules (Phase 0 contract sec. 4):
    %     Segments    timed word/token segments ONLY when a real alignment
    %                 produced them; CTC/attention decoding alone does not, so it
    %                 stays empty -- never invented from frame positions.
    %     Language    "" unless supplied by the caller or actually inferred.
    %     Score       the decoder's own log-score; NaN when the decoder has none.
    %                 It is not a calibrated confidence.
    %     Candidates  N-best struct array (Text, TokenIds, Score), may be empty.

    properties (SetAccess = immutable)
        Text (1,1) string
        RawText (1,1) string
        TokenIds (1,:) double
        Segments struct
        Language (1,1) string
        Decoder (1,1) string
        DecoderOptions struct
        ModelId (1,1) string
        Score (1,1) double
        Candidates struct
    end

    methods
        function obj = Transcript(text, opts)
            arguments
                text (1,1) string
                opts.RawText (1,1) string = missing
                opts.TokenIds (1,:) double = []
                opts.Segments struct = struct('Text', {}, 'StartTime', {}, 'EndTime', {})
                opts.Language (1,1) string = ""
                opts.Decoder (1,1) string = ""
                opts.DecoderOptions struct = struct()
                opts.ModelId (1,1) string = ""
                opts.Score (1,1) double = NaN
                opts.Candidates struct = struct('Text', {}, 'TokenIds', {}, 'Score', {})
            end
            if ~isempty(opts.Segments) && ~all(isfield(opts.Segments, {'Text', 'StartTime', 'EndTime'}))
                error('OceanNLP:Speech:Core:Transcript:Segments', 'Segments need Text, StartTime, EndTime.');
            end
            obj.Text = text;
            obj.RawText = opts.RawText;
            % missing means "caller did not supply a distinct raw rendering";
            % an explicitly supplied empty string remains empty.
            if ismissing(obj.RawText), obj.RawText = text; end
            obj.TokenIds = opts.TokenIds;
            obj.Segments = opts.Segments;
            obj.Language = opts.Language;
            obj.Decoder = opts.Decoder;
            obj.DecoderOptions = opts.DecoderOptions;
            obj.ModelId = opts.ModelId;
            obj.Score = opts.Score;
            obj.Candidates = opts.Candidates;
        end

        function s = texts(objs)
            % TEXTS  String array of the Text of a Transcript array.
            s = arrayfun(@(o) o.Text, objs);
        end
    end
end
