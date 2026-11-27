classdef TranscriptNormalizer < handle
    % TRANSCRIPTNORMALIZER  Deterministic text normalization for transcripts and scoring.
    %
    %   n = OceanNLP.Speech.Data.TranscriptNormalizer('Lowercase', true, 'StripPunctuation', true)
    %   t = n.forward(text)          string array in, same-size string array out
    %   c = n.config()               struct (Type + every option); fromConfig(c) rebuilds
    %
    %   Steps, in order (each a pure character map, no lexicon or language model):
    %     1. Lowercase          t <- lower(t)                    (MATLAB lower, Unicode-aware)
    %     2. StripPunctuation   delete every character of Unicode general category P*
    %                           (Pc Pd Ps Pe Pi Pf Po) in U+0000..U+00FF and U+2000..U+206F
    %                           -- deleted, not replaced by a space ("o'clock" -> "oclock",
    %                           "twenty-one" -> "twentyone"), exactly as jiwer.RemovePunctuation.
    %                           Symbols (category S*: $ + < = > ^ ` | ~) are kept.
    %     3. whitespace         every run of [ \t\n\r\f\v] -> one space, then strip
    %
    %   With both options on this equals the jiwer pipeline
    %   Compose([ToLowerCase(), RemovePunctuation(), RemoveWhiteSpace(replace_by_space=True),
    %   RemoveMultipleSpaces(), Strip()]) for text whose punctuation lies in the two
    %   ranges above (verified on a fixture generated with Python unicodedata/jiwer).
    %   Punctuation outside those ranges (e.g. CJK U+3000 block) is NOT removed --
    %   a documented gap, not silently approximated.

    properties (SetAccess = immutable)
        Lowercase (1,1) logical
        StripPunctuation (1,1) logical
    end

    properties (Constant)
        % [first last] code points of Unicode general category P* in the covered blocks
        % (Unicode 15 data via Python unicodedata; checked by the test fixture).
        % (u32 suffix: unsuffixed hex literals are uint8/uint16 and would saturate on concatenation)
        PunctuationRanges = double([ ...
            0x21u32 0x23u32; 0x25u32 0x2Au32; 0x2Cu32 0x2Fu32; 0x3Au32 0x3Bu32; 0x3Fu32 0x40u32; ...
            0x5Bu32 0x5Du32; 0x5Fu32 0x5Fu32; 0x7Bu32 0x7Bu32; 0x7Du32 0x7Du32; 0xA1u32 0xA1u32; ...
            0xA7u32 0xA7u32; 0xABu32 0xABu32; 0xB6u32 0xB7u32; 0xBBu32 0xBBu32; 0xBFu32 0xBFu32; ...
            0x2010u32 0x2027u32; 0x2030u32 0x2043u32; 0x2045u32 0x2051u32; 0x2053u32 0x205Eu32])
    end

    methods
        function obj = TranscriptNormalizer(opts)
            arguments
                opts.Lowercase (1,1) logical = true
                opts.StripPunctuation (1,1) logical = true
            end
            obj.Lowercase = opts.Lowercase;
            obj.StripPunctuation = opts.StripPunctuation;
        end

        function out = forward(obj, text)
            arguments
                obj
                text string
            end
            if any(ismissing(text), 'all')
                error('OceanNLP:Speech:Data:TranscriptNormalizer:Missing', 'Transcripts must not be <missing>.');
            end
            out = text;
            if obj.Lowercase
                out = lower(out);
            end
            if obj.StripPunctuation
                for k = 1:numel(out)
                    c = char(out(k));
                    out(k) = string(c(~OceanNLP.Speech.Data.TranscriptNormalizer.isPunctuation(double(c))));
                end
            end
            out = strip(regexprep(out, '\s+', ' '));
        end

        function cfg = config(obj)
            cfg = struct('Type', "TranscriptNormalizer", 'Lowercase', obj.Lowercase, ...
                'StripPunctuation', obj.StripPunctuation, ...
                'Punctuation', "Unicode P* in U+0000-00FF and U+2000-206F, deleted", ...
                'Whitespace', "runs of \s -> one space, stripped");
        end
    end

    methods (Static)
        function tf = isPunctuation(codes)
            % ISPUNCTUATION  True for UTF-16 code units inside PunctuationRanges.
            arguments
                codes double
            end
            R = OceanNLP.Speech.Data.TranscriptNormalizer.PunctuationRanges;
            tf = false(size(codes));
            for k = 1:size(R, 1)
                tf = tf | (codes >= R(k, 1) & codes <= R(k, 2));
            end
        end

        function obj = fromConfig(cfg)
            arguments
                cfg (1,1) struct
            end
            if ~isfield(cfg, 'Type') || string(cfg.Type) ~= "TranscriptNormalizer"
                error('OceanNLP:Speech:Data:TranscriptNormalizer:Config', 'Not a TranscriptNormalizer config.');
            end
            obj = OceanNLP.Speech.Data.TranscriptNormalizer('Lowercase', logical(cfg.Lowercase), ...
                'StripPunctuation', logical(cfg.StripPunctuation));
        end
    end
end
