classdef RegexTokenizer < OceanNLP.Tokenize.ITokenizer
    % REGEXTOKENIZER  Tokenize by extracting all matches of a custom regex.
    %
    % THEORY:
    %   Generalizes WhitespaceTokenizer: instead of splitting on
    %   whitespace and cleaning up the edges, a token is defined
    %   POSITIVELY as any substring matching Pattern (via MATLAB's
    %   regexp(...,'match')). This is the strategy to reach for when the
    %   whitespace+strip heuristic isn't expressive enough — e.g. token
    %   patterns for hashtags, URLs, numbers-with-units, or CJK text
    %   with no whitespace at all.
    %
    % POLICY (read before using):
    %   1. DEFAULT PATTERN — '\w+' (one-or-more word characters: letters,
    %      digits, underscore). This naturally strips ALL punctuation,
    %      because non-word characters are simply never part of a match —
    %      there is no separate "strip" step like WhitespaceTokenizer's.
    %      IMPORTANT CONSEQUENCE: apostrophes are not word characters, so
    %      contractions SPLIT into two tokens under the default pattern:
    %          "isn't" -> ["isn" "t"]     (NOT ["isn't"])
    %          "don't" -> ["don" "t"]
    %      This differs from WhitespaceTokenizer, which keeps contractions
    %      intact. If contraction-preserving behavior is wanted, supply a
    %      custom pattern that includes the apostrophe, e.g.:
    %          RegexTokenizer('Pattern', "[\w']+")
    %      Choose the default deliberately: '\w+' is the simplest, most
    %      predictable "word run" pattern and matches the common case
    %      (sklearn's CountVectorizer default token pattern is similarly
    %      a bare word-character run, '(?u)\b\w\w+\b').
    %   2. CASE FOLDING — lowercases input by default (CaseSensitive=false),
    %      applied to the WHOLE input string before matching, for the same
    %      vocabulary-collision rationale as WhitespaceTokenizer. Pass
    %      'CaseSensitive', true to preserve case.
    %   3. EMPTY / NO-MATCH INPUT — if Pattern has zero matches (empty
    %      input, or input that contains no matching substrings at all,
    %      e.g. pure punctuation against the default '\w+' pattern),
    %      returns a (1x0) empty string array. Never errors.
    %
    % USAGE:
    %   tok    = OceanNLP.Tokenize.RegexTokenizer();               % default '\w+'
    %   tokens = tok.forward("Hello, world! Isn't MATLAB great?");
    %   % tokens = ["hello" "world" "isn" "t" "matlab" "great"]
    %
    %   tokAp  = OceanNLP.Tokenize.RegexTokenizer('Pattern', "[\w']+");
    %   tokens = tokAp.forward("Isn't MATLAB great?");
    %   % tokens = ["isn't" "matlab" "great"]
    %
    %   empty  = tok.forward("!!! ...");
    %   % empty = strings(1,0)
    %
    % See also: OceanNLP.Tokenize.ITokenizer, OceanNLP.Tokenize.WhitespaceTokenizer,
    %           OceanNLP.Tokenize.TokenizerFactory
    %
    % 3-PILLAR CONTRACT: implements getMetadata()/getPipelineSpec() per
    % Planning/NLP/Dual_Representation_Addendum.md; honestly non-symbolic
    % (IsSymbolic=false, R7) — tokenization has no closed-form phi.

    properties
        Pattern       (1,1) string  = "\w+"   % regexp pattern; matches become tokens
        CaseSensitive (1,1) logical = false   % true preserves original case
    end

    methods
        function obj = RegexTokenizer(options)
            % REGEXTOKENIZER  Construct with an optional custom pattern.
            %   tok = RegexTokenizer('Pattern', "[\w']+", 'CaseSensitive', true)
            arguments
                options.Pattern       (1,1) string  = "\w+"
                options.CaseSensitive (1,1) logical = false
            end
            % Fail loudly at construction time on an invalid pattern,
            % rather than deferring the error to the first forward() call.
            try
                regexp("", options.Pattern, 'match');
            catch cause
                err = MException('OceanNLP:Tokenize:RegexTokenizer:InvalidPattern', ...
                    'Pattern "%s" is not a valid regular expression.', options.Pattern);
                err = addCause(err, cause);
                throw(err);
            end
            obj.Pattern       = options.Pattern;
            obj.CaseSensitive = options.CaseSensitive;
        end

        function tokens = forward(obj, rawText)
            % FORWARD  Tokenize one document by extracting regex matches.
            %   tokens = tok.forward(rawText) -> (1 x T) string array.
            %   rawText — char or string scalar (one document).
            arguments
                obj
                rawText (1,1) string
            end

            txt = rawText;
            if ~obj.CaseSensitive
                txt = lower(txt);
            end

            matches = regexp(txt, obj.Pattern, 'match');
            tokens  = reshape(string(matches), 1, []);
        end

        function s = name(~)
            s = "regex";
        end

        function meta = getMetadata(~)
            % GETMETADATA  Pillar 1 (symbolic) — honest-opacity stub (R7).
            % Regex tokenization is a discrete string transform, not a
            % closed-form numeric map: no SymbolicLatex is fabricated for it.
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            meta.Assumptions = {'case-folding applied by default (CaseSensitive=false lowercases the whole input before matching; pass CaseSensitive=true to preserve case)', ...
                                 'default Pattern ''\w+'' strips all punctuation and splits contractions into separate tokens (e.g. "isn''t" -> ["isn" "t"]); supply a custom Pattern (e.g. "[\w'']+") to preserve them'};
            meta.IsSymbolic = false;
            meta.Reference = "discrete tokenization procedure, not a closed-form transform";
        end

        function spec = getPipelineSpec(~)
            % GETPIPELINESPEC  Pillar 3 (diagram) — port signature and canvas
            % metadata for this NONVIRTUAL leaf operator.
            spec = struct();
            spec.Inputs = struct('Name', "rawText", 'Domain', "raw-text", 'Dtype', "string", 'Shape', "1 x 1 (scalar document)");
            spec.Outputs = struct('Name', "tokens", 'Domain', "token-sequence", 'Dtype', "string", 'Shape', "1 x T");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "tokenize");
            spec.ComposesFrom = {};
        end
    end
end
