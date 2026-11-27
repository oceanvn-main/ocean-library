classdef WhitespaceTokenizer < OceanNLP.Tokenize.ITokenizer
    % WHITESPACETOKENIZER  Split raw text into tokens on whitespace runs.
    %
    % THEORY:
    %   The simplest possible tokenization strategy: any run of whitespace
    %   (spaces, tabs, newlines) is a token boundary. No linguistic
    %   awareness — this is a baseline strategy, not a linguistically
    %   informed tokenizer (no clitic splitting, no multi-word entity
    %   detection, etc.).
    %
    % POLICY (read before using — deliberate design choices, documented
    % here rather than left as hidden implementation detail):
    %   1. CASE FOLDING — lowercases input by default (CaseSensitive=false).
    %      Rationale: downstream bag-of-words / TF-IDF / BM25 pipelines
    %      normally want "Ocean" and "ocean" to collide to one vocabulary
    %      entry. Pass 'CaseSensitive', true to preserve original case
    %      (e.g. when a downstream step is case-sensitive, such as a
    %      named-entity heuristic).
    %   2. PUNCTUATION — leading/trailing punctuation attached to a
    %      whitespace-delimited chunk is stripped:
    %          "cat,"    -> "cat"
    %          "(great)" -> "great"
    %          "world!"  -> "world"
    %      but punctuation INTERNAL to a chunk is preserved:
    %          "don't"      -> "don't"      (contraction apostrophe kept)
    %          "well-known" -> "well-known" (compound hyphen kept)
    %          "3.14"       -> "3.14"       (decimal point kept)
    %      Rationale: stripping only the edges removes sentence-boundary
    %      noise (commas, periods, quotes, brackets) without destroying
    %      contractions, hyphenated compounds, or decimal numbers, which
    %      are usually meaningful as a single token. A chunk that is PURE
    %      punctuation (e.g. "--", "...") strips to the empty string and
    %      is DROPPED — it is never emitted as a zero-length token.
    %   3. EMPTY / WHITESPACE-ONLY INPUT — returns a (1x0) empty string
    %      array. Never errors, never returns a 1x1 array holding "".
    %      A string that is only punctuation (e.g. "!!! ... ???") also
    %      reduces to a (1x0) empty array, per rule 2.
    %
    % USAGE:
    %   tok    = OceanNLP.Tokenize.WhitespaceTokenizer();
    %   tokens = tok.forward("Hello, world! Isn't MATLAB great?");
    %   % tokens = ["hello" "world" "isn't" "matlab" "great"]
    %
    %   tokCS  = OceanNLP.Tokenize.WhitespaceTokenizer('CaseSensitive', true);
    %   tokens = tokCS.forward("Hello World");
    %   % tokens = ["Hello" "World"]
    %
    %   empty  = tok.forward("   ");
    %   % empty = strings(1,0)
    %
    % See also: OceanNLP.Tokenize.ITokenizer, OceanNLP.Tokenize.RegexTokenizer,
    %           OceanNLP.Tokenize.TokenizerFactory
    %
    % 3-PILLAR CONTRACT: implements getMetadata()/getPipelineSpec() per
    % Planning/NLP/Dual_Representation_Addendum.md; honestly non-symbolic
    % (IsSymbolic=false, R7) — tokenization has no closed-form phi.

    properties
        CaseSensitive (1,1) logical = false   % true preserves original case
    end

    methods
        function obj = WhitespaceTokenizer(options)
            % WHITESPACETOKENIZER  Construct with optional case-folding flag.
            %   tok = WhitespaceTokenizer('CaseSensitive', true)
            arguments
                options.CaseSensitive (1,1) logical = false
            end
            obj.CaseSensitive = options.CaseSensitive;
        end

        function tokens = forward(obj, rawText)
            % FORWARD  Tokenize one document on whitespace.
            %   tokens = tok.forward(rawText) -> (1 x T) string array.
            %   rawText — char or string scalar (one document).
            arguments
                obj
                rawText (1,1) string
            end

            txt = strtrim(rawText);
            if txt == ""
                tokens = strings(1, 0);
                return
            end

            if ~obj.CaseSensitive
                txt = lower(txt);
            end

            % One pass: a token is a whitespace-free chunk with its leading/trailing non-word characters stripped (internal
            % punctuation - apostrophes, hyphens, decimal points - survives), i.e. a match of \w(?:\S*\w)? (starts and ends on a
            % word character, anything but whitespace in between). Identical to split-on-whitespace + strip-edges + drop-empty,
            % about 2.5x faster.
            tokens = reshape(string(regexp(txt, '\w(?:\S*\w)?', 'match')), 1, []);
        end

        function s = name(~)
            s = "whitespace";
        end

        function meta = getMetadata(~)
            % GETMETADATA  Pillar 1 (symbolic) — honest-opacity stub (R7).
            % Whitespace tokenization is a discrete string transform, not a
            % closed-form numeric map: no SymbolicLatex is fabricated for it.
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            meta.Assumptions = {'case-folding applied by default (CaseSensitive=false lowercases input; pass CaseSensitive=true to preserve case)', ...
                                 'leading/trailing punctuation on each whitespace-delimited chunk is stripped; internal punctuation (apostrophes, hyphens, decimal points) is preserved'};
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
