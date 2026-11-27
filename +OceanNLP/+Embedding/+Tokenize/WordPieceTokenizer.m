classdef WordPieceTokenizer < OceanNLP.Tokenize.ITokenizer
    % WORDPIECETOKENIZER  BERT's greedy longest-match-first subword tokenizer
    % over a FIXED, pretrained vocab.txt (never fit on the user's corpus).
    %
    % POLICY — FIXED VOCABULARY, NOT A LEARNED ONE (read before using; same
    % project habit of stating policy differences plainly, per
    % WhitespaceTokenizer's case-folding-policy precedent):
    %   This is NOT OceanNLP.Vocab.Vocabulary. Vocabulary is fit(corpus) —
    %   it walks a tokenized corpus and BUILDS a token<->id mapping from
    %   whatever tokens it sees, with a reserved 1-indexed OOV slot at id 1.
    %   WordPieceTokenizer is the opposite: its vocabulary is LOADED WHOLE
    %   from a pretrained checkpoint's vocab.txt file at construction time
    %   and is never touched again — there is no fit() method here at all,
    %   because there is nothing corpus-specific to learn (same "pretrained,
    %   not learned" policy as TokenEmbedding/MultiHeadAttention's weights,
    %   master plan §3.3). Constructing two WordPieceTokenizer instances
    %   from the same vocab.txt always produces identical ids for identical
    %   input — this is a deterministic lookup table, not a statistic
    %   estimated from data.
    %
    % ALGORITHM — GREEDY LONGEST-MATCH-FIRST (maximal matching), BERT's
    % actual documented WordPiece tokenization procedure (Wu et al., 2016 /
    % Devlin et al., 2018 §3.1; Schuster & Nakajima, 2012 for the original
    % WordPiece scheme), not an approximation of it:
    %   For each whitespace-split word (LOWERCASED first, matching the
    %   `bert-base-uncased`/`bert-tiny` convention):
    %     1. If the WHOLE word is in the vocabulary, it is one token. Done.
    %     2. Otherwise, starting from the full remaining substring, find the
    %        LONGEST prefix of it that is in the vocabulary — for the FIRST
    %        piece of the word, look up the substring AS-IS; for every
    %        SUBSEQUENT piece, look up "##" + substring (WordPiece marks
    %        non-word-initial continuation pieces with a "##" prefix). That
    %        longest match becomes one token; continue matching the
    %        remainder the same way (now always with the "##" prefix, since
    %        every later piece is non-word-initial).
    %     3. If at ANY point no valid prefix (with or without "##", as
    %        appropriate for that position) can be found in the vocabulary,
    %        DISCARD all partial matches accumulated so far for this word
    %        and emit a SINGLE [UNK]-equivalent token (obj.UnkToken) for the
    %        WHOLE word — partial pieces are never emitted for a word that
    %        ultimately fails to fully decompose.
    %   Worked example (standard from BERT documentation): given a vocab
    %   containing "un", "##aff", "##able" (but NOT "unaffable" itself), the
    %   word "unaffable" tokenizes to ["un", "##aff", "##able"]. A word with
    %   no valid decomposition under the loaded vocab becomes [obj.UnkToken]
    %   (a single token, whatever id obj.UnkToken actually occupies in the
    %   loaded vocab.txt — see the ENCODE / 0-INDEXING note below, this is
    %   NOT hardcoded to BERT's usual UNK id of 100, since a hand-built test
    %   vocab.txt will not have BERT's real special-token layout).
    %
    % TWO METHODS, TWO DIFFERENT JOBS (do not conflate them):
    %   forward(rawText) — ITokenizer contract: raw text (1x1 string) ->
    %     (1 x T) string array of WordPiece subword tokens. NO [CLS]/[SEP]
    %     tokens are inserted by this method — that is explicitly NOT this
    %     class's job; it belongs to a later composition step (e.g.
    %     BertEncoder, a different agent's deliverable) that assembles a
    %     full model input sequence around raw WordPiece output.
    %   encode(tokens) — maps a (1 x T) token string array (as produced by
    %     forward(), or any other (1 x T) string array of tokens known to
    %     this vocabulary) to a (1 x T) 0-INDEXED integer id array via the
    %     FIXED vocabulary loaded at construction.
    %
    % ENCODE / 0-INDEXING — MANDATORY CONVENTION, READ BEFORE CALLING (this
    % is not optional local style; it is a hard requirement carried forward
    % from OceanNLP.Embedding.Modules.TokenEmbedding, which this class's
    % output is meant to feed directly): ids returned by encode() are
    % BERT-NATIVE 0-INDEXED — id 0 is vocab.txt's FIRST line, id 1 is its
    % second line, etc. (vocab.txt line number, 0-based). Internally,
    % vocab.txt is read into obj.Tokens as a 1-indexed MATLAB string array
    % (obj.Tokens(1) = vocab.txt line 1), so encode() subtracts exactly 1
    % from the internal 1-indexed row lookup at the single point of use,
    % and nowhere else — mirroring TokenEmbedding's "convert at the point
    % of use, nowhere else" discipline. obj.UnkToken's id is READ FROM THE
    % ACTUAL LOADED VOCAB FILE (obj.UnkId), never hardcoded, since a
    % hand-built synthetic test vocab.txt will not place [UNK] at BERT's
    % real-checkpoint position.
    %
    % USAGE:
    %   tok    = OceanNLP.Embedding.Tokenize.WordPieceTokenizer('VocabFile', "vocab.txt");
    %   tokens = tok.forward("unaffable words");
    %   ids    = tok.encode(tokens);   % 0-indexed, matches vocab.txt line numbers
    %
    %   tok2 = OceanNLP.Embedding.Tokenize.WordPieceTokenizer('VocabFile', "v.txt", 'UnkToken', "<unk>");
    %
    % See also: OceanNLP.Tokenize.ITokenizer, OceanNLP.Embedding.Modules.TokenEmbedding,
    %           OceanNLP.Vocab.Vocabulary (the DIFFERENT, fit()-on-corpus vocabulary class)
    %
    % 3-PILLAR CONTRACT: implements getMetadata()/getPipelineSpec() per
    % Planning/NLP/Dual_Representation_Addendum.md; honestly non-symbolic
    % (IsSymbolic=false, R7, same reasoning as WhitespaceTokenizer/
    % RegexTokenizer) — subword tokenization, even greedy longest-match,
    % has no closed-form phi.

    % BASICTOKENIZE LIMITS (deliberate, disclosed): with BasicTokenize=true only
    % ASCII punctuation is split. NOT implemented vs. BERT's BasicTokenizer:
    % accent stripping ("cafe" vs "café" -> [UNK]), Unicode punctuation
    % (curly quotes, em-dash), CJK spacing, control/zero-width character
    % removal, and never-split special tokens ("[MASK]" in text is split).
    % Cranfield is pure ASCII so validation is unaffected; non-ASCII text can
    % tokenize differently from the reference.

    properties (SetAccess = private)
        Tokens (1,:) string    % vocab.txt lines in order; Tokens(1) = line 1 (1-indexed MATLAB)
        UnkToken (1,1) string = "[UNK]"   % the literal unknown-word token string
        UnkId (1,1) double                % 0-indexed id of UnkToken within the loaded vocab
        % BERT's BasicTokenizer splits every punctuation character into its
        % own word BEFORE WordPiece runs ("gas-turbine." -> gas - turbine .).
        % Default false keeps the original whitespace-only behaviour
        % (byte-identical for every pre-existing caller); the neural-retrieval
        % backbone turns it on because real checkpoints were trained with it.
        BasicTokenize (1,1) logical = false
    end

    properties (Access = private)
        Vocab_        % OceanNLP.Vocab.Vocabulary (fixed): the single token<->id table, hash lookups
        Algo_         % OceanNLP.Subword.Algorithm.WordPieceAlgorithm: the greedy matcher over the same pieces
    end

    properties (Dependent, SetAccess = private)
        VocabSize   % numel(Tokens)
    end

    methods
        function obj = WordPieceTokenizer(options)
            % WORDPIECETOKENIZER  Load a FIXED vocab.txt (one token per line,
            % plain text — the real BERT vocab file format, no library
            % needed to parse it).
            %   tok = WordPieceTokenizer('VocabFile', path)
            %   tok = WordPieceTokenizer('VocabFile', path, 'UnkToken', "[UNK]")
            arguments
                options.VocabFile (1,1) string
                options.UnkToken  (1,1) string = "[UNK]"
                options.BasicTokenize (1,1) logical = false
            end
            if ~isfile(options.VocabFile)
                error('OceanNLP:Embedding:Tokenize:WordPieceTokenizer:VocabFileNotFound', ...
                    'VocabFile "%s" does not exist.', options.VocabFile);
            end

            lines = OceanNLP.Embedding.Tokenize.WordPieceTokenizer.readVocabLines_(options.VocabFile);
            if isempty(lines)
                error('OceanNLP:Embedding:Tokenize:WordPieceTokenizer:EmptyVocab', ...
                    'VocabFile "%s" contains no tokens.', options.VocabFile);
            end

            [u, firstIdx] = unique(lines, 'stable');
            if numel(u) ~= numel(lines)
                dupPos = setdiff(1:numel(lines), firstIdx);
                error('OceanNLP:Embedding:Tokenize:WordPieceTokenizer:DuplicateToken', ...
                    'VocabFile "%s" contains the token "%s" more than once (lines must be unique).', ...
                    options.VocabFile, lines(dupPos(1)));
            end

            if ~any(lines == options.UnkToken)
                error('OceanNLP:Embedding:Tokenize:WordPieceTokenizer:UnkTokenNotInVocab', ...
                    'UnkToken "%s" was not found in VocabFile "%s".', options.UnkToken, options.VocabFile);
            end

            % The token<->id table is the shared OceanNLP.Vocab.Vocabulary (v2), not a private map.
            obj.Vocab_      = OceanNLP.Vocab.Vocabulary.fromTokens(lines, 'Unk', options.UnkToken);
            obj.Tokens      = lines;
            obj.UnkToken    = options.UnkToken;
            obj.BasicTokenize = options.BasicTokenize;
            obj.UnkId       = obj.Vocab_.UnkId - 1;   % 0-indexed, from the ACTUAL loaded vocab
            obj.Algo_       = OceanNLP.Subword.Algorithm.WordPieceAlgorithm.fromPieces(lines, 'UnkToken', options.UnkToken);
        end

        function tokens = forward(obj, rawText)
            % FORWARD  Greedy longest-match-first WordPiece tokenization of
            % ONE document. NO [CLS]/[SEP] is inserted (see class TWO
            % METHODS note) — a later composition step's job, not this one.
            %   tokens = tok.forward(rawText) -> (1 x T) string array.
            arguments
                obj
                rawText (1,1) string
            end

            txt = lower(strtrim(rawText));
            if obj.BasicTokenize
                % ASCII punctuation/symbol ranges (BERT BasicTokenizer._is_punctuation, ASCII
                % part): each becomes its own word. Non-ASCII Unicode punctuation and
                % accent stripping are NOT handled (see BASICTOKENIZE LIMITS note).
                txt = strtrim(regexprep(txt, '([!-/:-@\[-`{-~])', ' $1 '));
            end
            if txt == ""
                tokens = strings(1, 0);
                return
            end

            words = string(regexp(txt, '\s+', 'split'));
            tokens = strings(1, 0);
            for i = 1:numel(words)
                tokens = [tokens, obj.wordpieceOneWord_(words(i))]; %#ok<AGROW>
            end
        end

        function ids = encode(obj, tokens)
            % ENCODE  Map a (1 x T) token string array to a (1 x T)
            % 0-INDEXED integer id array via the FIXED loaded vocabulary
            % (see class ENCODE / 0-INDEXING note — this is the mandatory
            % convention TokenEmbedding expects). A token not present in the
            % loaded vocabulary maps to obj.UnkId (not necessarily 100 — see
            % class ALGORITHM note).
            arguments
                obj
                tokens (1,:) string
            end
            ids = obj.Vocab_.id0(tokens);          % 0-indexed; unknown -> UnkId (Vocabulary hash lookup)
        end

        function [ids, segmentIds] = encodePair(obj, query, doc, options)
            % ENCODEPAIR  [CLS] query [SEP] doc [SEP] as 0-indexed ids plus
            % segment ids (0 for [CLS]+query+[SEP], 1 for doc+[SEP]).
            % Truncation so the pair fits MaxLength is applied to the
            % SUBWORD tokens, matching HuggingFace:
            %   'longest_first' (HF truncation=True, the sentence-transformers
            %     CrossEncoder default). Empirically verified against
            %     transformers 5.16.1 (see compare_neural validation notes): the
            %     budget is split EVENLY, and an odd extra token goes to the
            %     originally LONGER side (the document on a tie). Closed form:
            %     shorter side keeps min(len, floor(budget/2)); the longer side
            %     keeps the rest. (A token-by-token "trim the longer side" loop
            %     matches HF less often.)
            %     KNOWN DEVIATION: over a 10,545-combination grid of repeated-word
            %     inputs this matches HF in 93.2% of cases; the 714 mismatches are
            %     all ODD budget, query longer than doc, both sides above
            %     floor(budget/2) (HF then sometimes gives the odd token to the
            %     document). HF's exact rule for that corner is not pinned down.
            %     Unreachable for realistic query/doc lengths (short query, long doc).
            %   'only_second': truncate the document only; errors (as HF does)
            %     when the query leaves no room for at least one document token.
            arguments
                obj
                query (1,1) string
                doc   (1,1) string
                options.MaxLength (1,1) double {mustBeInteger, mustBeGreaterThan(options.MaxLength,3)} = 512
                options.Truncation (1,1) string {mustBeMember(options.Truncation, ["longest_first","only_second"])} = "longest_first"
            end
            q = obj.forward(query);
            d = obj.forward(doc);
            budget = options.MaxLength - 3;              % [CLS] + 2 x [SEP]
            if options.Truncation == "only_second"
                if numel(q) >= budget
                    error('OceanNLP:Embedding:Tokenize:WordPieceTokenizer:QueryTooLong', ...
                        ['Truncation="only_second" needs room for at least one document token: the query has ' ...
                         '%d subword tokens but MaxLength=%d leaves a budget of %d.'], numel(q), options.MaxLength, budget);
                end
                d = d(1:min(numel(d), budget - numel(q)));
            elseif numel(q) + numel(d) > budget
                if numel(q) > numel(d)
                    longerIsQuery = true;
                    lenShorter = numel(d);
                else
                    longerIsQuery = false;                       % tie -> document is "longer"
                    lenShorter = numel(q);
                end
                keepShorter = min(lenShorter, floor(budget / 2));
                keepLonger = budget - keepShorter;
                if longerIsQuery
                    q = q(1:keepLonger);  d = d(1:keepShorter);
                else
                    d = d(1:keepLonger);  q = q(1:keepShorter);
                end
            end
            ids = obj.encode(["[CLS]", q, "[SEP]", d, "[SEP]"]);
            segmentIds = [zeros(1, numel(q) + 2), ones(1, numel(d) + 1)];
        end

        function s = name(~)
            s = "wordpiece";
        end

        function v = get.VocabSize(obj)
            v = numel(obj.Tokens);
        end

        function meta = getMetadata(~)
            % GETMETADATA  Pillar 1 (symbolic) — honest-opacity stub (R7).
            % Greedy longest-match-first WordPiece tokenization is a
            % discrete string-matching procedure, not a closed-form numeric
            % map: no SymbolicLatex is fabricated for it (same reasoning as
            % WhitespaceTokenizer/RegexTokenizer).
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            meta.Assumptions = { ...
                'input is lowercased before matching (matches bert-base-uncased/bert-tiny convention)', ...
                'whole-word match wins immediately if the word is present in the vocabulary as-is', ...
                'otherwise greedy longest-match-first: longest vocabulary-matching prefix of the remaining substring is taken as each piece, subsequent pieces looked up with a "##" prefix', ...
                'a word with no valid full decomposition under the loaded vocabulary discards all partial matches and emits a single UnkToken for the whole word (never partial pieces)', ...
                'vocabulary is FIXED at construction (loaded from vocab.txt), never fit()/learned from the input corpus — see class POLICY note'};
            meta.IsSymbolic = false;
            meta.Reference = "discrete greedy longest-match-first (maximal matching) WordPiece procedure (Wu et al., 2016; Devlin et al., 2018, §3.1), not a closed-form transform";
        end

        function spec = getPipelineSpec(~)
            % GETPIPELINESPEC  Pillar 3 (diagram) — port signature and canvas
            % metadata for this NONVIRTUAL leaf operator. Exposes two
            % distinct operations with different port shapes (forward and
            % encode), same multi-operation pattern as Vocabulary.
            spec = struct();
            spec.Inputs = [ ...
                struct('Name', "rawText", 'Domain', "raw-text", 'Dtype', "string", 'Shape', "1 x 1 (scalar document, forward)"), ...
                struct('Name', "tokens", 'Domain', "token-sequence", 'Dtype', "string", 'Shape', "1 x T (encode)")];
            spec.Outputs = [ ...
                struct('Name', "tokens", 'Domain', "token-sequence", 'Dtype', "string", 'Shape', "1 x T (forward)"), ...
                struct('Name', "ids", 'Domain', "term-id-sequence", 'Dtype', "double", 'Shape', "1 x T, 0-indexed (encode)")];
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "wordpiece");
            spec.ComposesFrom = {'OceanNLP.Vocab.Vocabulary', 'OceanNLP.Subword.Algorithm.WordPieceAlgorithm'};
        end
    end

    methods (Access = private)
        function pieces = wordpieceOneWord_(obj, word)
            % Greedy longest-match-first is OceanNLP.Subword.Algorithm.WordPieceAlgorithm (single
            % implementation shared with the trainable SubwordTokenizer): whole word if in the vocab,
            % else longest-match with "##" continuations, no full decomposition or > 100 chars -> UnkToken.
            pieces = obj.Algo_.segment(word);
        end
    end

    methods (Static, Access = private)
        function lines = readVocabLines_(path)
            % READVOCABLINES_  Read a plain-text vocab.txt, one token per
            % line, preserving line order exactly (line order IS the id
            % order — see class ENCODE / 0-INDEXING note). A trailing blank
            % line from a final newline is dropped (not a real vocab entry);
            % any other blank line is kept as a literal empty-string token,
            % since this is a byte-faithful line reader, not a "skip blank
            % lines" text-cleanup pass.
            raw = readlines(path);
            raw = reshape(string(raw), 1, []);
            if numel(raw) > 0 && strlength(raw(1)) > 0 && double(char(extractBefore(raw(1) + " ", 2))) == 65279
                raw(1) = extractAfter(raw(1), 1);        % UTF-8 byte-order mark
            end
            if numel(raw) > 0 && raw(end) == ""
                raw(end) = [];
            end
            lines = raw;
        end
    end
end
