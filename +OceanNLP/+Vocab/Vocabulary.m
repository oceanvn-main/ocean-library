classdef Vocabulary < handle
    % VOCABULARY  Deterministic token <-> integer-id mapping, build-once-use-many.
    %
    % THEORY:
    %   Mirrors the torchtext.vocab.Vocab pattern: fit() (there, typically
    %   built via build_vocab_from_iterator) walks a tokenized corpus once,
    %   counts token frequencies, and freezes a token->id table on the
    %   object. encode() is then a pure lookup — no re-counting, no corpus
    %   access — exactly like calling a frozen torchtext Vocab instance.
    %
    % CONVENTION (MATLAB 1-indexing — READ BEFORE PORTING FROM PYTHON):
    %   PyTorch/torchtext vocabularies are 0-indexed and conventionally
    %   reserve index 0 for the out-of-vocabulary ("<unk>") token. MATLAB
    %   arrays are 1-indexed throughout this library, so here the OOV
    %   token is reserved at id 1 (OOVId = 1, a Constant property), not 0.
    %   Every real vocabulary entry therefore occupies ids 2..VocabSize.
    %   Do not port an off-by-one assumption from Python code that checks
    %   `id == 0` for unknown — in OceanNLP the check is `id == 1`, or
    %   just compare against Vocabulary.OOVId.
    %
    % POLICY — OUT-OF-VOCABULARY HANDLING:
    %   The OOV token defaults to the string "<unk>" and is ALWAYS present
    %   at Tokens(1)/id 1, from construction onward (even before fit() is
    %   called) and after every fit() call. encode() maps any token not
    %   seen during fit() to OOVId — unknown tokens are never dropped and
    %   never error; they collapse to the reserved OOV column/id. If the
    %   literal OOV string appears in the corpus passed to fit(), that is
    %   treated as a reserved-token collision and fit() errors loudly
    %   (OceanNLP:Vocab:Vocabulary:ReservedTokenCollision) rather than
    %   silently merging a real word into the OOV slot — construct with
    %   a different 'OOVToken' name-value if the corpus legitimately
    %   contains the default "<unk>" string.
    %
    % POLICY — DETERMINISTIC ORDERING (must be reproducible given the same
    % input, so ids are stable across repeated fit() calls / re-runs):
    %   1. All tokens across all documents are pooled and counted.
    %   2. Unique tokens are tie-broken ALPHABETICALLY first: MATLAB's
    %      unique() returns unique strings in ascending (alphabetical,
    %      case-sensitive) order — this is the tie-break order.
    %   3. Tokens are then sorted by DESCENDING frequency. MATLAB's sort()
    %      is stable (equal-count entries keep their relative input order),
    %      so tokens with equal frequency remain in the alphabetical order
    %      established in step 2.
    %   4. Ids are assigned 2, 3, 4, ... in that final (freq desc, then
    %      alphabetical) order; id 1 is always the OOV token.
    %   Net rule: "most frequent first; ties broken alphabetically
    %   ascending." Given the same tokenized corpus, fit() always produces
    %   the same Tokens array and the same ids.
    %
    % USAGE:
    %   vocab = OceanNLP.Vocab.Vocabulary();
    %   vocab.fit({["the" "cat" "sat"], ["the" "dog" "ran"]});
    %   ids   = vocab.encode(["the" "cat" "zzz"]);   % zzz unseen -> OOVId
    %   % ids = [2 4 1]   (exact values depend on frequency ordering)
    %   vocab.VocabSize    % includes the reserved OOV entry
    %   vocab.isFitted()   % true
    %
    %   % Custom OOV token:
    %   vocab2 = OceanNLP.Vocab.Vocabulary('OOVToken', "[UNK]");
    %
    %   % Save / reload:
    %   s = vocab.stateDict();
    %   vocab3 = OceanNLP.Vocab.Vocabulary();
    %   vocab3.loadStateDict(s);
    %
    % See also: OceanNLP.FeatureExtraction.CountVectorizer, OceanNLP.Tokenize.ITokenizer

    properties (SetAccess = private)
        OOVToken (1,1) string = "<unk>"    % reserved out-of-vocabulary token string
        Tokens   (1,:) string              % index i -> token; Tokens(1) is always OOVToken
        Frequencies (1,:) double           % raw corpus counts aligned with Tokens; Frequencies(1) = 0
    end

    properties (Constant)
        OOVId (1,1) double = 1             % MATLAB 1-indexed reserved id for OOVToken (see CONVENTION above)
    end

    properties (SetAccess = private)
        % Special-token registry (v2): role name ("unk","pad","cls","sep","mask","bos","eos",
        % or any extra) -> token string. Only "unk" for a plain fit() vocabulary.
        SpecialTokens (1,1) struct = struct()
    end

    properties (Access = private)
        IsFitted_ (1,1) logical = false
        UnkIdx_ (1,1) double = 1          % 1-indexed position of the unknown token (1 for fit() vocabularies)
        Index_ = []                       % containers.Map token -> 1-indexed position (lazy hash lookup)
        Dict_ = []                        % dictionary token -> 1-indexed position (lazy; used by encode, built once per fit)
    end

    properties (Dependent, SetAccess = private)
        VocabSize   % numel(Tokens); includes the reserved OOV entry
        UnkId       % 1-indexed position of the unknown token (= OOVId for fit() vocabularies)
    end

    methods
        function obj = Vocabulary(options)
            % VOCABULARY  Construct an (unfitted) vocabulary with a reserved OOV slot.
            %   vocab = Vocabulary('OOVToken', "<unk>")
            arguments
                options.OOVToken (1,1) string = "<unk>"
            end
            obj.OOVToken    = options.OOVToken;
            obj.Tokens      = obj.OOVToken;    % id 1 reserved from construction
            obj.Frequencies = 0;
        end

        function fit(obj, tokenizedDocs, options)
            % FIT  Build a deterministic token->id mapping from a tokenized corpus.
            %   vocab.fit(tokenizedDocs)
            %   tokenizedDocs — cell array, one (1 x T) string array per document
            %                   (i.e. exactly what ITokenizer.forward returns per
            %                   document). Re-running fit() rebuilds the mapping
            %                   from scratch (clobbers any previous fit), matching
            %                   sklearn's CountVectorizer.fit() re-fit semantics.
            %   See the class DETERMINISTIC ORDERING policy above for the exact
            %   tie-break rule used to assign ids.
            %   Name-value options (v2; defaults reproduce the original behaviour):
            %     'MinCount' (default 1)   drop tokens seen fewer than MinCount times
            %     'MaxSize'  (default Inf) cap on VocabSize INCLUDING the OOV entry;
            %                              keeps the most frequent (ties: alphabetical)
            arguments
                obj
                tokenizedDocs (1,:) cell
                options.MinCount (1,1) double {mustBePositive} = 1
                options.MaxSize (1,1) double {mustBeGreaterThanOrEqual(options.MaxSize, 1)} = Inf
            end

            parts = cell(1, numel(tokenizedDocs));                  % collect, then ONE concatenation (growing an array in the loop is quadratic)
            for k = 1:numel(tokenizedDocs)
                doc = tokenizedDocs{k};
                if ~(isstring(doc) || ischar(doc) || iscellstr(doc))
                    error('OceanNLP:Vocab:Vocabulary:InvalidTokenizedDocs', ...
                        ['tokenizedDocs{%d} must be a string/char/cellstr token array ' ...
                         '(as produced by ITokenizer.forward), got %s.'], k, class(doc));
                end
                parts{k} = reshape(string(doc), 1, []);
            end
            if isempty(parts), flat = strings(1, 0); else, flat = [parts{:}]; end

            if isempty(flat)
                uniqueTokens = strings(1, 0);
                counts       = zeros(1, 0);
            else
                [uniqueTokens, ~, ic] = unique(flat);   % ascending alphabetical order
                counts = accumarray(ic(:), 1)';
            end

            if any(uniqueTokens == obj.OOVToken)
                error('OceanNLP:Vocab:Vocabulary:ReservedTokenCollision', ...
                    ['Corpus contains the reserved OOV token "%s" literally. ' ...
                     'Construct Vocabulary with a different ''OOVToken'' name-value ' ...
                     'if the corpus legitimately contains this string.'], obj.OOVToken);
            end

            % Stable sort on descending frequency preserves the alphabetical
            % tie-break order already established by unique() above.
            keep = counts >= options.MinCount;
            uniqueTokens = uniqueTokens(keep);
            counts = counts(keep);
            [~, order]   = sort(counts, 'descend');       % stable: ties stay alphabetical
            order = order(1:min(numel(order), options.MaxSize - 1));
            sortedTokens = uniqueTokens(order);
            sortedCounts = counts(order);

            obj.Tokens      = [obj.OOVToken, sortedTokens];
            obj.Frequencies = [0, sortedCounts];
            obj.IsFitted_   = true;
            obj.UnkIdx_     = 1;
            obj.SpecialTokens = struct('unk', obj.OOVToken);
            obj.Index_      = [];
            obj.Dict_       = [];
        end

        function ids = encode(obj, tokens)
            % ENCODE  Map a (1 x T) token array to a (1 x T) double array of ids.
            %   Unknown tokens (not seen at fit() time) map to OOVId (1).
            arguments
                obj
                tokens (1,:) string
            end
            if ~obj.isFitted()
                error('OceanNLP:Vocab:Vocabulary:NotFitted', ...
                    'Call fit(tokenizedDocs) before encode().');
            end
            % One hash lookup per token against a dictionary built once per fit (ismember(tokens, obj.Tokens) re-sorts the
            % whole vocabulary on EVERY call, which cost ~0.2 s per document for a 130k-word vocabulary).
            % First occurrence wins, exactly like ismember.
            if any(ismissing(tokens))
                [tf, loc] = ismember(tokens, obj.Tokens);      % rare path: a missing string is never a vocabulary term
                ids = double(loc);
                ids(~tf) = obj.UnkIdx_;
                return
            end
            d = obj.dict_();
            tf = isKey(d, tokens);
            ids = obj.UnkIdx_ * ones(size(tokens)); %All start as unknown
            if any(tf)
                ids(tf) = reshape(double(d(tokens(tf))), 1, []);
            end
        end

        function tf = contains(obj, token)
            % CONTAINS  O(1) membership test for one token (hash index).
            tf = isKey(obj.index_(), char(token));
        end

        function ids0 = id0(obj, tokens)
            % ID0  0-INDEXED ids (BERT/HF convention), unknown -> unk id. Hash lookup,
            % so it stays fast for the large fixed vocabularies subword tokenizers use.
            arguments
                obj
                tokens (1,:) string
            end
            m = obj.index_();
            ids0 = zeros(1, numel(tokens));
            for i = 1:numel(tokens)
                k = char(tokens(i));
                if isKey(m, k), ids0(i) = m(k) - 1; else, ids0(i) = obj.UnkIdx_ - 1; end
            end
        end

        function tokens = token0(obj, ids0)
            % TOKEN0  inverse of id0 for 0-indexed ids.
            arguments
                obj
                ids0 (1,:) double {mustBeInteger, mustBeNonnegative}
            end
            if any(ids0 >= obj.VocabSize)
                error('OceanNLP:Vocab:Vocabulary:IdOutOfRange', ...
                    'token0() received an id >= VocabSize (%d).', obj.VocabSize);
            end
            tokens = obj.Tokens(ids0 + 1);
        end

        function id = specialId(obj, role)
            % SPECIALID  0-indexed id of a registered special token by role ("cls","pad",...).
            arguments
                obj
                role (1,1) string
            end
            if ~isfield(obj.SpecialTokens, role)
                error('OceanNLP:Vocab:Vocabulary:UnknownSpecial', ...
                    'No special token registered for role "%s".', role);
            end
            id = obj.id0(string(obj.SpecialTokens.(role)));
        end

        function saveVocabTxt(obj, path)
            % SAVEVOCABTXT  One token per line in id order (the BERT vocab.txt format).
            arguments
                obj
                path (1,1) string
            end
            fid = fopen(path, 'w', 'n', 'UTF-8');
            cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
            fprintf(fid, '%s\n', obj.Tokens);
        end

        function tokens = decode(obj, ids)
            % DECODE  Inverse of encode(): map (1 x T) ids back to tokens.
            %   Provided for symmetry/debugging (torchtext lookup_tokens
            %   equivalent) — not required by any downstream consumer.
            arguments
                obj
                ids (1,:) double {mustBeInteger, mustBePositive}
            end
            if ~obj.isFitted()
                error('OceanNLP:Vocab:Vocabulary:NotFitted', ...
                    'Call fit(tokenizedDocs) before decode().');
            end
            if any(ids > obj.VocabSize)
                error('OceanNLP:Vocab:Vocabulary:IdOutOfRange', ...
                    'decode() received an id > VocabSize (%d).', obj.VocabSize);
            end
            tokens = obj.Tokens(ids);
        end

        function tf = isFitted(obj)
            tf = obj.IsFitted_;
        end

        function v = get.UnkId(obj)
            v = obj.UnkIdx_;
        end

        function v = get.VocabSize(obj)
            v = numel(obj.Tokens);
        end

        function s = stateDict(obj)
            % STATEDICT  Snapshot fitted buffers + config for save/reload.
            s = struct('OOVToken', obj.OOVToken, 'Tokens', obj.Tokens, ...
                        'Frequencies', obj.Frequencies, 'IsFitted', obj.IsFitted_, ...
                        'SpecialTokens', obj.SpecialTokens, 'UnkIdx', obj.UnkIdx_, 'Version', 2);
        end

        function loadStateDict(obj, s)
            % LOADSTATEDICT  Restore buffers + config from stateDict() output.
            arguments
                obj
                s (1,1) struct
            end
            obj.OOVToken    = s.OOVToken;
            obj.Tokens      = s.Tokens;
            obj.Frequencies = s.Frequencies;
            obj.IsFitted_   = s.IsFitted;
            if isfield(s, 'SpecialTokens'), obj.SpecialTokens = s.SpecialTokens; else, obj.SpecialTokens = struct('unk', s.OOVToken); end
            if isfield(s, 'UnkIdx'), obj.UnkIdx_ = s.UnkIdx; else, obj.UnkIdx_ = 1; end
            obj.Index_ = [];
            obj.Dict_ = [];
        end

        function meta = getMetadata(~)
            % GETMETADATA  Pillar 1 (symbolic) — honest-opacity stub (R7).
            % Vocabulary is a lookup table built by a counting/sorting
            % procedure (fit) and a pure array lookup (encode/decode), not a
            % closed-form numeric map: no SymbolicLatex is fabricated for it.
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            meta.Assumptions = {'id 1 is always reserved for the OOV token (MATLAB 1-indexed; unlike the 0-indexed torchtext convention — see class CONVENTION note)', ...
                                 'ids are deterministic given the same tokenized corpus: unique tokens tie-broken alphabetically ascending, then sorted by descending frequency (see class DETERMINISTIC ORDERING policy)', ...
                                 'encode() maps any token unseen at fit() time to OOVId; unknown tokens are never dropped and never error'};
            meta.IsSymbolic = false;
            meta.Reference = "deterministic build-once-use-many counting/sorting procedure (torchtext Vocab pattern), not a closed-form transform";
        end

        function spec = getPipelineSpec(~)
            % GETPIPELINESPEC  Pillar 3 (diagram) — port signature and canvas
            % metadata for this NONVIRTUAL leaf operator. Vocabulary exposes
            % two distinct operations with different port shapes (fit and
            % encode); both are enumerated in Inputs.
            spec = struct();
            spec.Inputs = [ ...
                struct('Name', "tokenizedDocs", 'Domain', "token-sequence", 'Dtype', "string", 'Shape', "1 x N cell of 1 x T arrays (fit)"), ...
                struct('Name', "tokens", 'Domain', "token-sequence", 'Dtype', "string", 'Shape', "1 x T (encode)")];
            spec.Outputs = struct('Name', "ids", 'Domain', "term-id-sequence", 'Dtype', "double", 'Shape', "1 x T");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "vocab");
            spec.ComposesFrom = {};
        end
    end
    methods (Access = private)
        function d = dict_(obj)
            if isempty(obj.Dict_)
                [u, ia] = unique(obj.Tokens(:), 'stable');       % first occurrence wins
                obj.Dict_ = dictionary(u, double(ia));
            end
            d = obj.Dict_;
        end

        function m = index_(obj)
            if isempty(obj.Index_)
                m = containers.Map('KeyType', 'char', 'ValueType', 'double');
                for i = 1:numel(obj.Tokens)
                    k = char(obj.Tokens(i));
                    if ~isKey(m, k), m(k) = i; end       % first occurrence wins
                end
                obj.Index_ = m;
            end
            m = obj.Index_;
        end
    end

    methods (Static)
        function obj = fromTokens(tokens, options)
            % FROMTOKENS  A FIXED vocabulary from an ordered token list: position = id
            % (0-indexed via id0/token0, 1-indexed internally). Unlike fit(), order is
            % taken as given (this is how pretrained vocab.txt / vocab.json are used).
            %   v = Vocabulary.fromTokens(tokens, 'Unk', "[UNK]", 'Special', struct('cls',"[CLS]"))
            arguments
                tokens (1,:) string
                options.Unk (1,1) string = "<unk>"
                options.Special (1,1) struct = struct()
            end
            if numel(unique(tokens)) ~= numel(tokens)
                error('OceanNLP:Vocab:Vocabulary:DuplicateToken', ...
                    'fromTokens requires unique tokens (ids must be a bijection).');
            end
            unkPos = find(tokens == options.Unk, 1);
            if isempty(unkPos)
                error('OceanNLP:Vocab:Vocabulary:UnkNotInVocab', ...
                    'Unknown token "%s" is not in the token list.', options.Unk);
            end
            roles = fieldnames(options.Special);
            for i = 1:numel(roles)
                if ~any(tokens == string(options.Special.(roles{i})))
                    error('OceanNLP:Vocab:Vocabulary:SpecialNotInVocab', ...
                        'Special token "%s" (role %s) is not in the token list.', ...
                        string(options.Special.(roles{i})), roles{i});
                end
            end
            obj = OceanNLP.Vocab.Vocabulary('OOVToken', options.Unk);
            obj.Tokens = tokens;
            obj.Frequencies = zeros(1, numel(tokens));
            obj.IsFitted_ = true;
            obj.UnkIdx_ = unkPos;
            obj.SpecialTokens = options.Special;
            obj.SpecialTokens.unk = options.Unk;
        end

        function obj = fromVocabTxt(path, options)
            % FROMVOCABTXT  BERT-style vocab.txt (one token per line, line order = id).
            arguments
                path (1,1) string
                options.Unk (1,1) string = "[UNK]"
                options.Special (1,1) struct = struct()
            end
            if ~isfile(path)
                error('OceanNLP:Vocab:Vocabulary:FileNotFound', 'Vocab file "%s" does not exist.', path);
            end
            raw = fileread(path, 'Encoding', 'UTF-8');
            if ~isempty(raw) && double(raw(1)) == 65279, raw(1) = []; end      % UTF-8 byte-order mark
            lines = string(regexp(raw, '\r?\n', 'split'));
            if ~isempty(lines) && lines(end) == "", lines(end) = []; end
            obj = OceanNLP.Vocab.Vocabulary.fromTokens(lines, 'Unk', options.Unk, 'Special', options.Special);
        end

        function obj = fromVocabJson(path, options)
            % FROMVOCABJSON  GPT-2/HF vocab.json ({"token": id, ...}). Parsed by hand:
            % jsondecode would mangle keys that are not valid MATLAB field names.
            % Ids must be exactly 0..N-1 (a bijection); tokens are placed by id.
            % Byte-level BPE never emits an unknown token; "Unk" only names the
            % fallback entry (default: GPT-2's "<|endoftext|>").
            arguments
                path (1,1) string
                options.Unk (1,1) string = "<|endoftext|>"
                options.Special (1,1) struct = struct()
            end
            raw = fileread(path, 'Encoding', 'UTF-8');
            kv = regexp(raw, '"((?:[^"\\]|\\.)*)"\s*:\s*(\d+)', 'tokens');
            n = numel(kv);
            toks = strings(1, n);
            idv = zeros(1, n);
            for i = 1:n
                toks(i) = string(OceanNLP.Vocab.Vocabulary.unescapeJson_(kv{i}{1}));
                idv(i) = str2double(kv{i}{2});
            end
            if ~isequal(sort(idv), 0:n-1)
                error('OceanNLP:Vocab:Vocabulary:BadIds', 'vocab.json ids must be exactly 0..%d.', n-1);
            end
            ordered = strings(1, n);
            ordered(idv + 1) = toks;
            obj = OceanNLP.Vocab.Vocabulary.fromTokens(ordered, 'Unk', options.Unk, 'Special', options.Special);
        end

        function s = unescapeJson_(str)
            % UNESCAPEJSON_  \uXXXX, \", \\, \/, \n, \t, \r, \b, \f
            s = regexprep(str, '\\(u[0-9a-fA-F]{4}|.)', '${OceanNLP.Vocab.Vocabulary.unescapeOne_($1)}');
        end

        function c = unescapeOne_(code)
            if code(1) == 'u' && numel(code) == 5
                c = char(hex2dec(code(2:5)));
            else
                switch code
                    case 'n', c = newline;
                    case 't', c = char(9);
                    case 'r', c = char(13);
                    case 'b', c = char(8);
                    case 'f', c = char(12);
                    otherwise, c = code;
                end
            end
        end
    end
end
