classdef WordPieceAlgorithm < OceanNLP.Subword.ISubwordAlgorithm
    % WORDPIECEALGORITHM  WordPiece vocabulary learning + greedy segmentation.
    %
    % SEGMENTATION (BERT inference, identical to WordPieceTokenizer): whole word
    % if it is a piece; else greedy longest-match-first, continuation pieces
    % carry the prefix ("##"); a word with no full decomposition, or longer than
    % 100 characters, becomes the single UnkToken (partial matches discarded).
    %
    % TRAINING follows HuggingFace `tokenizers` WordPieceTrainer, which is the
    % BPE merge procedure over symbols with a continuing-subword prefix (the
    % most FREQUENT pair is merged, NOT Google's original likelihood ratio,
    % whose reference trainer is not public and is therefore not implemented).
    % The learned vocabulary is then used with the greedy segmenter above.
    %
    %   a = WordPieceAlgorithm();  a.train(words, counts, 'VocabSize', 800);
    %   a = WordPieceAlgorithm.fromPieces(pieces)               % a pretrained vocab.txt
    %
    % See also: OceanNLP.Subword.SubwordTokenizer, OceanNLP.Embedding.Tokenize.WordPieceTokenizer

    properties (SetAccess = private)
        Pieces (1,:) string = strings(1, 0)
        ContinuingPrefix (1,1) string = "##"
        UnkToken (1,1) string = "[UNK]"
        MaxWordLength (1,1) double = 100
    end
    properties (Access = private)
        Index_ = []                        % containers.Map piece -> true
        Cache_ = []
    end
    properties (Constant)
        Reference = "Wu et al. (2016) WordPiece / Devlin et al. (2018) greedy longest-match; training as HF tokenizers WordPieceTrainer (BPE with '##' prefix)"
        InDomain = "token-sequence"
        OutDomain = "token-sequence"
    end

    methods
        function obj = WordPieceAlgorithm(options)
            arguments
                options.ContinuingPrefix (1,1) string = "##"
                options.UnkToken (1,1) string = "[UNK]"
                options.MaxWordLength (1,1) double = 100
            end
            obj.ContinuingPrefix = options.ContinuingPrefix;
            obj.UnkToken = options.UnkToken;
            obj.MaxWordLength = options.MaxWordLength;
        end

        function train(obj, words, counts, options)
            arguments
                obj
                words (1,:) string
                counts (1,:) double
                options.VocabSize (1,1) double {mustBeInteger, mustBePositive} = 1000
                options.Seed (1,:) string = strings(1, 0)
                options.MinFrequency (1,1) double {mustBeNonnegative} = 0
            end
            [merges, alphabet] = OceanNLP.Subword.Algorithm.BpeAlgorithm.trainMerges( ...
                words, counts, options.VocabSize, options.Seed, options.MinFrequency, "", strings(0, 2), obj.ContinuingPrefix);
            merged = strings(1, size(merges, 1));
            for r = 1:size(merges, 1)
                right = merges(r, 2);
                if startsWith(right, obj.ContinuingPrefix), right = extractAfter(right, strlength(obj.ContinuingPrefix)); end
                merged(r) = merges(r, 1) + right;
            end
            toks = [alphabet, merged];
            [~, first] = unique(toks, 'stable');
            obj.setPieces_(toks(first));
        end

        function pieces = segment(obj, word)
            arguments
                obj
                word (1,1) string
            end
            if isempty(obj.Pieces)
                error('OceanNLP:Subword:WordPieceAlgorithm:NotTrained', 'Call train() or use fromPieces() before segment().');
            end
            if isempty(obj.Index_), obj.setPieces_(obj.Pieces); end
            key = char(word);
            if isKey(obj.Cache_, key), pieces = obj.Cache_(key); return; end
            pieces = obj.segmentUncached_(key);
            obj.Cache_(key) = pieces;
        end

        function toks = tokens(obj)
            toks = obj.Pieces;
        end

        function tf = isTrained(obj)
            tf = ~isempty(obj.Pieces);
        end

        function u = unkToken(obj)
            u = obj.UnkToken;
        end

        function text = detokenize(obj, pieces)
            % Words are separated by a space; continuation pieces lose their prefix and attach to the previous piece.
            arguments
                obj
                pieces (1,:) string
            end
            text = "";
            for i = 1:numel(pieces)
                p = pieces(i);
                if i > 1 && startsWith(p, obj.ContinuingPrefix)
                    text = text + extractAfter(p, strlength(obj.ContinuingPrefix));
                elseif i > 1
                    text = text + " " + p;
                else
                    text = p;
                end
            end
        end

        function s = name(~), s = "wordpiece"; end
    end

    methods (Access = protected)
        function a = assumptions(obj)
            a = {'segmentation: whole word if it is a piece, else greedy longest-match-first with the continuation prefix; unsegmentable or over-long words become the unk token', ...
                 'training: HF WordPieceTrainer semantics = BPE merges over prefixed symbols, most FREQUENT pair first (not the original likelihood-ratio criterion)', ...
                 ['piece ORDER (ids) of equal-frequency alphabet entries is implementation-defined in HF and not reproduced; the piece SET ' ...
                  'matched HF on the validated corpus but not on every tie-heavy corpus'], ...
                 sprintf('words longer than %d code points -> unk', obj.MaxWordLength)};
        end
    end

    methods (Access = private)
        function setPieces_(obj, pieces)
            obj.Pieces = pieces;
            obj.Index_ = containers.Map(cellstr(pieces), true(1, numel(pieces)));
            obj.Cache_ = containers.Map('KeyType', 'char', 'ValueType', 'any');
        end

        function pieces = segmentUncached_(obj, chars)
            nChars = numel(OceanNLP.Text.Unicode.codepoints(string(chars)));      % HF counts characters, not UTF-16 units
            if nChars > obj.MaxWordLength || isempty(chars)
                if isempty(chars), pieces = strings(1, 0); else, pieces = obj.UnkToken; end
                return
            end
            if isKey(obj.Index_, chars), pieces = string(chars); return; end
            pref = char(obj.ContinuingPrefix);
            out = strings(1, 0);
            pos = 1;  n = numel(chars);  first = true;
            while pos <= n
                matched = 0;
                for e = n:-1:pos
                    cand = chars(pos:e);
                    if ~first, cand = [pref cand]; end %#ok<AGROW>
                    if isKey(obj.Index_, cand)
                        matched = e;
                        out(end+1) = string(cand); %#ok<AGROW>
                        break
                    end
                end
                if matched == 0, pieces = obj.UnkToken; return; end
                pos = matched + 1;  first = false;
            end
            pieces = out;
        end
    end

    methods (Static)
        function obj = fromPieces(pieces, options)
            % FROMPIECES  Install a fixed piece list (e.g. a pretrained vocab.txt).
            arguments
                pieces (1,:) string
                options.ContinuingPrefix (1,1) string = "##"
                options.UnkToken (1,1) string = "[UNK]"
            end
            obj = OceanNLP.Subword.Algorithm.WordPieceAlgorithm('ContinuingPrefix', options.ContinuingPrefix, 'UnkToken', options.UnkToken);
            obj.setPieces_(pieces);
        end
    end
end
