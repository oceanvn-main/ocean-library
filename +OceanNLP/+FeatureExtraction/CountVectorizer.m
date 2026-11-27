classdef CountVectorizer < OceanNLP.FeatureExtraction.IVectorizer
    % COUNTVECTORIZER  Raw document corpus -> sparse (Observations x Features) (N x V) bag-of-words count matrix.
    %
    % THEORY:
    %   Composes an ITokenizer (raw text -> tokens) with an
    %   OceanNLP.Vocab.Vocabulary (tokens -> ids) to produce the classic
    %   bag-of-words representation: X(d, v) = number of times vocabulary
    %   term v occurs in document d. This is the same role as sklearn's
    %   CountVectorizer / torchtext's vocab-transform pipeline.
    %
    % POLICY:
    %   - Tokenizer defaults to OceanNLP.Tokenize.WhitespaceTokenizer()
    %     (lowercases, strips edge punctuation — see that class's help).
    %   - fit() builds a FRESH Vocabulary each call (clobbers any previous
    %     fit), consistent with sklearn's CountVectorizer.fit() semantics.
    %   - forward() tokenizes with the SAME Tokenizer, then encodes via the
    %     fitted Vocabulary — out-of-vocabulary tokens in a new corpus
    %     collapse into the reserved OOV column (Vocabulary.OOVId), they
    %     are never dropped or treated as an error.
    %   - Output is built via the sparse triplet pattern (accumulate
    %     (docIndex, termId, count) triples, then ONE sparse(i,j,v,N,V)
    %     call) per Master Plan §3.2 — never a dense zeros(N,V) matrix.
    %
    % USAGE:
    %   cv = OceanNLP.FeatureExtraction.CountVectorizer();
    %   cv.fit(["the cat sat" "the dog ran"]);
    %   X  = cv.forward(["the cat sat" "the dog ran"]);   % (2 x V) sparse
    %   Xnew = cv.forward("a brand new zzzunseen word");   % OOV -> col 1
    %
    %   cv2 = OceanNLP.FeatureExtraction.CountVectorizer( ...
    %       'Tokenizer', OceanNLP.Tokenize.RegexTokenizer('Pattern', "[\w']+"));
    %
    % See also: OceanNLP.FeatureExtraction.IVectorizer, OceanNLP.Vocab.Vocabulary,
    %           OceanNLP.FeatureExtraction.TfidfVectorizer

    properties
        Tokenizer (1,1) OceanNLP.Tokenize.ITokenizer = OceanNLP.Tokenize.WhitespaceTokenizer()
    end

    properties (SetAccess = private)
        Vocabulary (1,1) OceanNLP.Vocab.Vocabulary = OceanNLP.Vocab.Vocabulary()
    end

    methods
        function obj = CountVectorizer(options)
            % COUNTVECTORIZER  Construct with an optional tokenizer.
            %   cv = CountVectorizer('Tokenizer', someITokenizer)
            arguments
                options.Tokenizer (1,1) OceanNLP.Tokenize.ITokenizer = OceanNLP.Tokenize.WhitespaceTokenizer()
            end
            obj.Tokenizer = options.Tokenizer;
        end

        function fit(obj, corpus)
            % FIT  Tokenize the corpus and build a fresh Vocabulary from it.
            arguments
                obj
                corpus (1,:) {mustBeText}
            end
            corpus = string(corpus);
            tokenizedDocs = obj.tokenizeCorpus_(corpus);

            newVocab = OceanNLP.Vocab.Vocabulary();
            newVocab.fit(tokenizedDocs);
            obj.Vocabulary = newVocab;
        end

        function X = forward(obj, corpus)
            % FORWARD  Tokenize + encode the corpus into a sparse (N x V) count matrix.
            %   Built via the sparse triplet pattern (Master Plan §3.2) — one vocabulary lookup for the whole
            %   corpus and one sparse(i,j,1,N,V) call (repeated pairs are summed = the counts), never a dense intermediate.
            arguments
                obj
                corpus (1,:) {mustBeText}
            end
            if ~obj.isFitted()
                error('OceanNLP:FeatureExtraction:CountVectorizer:NotFitted', ...
                    'Call fit(corpus) before forward().');
            end
            corpus = string(corpus);
            N = numel(corpus);
            V = obj.Vocabulary.VocabSize;

            % 1. tokenize every document (the only per-document work: an ITokenizer takes one document at a time)
            tokenParts = cell(1, N);
            for d = 1:N
                tokenParts{d} = reshape(string(obj.Tokenizer.forward(corpus(d))), 1, []);   % ["the", "cat", "sat", "on", "the", "mat"]
            end
            lens = cellfun(@numel, tokenParts);
            if sum(lens) == 0
                X = sparse(N, V);
                return
            end
            % 2. ONE vocabulary lookup for all tokens of the corpus (unknown tokens -> the OOV column)
            ids = obj.Vocabulary.encode([tokenParts{:}]);                                     % [5, 8, 12, 15, 5, 18, ...]
            % 3. ONE sparse() call: the triplets (document, term, 1) are summed for repeated pairs, which IS the count
            docIdx = repelem(1:N, lens);                                                     % document of every token
            X = sparse(docIdx, ids, 1, N, V);                                                % X(d, v) = occurrences of term v in document d
        end

        function tf = isFitted(obj)
            tf = obj.Vocabulary.isFitted();
        end

        function meta = getMetadata(~)
            % GETMETADATA  Pillar 1 (symbolic) — honest-opacity stub (R7).
            % CountVectorizer composes tokenize (no closed form) + count: no
            % single end-to-end SymbolicLatex is fabricated for it.
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            meta.Assumptions = {'fit() builds a FRESH Vocabulary each call, clobbering any previous fit (sklearn CountVectorizer.fit() semantics)', ...
                                 'forward() tokenizes with the same Tokenizer used at fit-time, then encodes via the fitted Vocabulary', ...
                                 'out-of-vocabulary tokens in a new corpus collapse into the reserved OOV column (Vocabulary.OOVId); they are never dropped or treated as an error', ...
                                 'output is built via the sparse triplet pattern, never a dense intermediate'};
            meta.IsSymbolic = false;
            meta.Reference = "composition of tokenize + count, no closed-form end-to-end transform (bag-of-words / sklearn CountVectorizer convention)";
        end

        function spec = getPipelineSpec(~)
            % GETPIPELINESPEC  Pillar 3 (diagram) — port signature and canvas
            % metadata for this NONVIRTUAL leaf operator. NONVIRTUAL despite
            % internally using a Tokenizer: CountVectorizer owns real fitted
            % state (its Vocabulary), and no other already-registered
            % operator stands between it and its tokenizer in this pass.
            spec = struct();
            spec.Inputs = struct('Name', "corpus", 'Domain', "raw-text", 'Dtype', "string", 'Shape', "1 x N");
            spec.Outputs = struct('Name', "X", 'Domain', "count-matrix", 'Dtype', "double", 'Shape', "N x V");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "count");
            spec.ComposesFrom = {};
        end
    end

    methods (Access = private)
        function tokenizedDocs = tokenizeCorpus_(obj, corpus)
            N = numel(corpus);
            tokenizedDocs = cell(1, N);
            for d = 1:N
                tokenizedDocs{d} = obj.Tokenizer.forward(corpus(d));
            end
        end
    end
end
