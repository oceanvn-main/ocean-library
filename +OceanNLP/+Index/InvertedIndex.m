classdef InvertedIndex < OceanNLP.Index.IIndexingStrategy
    % INVERTEDINDEX  Document corpus -> per-term posting lists & BM25 strategy.
    %
    % Implements OceanNLP.Index.IIndexingStrategy directly. Combines raw posting list 
    % storage (docId, count) with Okapi BM25 score() evaluation over posting lists.
    %
    % See also: OceanNLP.Index.IIndexingStrategy,
    %           OceanNLP.Index.MatrixIndexStrategy,
    %           OceanNLP.Retrieval.BM25Ranker

    properties (SetAccess = protected)
        Tokenizer = OceanNLP.Tokenize.WhitespaceTokenizer()
        NumDocs = 0
        Weighting   % injected scoring object (default BM25Weighting) — owns ALL scoring math & params
    end

    properties (SetAccess = private)
        Vocabulary = OceanNLP.Vocab.Vocabulary()
        Postings   cell     = {}    % (1 x VocabSize); Postings{id} = struct('DocIds',...,'Counts',...)
        DocLengths (1,:) double = zeros(1, 0)   % (1 x N); total token count per document (OOV included)
    end

    properties (Access = private)
        IsFitted_ (1,1) logical = false
    end

    methods
        function obj = InvertedIndex(options)
            % INVERTEDINDEX  Construct with optional tokenizer and injected weighting.
            %   The index stores statistics only; scoring parameters (K1, B, idf
            %   scheme) belong to the Weighting object, not to the index.
            arguments
                options.Tokenizer (1,1) OceanNLP.Tokenize.ITokenizer = OceanNLP.Tokenize.WhitespaceTokenizer()
                options.Weighting (1,1) OceanStats.Data.Weighting.BM25Weighting = OceanStats.Data.Weighting.BM25Weighting()
            end
            obj.Tokenizer = options.Tokenizer;
            obj.Weighting = options.Weighting;
        end

        function fit(obj, corpus)
            % FIT  Tokenize the corpus, build a fresh Vocabulary, and bucket
            % (docId, count) pairs into per-term posting lists.
            arguments
                obj
                corpus (1,:) {mustBeText}
            end
            corpus = string(corpus);
            N = numel(corpus);

            tokenizedDocs = cell(1, N);
            for d = 1:N
                tokenizedDocs{d} = obj.Tokenizer.forward(corpus(d));
            end

            newVocab = OceanNLP.Vocab.Vocabulary();
            newVocab.fit(tokenizedDocs); % Built dict
            V = newVocab.VocabSize;
            oovId = newVocab.OOVId;

            docIdxParts  = cell(1, N);
            termIdxParts = cell(1, N);
            countParts   = cell(1, N);
            docLengths   = zeros(1, N);

            for d = 1:N
                ids = newVocab.encode(tokenizedDocs{d});
                docLengths(d) = numel(ids);   % OOV tokens counted toward length
                if isempty(ids)
                    docIdxParts{d}  = zeros(1, 0);
                    termIdxParts{d} = zeros(1, 0);
                    countParts{d}   = zeros(1, 0);
                    continue
                end
                [uTerms, ~, ic] = unique(ids);
                c = accumarray(ic(:), 1)';
                keep = uTerms ~= oovId;   % never build a posting list for OOV
                docIdxParts{d}  = repmat(d, 1, nnz(keep));
                termIdxParts{d} = uTerms(keep);
                countParts{d}   = c(keep);
            end

            docIdxAll  = [docIdxParts{:}];
            termIdxAll = [termIdxParts{:}];
            countAll   = [countParts{:}];

            postings = cell(1, V);
            emptyEntry = struct('DocIds', zeros(1, 0), 'Counts', zeros(1, 0));
            for termId = 1:V
                postings{termId} = emptyEntry;
            end
            if ~isempty(termIdxAll)
                [sortedTerms, order] = sort(termIdxAll);
                sortedDocIdx = docIdxAll(order);
                sortedCounts = countAll(order);
                [uniqueTerms, ~, grp] = unique(sortedTerms);
                for k = 1:numel(uniqueTerms)
                    mask = grp == k;
                    postings{uniqueTerms(k)} = struct( ...
                        'DocIds', sortedDocIdx(mask), 'Counts', sortedCounts(mask));
                end
            end

            obj.Vocabulary  = newVocab;
            obj.Postings    = postings;
            obj.DocLengths  = docLengths;
            obj.NumDocs     = N;
            obj.IsFitted_   = true;
        end

        function scoresByDoc = score(obj, query)
            % SCORE  Evaluate BM25 relevance scores over posting lists.
            arguments
                obj
                query (1,1) string
            end
            
            N = obj.NumDocs;
            scoresByDoc = zeros(1, N);
            if N == 0 || ~obj.isFitted()
                return;
            end

            vocab = obj.Vocabulary;
            tokens = obj.Tokenizer.forward(query);
            ids = vocab.encode(tokens);
            
            if isempty(ids)
                return;
            end

            [queryTermIds, ~, ic] = unique(ids);
            queryTermCounts = accumarray(ic(:), 1)';
            
            avgLen = obj.averageDocumentLength();
            oovId = vocab.OOVId;
            
            for k = 1:numel(queryTermIds)
                tId = queryTermIds(k);
                if tId == oovId || tId > obj.vocabularySize()
                    continue;
                end
                
                df = obj.documentFrequency(tId);
                if df == 0
                    continue;
                end
                
                % Index supplies raw statistics only; BM25Weighting owns the formula.
                [docIds, counts] = obj.postings(tId);
                w = obj.Weighting.computePostingWeights( ...
                    counts, df, obj.DocLengths(docIds), N, avgLen);
                scoresByDoc(docIds) = scoresByDoc(docIds) + w(:)' * queryTermCounts(k);
            end
        end

        function [docIds, counts] = postings(obj, termId)
            % POSTINGS  (docIds, counts) for a term.
            arguments
                obj
                termId (1,1) double {mustBeInteger, mustBePositive}
            end
            obj.checkFitted_();
            if termId > obj.vocabularySize()
                error('OceanNLP:Index:InvertedIndex:TermIdOutOfRange', ...
                    'termId (%d) exceeds vocabularySize (%d).', termId, obj.vocabularySize());
            end
            entry  = obj.Postings{termId};
            docIds = entry.DocIds;
            counts = entry.Counts;
        end

        function docFreq = documentFrequency(obj, termId)
            [docIds, ~] = obj.postings(termId);
            docFreq = numel(docIds);
        end

        function len = documentLength(obj, docId)
            arguments
                obj
                docId (1,1) double {mustBeInteger, mustBePositive}
            end
            obj.checkFitted_();
            if docId > obj.numDocuments()
                error('OceanNLP:Index:InvertedIndex:DocIdOutOfRange', ...
                    'docId (%d) exceeds numDocuments (%d).', docId, obj.numDocuments());
            end
            len = obj.DocLengths(docId);
        end

        function avgLen = averageDocumentLength(obj)
            obj.checkFitted_();
            if obj.NumDocs == 0
                avgLen = 0;
            else
                avgLen = mean(obj.DocLengths);
            end
        end

        function n = numDocuments(obj)
            n = obj.NumDocs;
        end

        function v = vocabularySize(obj)
            v = obj.Vocabulary.VocabSize;
        end

        function tf = isFitted(obj)
            tf = obj.IsFitted_;
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = "score(D,Q) = \sum_{t \in Q} IDF(t) \cdot \frac{tf(t,D) \cdot (k_1 + 1)}{tf(t,D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{avgDl})} \cdot qtf(t)";
            
            symbols = containers.Map();
            symbols("Q") = "query string (tokenized)";
            symbols("D") = "document string (tokenized)";
            symbols("t") = "a term in Q";
            symbols("qtf(t)") = "count of term t in Q";
            symbols("tf(t,D)") = "count of term t in D";
            symbols("IDF(t)") = "\ln(\frac{N - df(t) + 0.5}{df(t) + 0.5} + 1)";
            symbols("k_1") = "Term frequency saturation parameter (K1)";
            symbols("b") = "Document length normalization parameter (B)";
            symbols("|D|") = "Document length (number of tokens)";
            symbols("avgDl") = "Average document length in corpus";
            
            meta.Symbols = symbols;
            meta.Assumptions = {'Dynamic score accumulation over inverted posting lists at search time', ...
                                'IDF and term weighting match BM25Weighting exact formula'};
            meta.IsSymbolic = true;
            meta.Reference = "Okapi BM25 (Inverted Index Strategy)";
        end

        function spec = getPipelineSpec(~)
            spec = struct();
            spec.Inputs = [ ...
                struct('Name', "corpus", 'Domain', "raw-text", 'Dtype', "string", 'Shape', "1 x N (fit)"), ...
                struct('Name', "query", 'Domain', "raw-text", 'Dtype', "string", 'Shape', "1 x 1 (score)")];
            spec.Outputs = struct('Name', "scoresByDoc", 'Domain', "score-array", 'Dtype', "double", 'Shape', "1 x N");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "Inverted Index BM25");
            spec.ComposesFrom = {};
        end
    end

    methods (Access = private)
        function checkFitted_(obj)
            if ~obj.isFitted()
                error('OceanNLP:Index:InvertedIndex:NotFitted', ...
                    'Call fit(corpus) before querying the index.');
            end
        end
    end
end
