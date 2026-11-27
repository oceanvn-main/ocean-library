classdef InvertedIndexStrategy < OceanNLP.Index.IIndexingStrategy
    % INVERTEDINDEXSTRATEGY  Posting list inverted index strategy for BM25 retrieval.
    %
    % Uses OceanNLP.Index.InvertedIndex to build term posting lists.
    % During score(), iterates over posting lists for query terms and accumulates 
    % BM25 weights. Scales linearly with document frequencies of query terms.
    %
    % See also: OceanNLP.Index.IIndexingStrategy, OceanNLP.Index.InvertedIndex,
    %           OceanNLP.Retrieval.BM25Ranker

    properties (SetAccess = protected)
        Tokenizer = OceanNLP.Tokenize.WhitespaceTokenizer()
        NumDocs = 0
        K1 = 1.2
        B  = 0.75
        Index OceanNLP.Index.InvertedIndex
    end

    properties (Dependent, SetAccess = private)
        Vocabulary
    end

    methods
        function obj = InvertedIndexStrategy(options)
            arguments
                options.Tokenizer (1,1) OceanNLP.Tokenize.ITokenizer = OceanNLP.Tokenize.WhitespaceTokenizer()
                options.K1 (1,1) double {mustBePositive} = 1.2
                options.B  (1,1) double {mustBeInRange(options.B,0,1)} = 0.75
            end
            obj.Tokenizer = options.Tokenizer;
            obj.K1 = options.K1;
            obj.B = options.B;
            obj.Index = OceanNLP.Index.InvertedIndex('Tokenizer', obj.Tokenizer);
        end

        function fit(obj, corpus)
            arguments
                obj
                corpus (1,:) {mustBeText}
            end
            obj.Index.fit(corpus);
            obj.NumDocs = obj.Index.numDocuments();
        end

        function scoresByDoc = score(obj, query)
            arguments
                obj
                query (1,1) string
            end
            
            N = obj.NumDocs;
            scoresByDoc = zeros(1, N);
            if N == 0
                return;
            end

            vocab = obj.Index.Vocabulary;
            tokens = obj.Tokenizer.forward(query);
            ids = vocab.encode(tokens);
            
            if isempty(ids)
                return;
            end

            [queryTermIds, ~, ic] = unique(ids);
            queryTermCounts = accumarray(ic(:), 1)';
            
            avgLen = obj.Index.averageDocumentLength();
            oovId = vocab.OOVId;
            
            for k = 1:numel(queryTermIds)
                tId = queryTermIds(k);
                if tId == oovId
                    continue;
                end
                
                df = obj.Index.documentFrequency(tId);
                if df == 0
                    continue;
                end
                
                % IDF formula matching BM25Weighting
                idf = log((N - df + 0.5) / (df + 0.5) + 1);
                
                [docIds, counts] = obj.Index.postings(tId);
                
                % Accumulate term-level BM25 scores across postings
                for i = 1:numel(docIds)
                    dId = docIds(i);
                    tf = counts(i);
                    L = obj.Index.documentLength(dId);
                    
                    num = tf * (obj.K1 + 1);
                    den = tf + obj.K1 * (1 - obj.B + obj.B * (L / avgLen));
                    
                    termScore = idf * (num / den);
                    scoresByDoc(dId) = scoresByDoc(dId) + termScore * queryTermCounts(k);
                end
            end
        end

        function tf = isFitted(obj)
            tf = ~isempty(obj.Index) && obj.Index.isFitted();
        end

        function v = get.Vocabulary(obj)
            v = obj.Index.Vocabulary;
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
            spec.ComposesFrom = {'OceanNLP.Index.InvertedIndex'};
        end
    end
end
