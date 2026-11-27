classdef MatrixIndexStrategy < OceanNLP.Index.IIndexingStrategy
    % MATRIXINDEXSTRATEGY  Sparse count matrix & BM25 weighting strategy for retrieval.
    %
    % Uses CountVectorizer and BM25Weighting to transform a text corpus into a static 
    % (N x V) sparse weight matrix. During score(), computes a single sparse 
    % matrix-vector product for high throughput when the corpus fits in memory.
    %
    % See also: OceanNLP.Index.IIndexingStrategy,
    %           OceanNLP.FeatureExtraction.CountVectorizer,
    %           OceanStats.Data.Weighting.BM25Weighting,
    %           OceanNLP.Retrieval.BM25Ranker

    properties (SetAccess = protected)
        Tokenizer = OceanNLP.Tokenize.WhitespaceTokenizer()
        NumDocs = 0
        Vectorizer OceanNLP.FeatureExtraction.CountVectorizer
        Weighting OceanStats.Data.Weighting.BM25Weighting
        WeightMatrix double = []
    end

    properties (Dependent, SetAccess = private) %Dependent avoid duplicate memory of Vocab
        Vocabulary
    end

    methods
        function obj = MatrixIndexStrategy(options)
            arguments
                options.Tokenizer (1,1) OceanNLP.Tokenize.ITokenizer = OceanNLP.Tokenize.WhitespaceTokenizer()
                options.Weighting (1,1) OceanStats.Data.Weighting.BM25Weighting = OceanStats.Data.Weighting.BM25Weighting()
            end
            obj.Tokenizer = options.Tokenizer;
            obj.Vectorizer = OceanNLP.FeatureExtraction.CountVectorizer('Tokenizer', obj.Tokenizer);
            obj.Weighting  = options.Weighting;   % scoring params (K1, B) live here, not in the index
        end

        function fit(obj, corpus)
            arguments
                obj
                corpus (1,:) {mustBeText}
            end
            obj.Vectorizer.fit(corpus);
            counts = obj.Vectorizer.forward(corpus); %Forward build space, fit doesn't, Check docs
            obj.Weighting.fit(counts);
            obj.WeightMatrix = obj.Weighting.forward(counts);
            obj.NumDocs = size(counts, 1);
        end

        function scoresByDoc = score(obj, query)
            arguments
                obj
                query (1,1) string
            end
            
            N = obj.NumDocs;
            if N == 0
                scoresByDoc = zeros(1, 0);
                return;
            end

            vocab = obj.Vectorizer.Vocabulary;
            tokens = obj.Tokenizer.forward(query);
            ids = vocab.encode(tokens);
            
            if isempty(ids)
                scoresByDoc = zeros(1, N);
                return;
            end

            [queryTermIds, ~, ic] = unique(ids);
            queryTermCounts = accumarray(ic(:), 1)';
            
            rawScores = obj.WeightMatrix(:, queryTermIds) * queryTermCounts';
            scoresByDoc = full(rawScores(:))';
        end

        function tf = isFitted(obj)
            tf = ~isempty(obj.Vectorizer) && obj.Vectorizer.isFitted() && obj.Weighting.isFitted();
        end

        function v = get.Vocabulary(obj)
            v = obj.Vectorizer.Vocabulary;
        end

        function meta = getMetadata(~)
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            meta.Assumptions = {'VIRTUAL composition of CountVectorizer + BM25Weighting', ...
                                'Evaluated as a sparse matrix-vector product at query time'};
            meta.IsSymbolic = false;
            meta.Reference = "Okapi BM25 (Matrix Strategy)";
        end

        function spec = getPipelineSpec(~)
            spec = struct();
            spec.Inputs = [ ...
                struct('Name', "corpus", 'Domain', "raw-text", 'Dtype', "string", 'Shape', "1 x N (fit)"), ...
                struct('Name', "query", 'Domain', "raw-text", 'Dtype', "string", 'Shape', "1 x 1 (score)")];
            spec.Outputs = struct('Name', "scoresByDoc", 'Domain', "score-array", 'Dtype', "double", 'Shape', "1 x N");
            spec.Kind = "VIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "Matrix BM25");
            spec.ComposesFrom = {'OceanNLP.FeatureExtraction.CountVectorizer', 'OceanStats.Data.Weighting.BM25Weighting'};
        end
    end
end
