classdef BM25Ranker < OceanNLP.Retrieval.IRanker
    % BM25RANKER  Okapi BM25 lexical (term-overlap) retrieval: fit(corpus), rank(query).
    %
    % Delegates corpus indexing and query scoring to a strategy object implementing 
    % OceanNLP.Index.IIndexingStrategy ("inverted_index" or "matrix").
    %
    % Strategies:
    %   - "inverted_index" (default): Uses OceanNLP.Index.InvertedIndex to build
    %     posting lists directly. Scales linearly with document frequencies of query terms.
    %
    %   - "matrix": Uses OceanNLP.Index.MatrixIndexStrategy to build a sparse 
    %     weight matrix (N x V). Evaluates queries via sparse matrix-vector product.
    %
    % USAGE:
    %   bm25 = OceanNLP.Retrieval.BM25Ranker('Strategy', "inverted_index");
    %   bm25.fit(corpus);
    %   [idx, scores] = bm25.rank("cat food");
    %
    % See also: OceanNLP.Retrieval.IRanker,
    %           OceanNLP.Index.IIndexingStrategy,
    %           OceanNLP.Index.InvertedIndex,
    %           OceanNLP.Index.MatrixIndexStrategy,
    %           OceanNLP.Index.IndexStrategyFactory

    properties (SetAccess = private)
        IndexStrategy
    end

    properties (Dependent, SetAccess = private)
        Strategy
        Tokenizer
        TermSaturation        % BM25 k1 (owned by Weighting)
        LengthNormalization   % BM25 b  (owned by Weighting)
        K1                    % legacy alias of TermSaturation
        B                     % legacy alias of LengthNormalization
        Vocabulary
        NumDocs
        
        % Forwarded strategy buffers for backward compatibility
        Index
        Vectorizer
        Weighting
        WeightMatrix
    end

    methods
        function obj = BM25Ranker(options)
            arguments
                options.Strategy = "inverted_index"
                options.Tokenizer (1,1) OceanNLP.Tokenize.ITokenizer = OceanNLP.Tokenize.WhitespaceTokenizer()
                options.TermSaturation (1,1) double {mustBePositive} = 1.2
                options.LengthNormalization (1,1) double {mustBeInRange(options.LengthNormalization,0,1)} = 0.75
                options.K1 double {mustBePositive} = []                     % legacy alias
                options.B  double {mustBeInRange(options.B,0,1)} = []        % legacy alias
            end
            
            if isa(options.Strategy, 'OceanNLP.Index.IIndexingStrategy')
                obj.IndexStrategy = options.Strategy;
            elseif isstring(options.Strategy) || ischar(options.Strategy)
                strName = string(options.Strategy);
                % The ranker owns the BM25 parameters; the index only stores stats.
                weighting = OceanStats.Data.Weighting.BM25Weighting( ...
                    'TermSaturation', options.TermSaturation, ...
                    'LengthNormalization', options.LengthNormalization, ...
                    'K1', options.K1, 'B', options.B);
                obj.IndexStrategy = OceanNLP.Index.IndexStrategyFactory.create(strName, ...
                    'Tokenizer', options.Tokenizer, ...
                    'Weighting', weighting);
            else
                error('OceanNLP:Retrieval:BM25Ranker:InvalidStrategy', ...
                    'Strategy must be a string ("inverted_index" or "matrix") or an IIndexingStrategy object.');
            end
        end

        function fit(obj, corpus)
            arguments
                obj
                corpus (1,:) {mustBeText}
            end
            % Delegate directly to the indexing strategy — no string checking!
            obj.IndexStrategy.fit(corpus);
        end

        function [rankedIdx, scores] = rank(obj, query)
            arguments
                obj
                query (1,1) string
            end
            if ~obj.isFitted()
                error('OceanNLP:Retrieval:BM25Ranker:NotFitted', ...
                    'Call fit(corpus) before rank().');
            end

            % Delegate scoring directly to the strategy object
            scoresByDoc = obj.IndexStrategy.score(query);
            
            [sortedScores, order] = sort(scoresByDoc, 'descend');
            rankedIdx = order;
            scores = sortedScores;
        end

        function tf = isFitted(obj)
            tf = ~isempty(obj.IndexStrategy) && obj.IndexStrategy.isFitted();
        end

        function s = name(~)
            s = "bm25";
        end

        % Dependent getters for backward compatibility
        function s = get.Strategy(obj)
            if isa(obj.IndexStrategy, 'OceanNLP.Index.InvertedIndex')
                s = "inverted_index";
            elseif isa(obj.IndexStrategy, 'OceanNLP.Index.MatrixIndexStrategy')
                s = "matrix";
            else
                s = "custom";
            end
        end

        function tok = get.Tokenizer(obj)
            tok = obj.IndexStrategy.Tokenizer;
        end

        function k = get.TermSaturation(obj)
            k = obj.Weighting.TermSaturation;
        end

        function b = get.LengthNormalization(obj)
            b = obj.Weighting.LengthNormalization;
        end

        function k = get.K1(obj)
            k = obj.TermSaturation;
        end

        function b = get.B(obj)
            b = obj.LengthNormalization;
        end

        function v = get.Vocabulary(obj)
            v = obj.IndexStrategy.Vocabulary;
        end

        function n = get.NumDocs(obj)
            n = obj.IndexStrategy.NumDocs;
        end

        function idx = get.Index(obj)
            if isa(obj.IndexStrategy, 'OceanNLP.Index.InvertedIndex')
                idx = obj.IndexStrategy;
            else
                idx = [];
            end
        end

        function v = get.Vectorizer(obj)
            if isa(obj.IndexStrategy, 'OceanNLP.Index.MatrixIndexStrategy')
                v = obj.IndexStrategy.Vectorizer;
            else
                v = [];
            end
        end

        function w = get.Weighting(obj)
            w = obj.IndexStrategy.Weighting;
        end

        function wm = get.WeightMatrix(obj)
            if isa(obj.IndexStrategy, 'OceanNLP.Index.MatrixIndexStrategy')
                wm = obj.IndexStrategy.WeightMatrix;
            else
                wm = [];
            end
        end

        function meta = getMetadata(obj)
            meta = obj.IndexStrategy.getMetadata();
        end

        function spec = getPipelineSpec(obj)
            spec = obj.IndexStrategy.getPipelineSpec();
        end
    end
end
