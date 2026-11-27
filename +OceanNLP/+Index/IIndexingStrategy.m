classdef (Abstract) IIndexingStrategy < handle
    % IINDEXINGSTRATEGY  Abstract strategy interface for document corpus indexing & scoring.
    %
    % Strategy pattern interface for search engine indexing backends (e.g. inverted 
    % posting lists vs sparse count matrices). Concrete implementations encapsulate 
    % corpus fit() data structures and query score() computations without requiring 
    % high-level rankers to perform conditional branching or type checks.
    %
    % USAGE:
    %   strategy = OceanNLP.Index.InvertedIndexStrategy();
    %   strategy.fit(corpus);
    %   scoresByDoc = strategy.score("query term");
    %
    % See also: OceanNLP.Index.InvertedIndexStrategy,
    %           OceanNLP.Index.MatrixIndexStrategy,
    %           OceanNLP.Retrieval.BM25Ranker

    properties (Abstract, SetAccess = protected)
        Tokenizer
        NumDocs
    end

    properties (Abstract, Dependent, SetAccess = private)
        Vocabulary
    end

    methods (Abstract)
        % FIT  Index a document collection.
        fit(obj, corpus)

        % SCORE  Evaluate all corpus documents against a query string.
        scoresByDoc = score(obj, query)

        % ISFITTED  Returns true if the indexing structure is fitted.
        tf = isFitted(obj)

        % GETMETADATA  Return Pillar 1 metadata
        meta = getMetadata(obj)

        % GETPIPELINESPEC  Return Pillar 3 diagram pipeline spec
        spec = getPipelineSpec(obj)
    end
end
