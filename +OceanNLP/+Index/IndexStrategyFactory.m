classdef IndexStrategyFactory
    % INDEXSTRATEGYFACTORY  Static factory for constructing corpus indexing strategies.
    %
    % Instantiates concrete strategy objects implementing OceanNLP.Index.IIndexingStrategy
    % ("inverted_index" or "matrix").
    %
    % USAGE:
    %   strategy = OceanNLP.Index.IndexStrategyFactory.create("inverted_index", ...
    %       'Weighting', OceanStats.Data.Weighting.BM25Weighting('TermSaturation', 1.2, 'LengthNormalization', 0.75));
    %   strategy = OceanNLP.Index.IndexStrategyFactory.create("matrix", 'Tokenizer', customTok);
    %
    % See also: OceanNLP.Index.IIndexingStrategy,
    %           OceanNLP.Index.InvertedIndex,
    %           OceanNLP.Index.MatrixIndexStrategy

    methods (Static)
        function strategy = create(strategyName, options)
            arguments
                strategyName (1,1) string {mustBeMember(strategyName, ["inverted_index", "matrix"])} = "inverted_index"
                options.Tokenizer (1,1) OceanNLP.Tokenize.ITokenizer = OceanNLP.Tokenize.WhitespaceTokenizer()
                options.Weighting (1,1) OceanStats.Data.Weighting.BM25Weighting = OceanStats.Data.Weighting.BM25Weighting()
            end

            switch lower(strategyName)
                case "inverted_index"
                    strategy = OceanNLP.Index.InvertedIndex( ...
                        'Tokenizer', options.Tokenizer, ...
                        'Weighting', options.Weighting);
                case "matrix"
                    strategy = OceanNLP.Index.MatrixIndexStrategy( ...
                        'Tokenizer', options.Tokenizer, ...
                        'Weighting', options.Weighting);
                otherwise
                    error('OceanNLP:Index:IndexStrategyFactory:InvalidStrategy', ...
                        'Unknown indexing strategy: "%s". Must be "inverted_index" or "matrix".', strategyName);
            end
        end
    end
end
