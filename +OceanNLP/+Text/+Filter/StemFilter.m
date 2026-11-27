classdef StemFilter < OceanNLP.Text.ITokenFilter
    % STEMFILTER  Applies an IStemmer strategy to every token. The stemming
    % ALGORITHM is a constructor argument, so replacing Porter with any other
    % IStemmer never touches the analyzer or this class. Default = NoneStemmer
    % (identity: the basic version).
    %   f = StemFilter('Stemmer', OceanNLP.Text.Stem.PorterStemmer())
    properties (SetAccess = private)
        Stemmer
    end
    properties (Constant)
        Reference = "applies an IStemmer strategy per token"
        InDomain = "token-sequence"
        OutDomain = "token-sequence"
    end
    methods
        function obj = StemFilter(options)
            arguments
                options.Stemmer (1,1) OceanNLP.Text.IStemmer = OceanNLP.Text.Stem.NoneStemmer()
            end
            obj.Stemmer = options.Stemmer;
        end
        function out = forward(obj, tokens)
            arguments
                obj
                tokens (1,:) string
            end
            out = tokens;
            for i = 1:numel(tokens)
                out(i) = obj.Stemmer.stem(tokens(i));
            end
        end
        function s = name(obj), s = "stem:" + obj.Stemmer.name(); end
        function meta = getMetadata(obj)
            meta = getMetadata@OceanNLP.Text.TextStrategy(obj);
            meta.ComposedPartReference = struct('StemmerReference', obj.Stemmer.getMetadata().Reference);
        end
        function spec = getPipelineSpec(obj)
            spec = getPipelineSpec@OceanNLP.Text.TextStrategy(obj);
            spec.Kind = "VIRTUAL";
            spec.ComposesFrom = {class(obj.Stemmer)};
        end
    end
end
