classdef (Abstract) IStemmer < OceanNLP.Text.TextStrategy
    % ISTEMMER  Strategy: one word -> its stem. Wrapped by
    % OceanNLP.Text.Filter.StemFilter, so swapping the stemming algorithm
    % never touches the analyzer.
    %   stem = stemmer.stem(word)      word: 1x1 string  ->  1x1 string
    methods (Abstract)
        s = stem(obj, word)
    end
end
