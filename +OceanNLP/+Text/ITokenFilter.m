classdef (Abstract) ITokenFilter < OceanNLP.Text.TextStrategy
    % ITOKENFILTER  Token stream -> token stream, applied AFTER tokenization
    % (Lucene "token filter"): lowercase, drop stop words, stem, length limits.
    %   out = filter.forward(tokens)   tokens: 1xT string  ->  1xT' string
    methods (Abstract)
        out = forward(obj, tokens)
    end
end
