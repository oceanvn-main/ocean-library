classdef (Abstract) ICharFilter < OceanNLP.Text.TextStrategy
    % ICHARFILTER  Text -> text cleanup applied BEFORE tokenization
    % (Lucene "char filter"): strip markup, normalize Unicode, fold accents.
    %   out = filter.forward(text)     text: 1x1 string  ->  1x1 string
    methods (Abstract)
        out = forward(obj, text)
    end
end
