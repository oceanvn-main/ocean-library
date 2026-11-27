classdef (Abstract) IPreTokenizer < OceanNLP.Text.TextStrategy
    % IPRETOKENIZER  Strategy: raw text -> "words" that a subword algorithm
    % then segments independently (whitespace, BERT basic, GPT-2 regex).
    %   words = pre.forward(text)      text: 1x1 string  ->  1xW string
    methods (Abstract)
        words = forward(obj, text)
    end
end
