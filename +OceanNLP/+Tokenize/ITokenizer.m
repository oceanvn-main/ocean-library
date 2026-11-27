classdef (Abstract) ITokenizer < handle
    % ITOKENIZER  Abstract interface for raw text -> token sequence.
    %
    %   Stateless strategy (like IActivation). No corpus-level fitting here —
    %   that belongs to OceanNLP.Vocab.Vocabulary, which is built FROM
    %   tokenized output, keeping tokenization and vocabulary-building
    %   independently testable and swappable.
    %
    %   CONVENTION:
    %     forward(rawText) -> (1 x T) string array of tokens, T = token count.
    %     rawText — char or string scalar, one document.
    %
    %   See also: OceanNLP.Tokenize.TokenizerFactory
    methods (Abstract)
        tokens = forward(obj, rawText)
        s = name(obj)
    end
end
