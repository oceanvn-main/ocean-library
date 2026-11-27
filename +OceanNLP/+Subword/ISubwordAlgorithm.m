classdef (Abstract) ISubwordAlgorithm < OceanNLP.Text.TextStrategy
    % ISUBWORDALGORITHM  Strategy: how words are split into subword pieces and
    % how those pieces are learned (BPE merges, WordPiece likelihood, Unigram...).
    % OceanNLP.Subword.SubwordTokenizer holds one and never names a concrete
    % algorithm, so a new algorithm is one new file in +Algorithm (found by name
    % through OceanNLP.Text.Factory.create("subword", ...)).
    %
    % PyTorch-style: the algorithm object owns its learned state (merges, scores)
    % as buffers mutated by train(); segment()/tokens() are pure given that state.
    %
    %   train(words, counts, 'VocabSize', V, 'Seed', alphabetSeed, 'MinFrequency', m)
    %       words  1xU unique (already pre-tokenized / byte-mapped) words
    %       counts 1xU occurrence counts
    %       VocabSize = number of pieces the algorithm may produce (alphabet + merges)
    %   pieces = segment(word)         word -> 1xP string pieces
    %   toks   = tokens()              1xV pieces in creation order (alphabet first)
    %   tf     = isTrained()
    %
    % See also: OceanNLP.Subword.SubwordTokenizer, OceanNLP.Subword.Algorithm.BpeAlgorithm

    %   text   = detokenize(pieces)    pieces (1xP string) -> text (byte-level mapping, if any, is undone
    %                                  by SubwordTokenizer AFTER this call). Each algorithm owns how its
    %                                  pieces are glued back together (suffix markers, "##" prefixes, ...).
    %   unk    = unkToken()            the piece segment() emits for unsegmentable words, "" if it never does
    methods (Abstract)
        train(obj, words, counts, options)
        pieces = segment(obj, word)
        toks = tokens(obj)
        tf = isTrained(obj)
        text = detokenize(obj, pieces)
    end

    methods
        function u = unkToken(~)
            u = "";
        end
    end
end
