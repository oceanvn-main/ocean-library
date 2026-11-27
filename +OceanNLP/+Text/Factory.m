classdef Factory
    % FACTORY  One convention-based factory for every text strategy KIND.
    % Adding an algorithm never edits this file (classes are discovered by
    % OceanNLP.Text.StrategyResolver); only adding a brand-new KIND would.
    %
    %   f   = Factory.create("filter", "lowercase")
    %   st  = Factory.create("stemmer", "porter")
    %   tok = Factory.create("tokenizer", "regex", 'Pattern', "\w+")
    %   Factory.validNames("stemmer")            % {'none','porter'}
    %
    % Kinds: charfilter, filter, stemmer, pretokenizer, tokenizer, subword.
    %
    % See also: OceanNLP.Text.StrategyResolver, OceanNLP.Tokenize.TokenizerFactory

    methods (Static)
        function obj = create(kind, name, varargin)
            [pkg, suffix, base] = OceanNLP.Text.Factory.kindInfo_(kind);
            ctor = OceanNLP.Text.StrategyResolver.resolve(pkg, suffix, name, base);
            obj = ctor(varargin{:});
        end

        function n = validNames(kind)
            [pkg, suffix] = OceanNLP.Text.Factory.kindInfo_(kind);
            n = OceanNLP.Text.StrategyResolver.names(pkg, suffix);
        end

        function k = kinds()
            k = ["charfilter" "filter" "stemmer" "pretokenizer" "tokenizer" "subword"];
        end
    end

    methods (Static, Access = private)
        function [pkg, suffix, base] = kindInfo_(kind)
            switch lower(string(kind))
                case "charfilter",   pkg = "OceanNLP.Text.CharFilter";    suffix = "CharFilter";   base = "OceanNLP.Text.ICharFilter";
                case "filter",       pkg = "OceanNLP.Text.Filter";        suffix = "Filter";       base = "OceanNLP.Text.ITokenFilter";
                case "stemmer",      pkg = "OceanNLP.Text.Stem";          suffix = "Stemmer";      base = "OceanNLP.Text.IStemmer";
                case "pretokenizer", pkg = "OceanNLP.Text.PreTokenize";   suffix = "PreTokenizer"; base = "OceanNLP.Text.IPreTokenizer";
                case "tokenizer",    pkg = "OceanNLP.Tokenize";           suffix = "Tokenizer";    base = "OceanNLP.Tokenize.ITokenizer";
                case "subword",      pkg = "OceanNLP.Subword.Algorithm";  suffix = "Algorithm";    base = "OceanNLP.Subword.ISubwordAlgorithm";
                otherwise
                    error('OceanNLP:Text:Factory:UnknownKind', ...
                        'Unknown strategy kind "%s". Valid: %s.', kind, strjoin(OceanNLP.Text.Factory.kinds(), ', '));
            end
        end
    end
end
