classdef StopWordFilter < OceanNLP.Text.ITokenFilter
    % STOPWORDFILTER  Drops tokens found in a stop list (Lucene StopFilter).
    % Default list = Lucene's ENGLISH_STOP_WORDS_SET (33 words). Comparison is
    % exact on the incoming token, so put a LowercaseFilter BEFORE this one.
    %   f = StopWordFilter()                       % Lucene English list
    %   f = StopWordFilter('Words', ["the" "of"])  % custom list
    properties (SetAccess = private)
        Words (1,:) string
    end
    properties (Constant)
        Reference = "Lucene StopFilter with ENGLISH_STOP_WORDS_SET"
        InDomain = "token-sequence"
        OutDomain = "token-sequence"
        LuceneEnglish = ["a" "an" "and" "are" "as" "at" "be" "but" "by" "for" "if" "in" "into" "is" "it" ...
                         "no" "not" "of" "on" "or" "such" "that" "the" "their" "then" "there" "these" ...
                         "they" "this" "to" "was" "will" "with"]
    end
    methods
        function obj = StopWordFilter(options)
            arguments
                options.Words (1,:) string = OceanNLP.Text.Filter.StopWordFilter.LuceneEnglish
            end
            obj.Words = options.Words;
        end
        function out = forward(obj, tokens)
            arguments
                obj
                tokens (1,:) string
            end
            out = tokens(~ismember(tokens, obj.Words));
        end
        function s = name(~), s = "stopword"; end
    end
    methods (Access = protected)
        function a = assumptions(obj)
            a = {sprintf('%d stop words; exact (case-sensitive) match, so lowercase first', numel(obj.Words))};
        end
    end
end
