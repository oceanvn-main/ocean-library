classdef NoneStemmer < OceanNLP.Text.IStemmer
    % NONESTEMMER  Identity stemmer (the "basic version": no stemming).
    properties (Constant)
        Reference = "identity: no stemming"
        InDomain = "token-sequence"
        OutDomain = "token-sequence"
    end
    methods
        function s = stem(~, word)
            arguments
                ~
                word (1,1) string
            end
            s = word;
        end
        function s = name(~), s = "none"; end
    end
end
