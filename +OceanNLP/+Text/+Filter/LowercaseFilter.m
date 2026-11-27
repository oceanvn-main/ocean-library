classdef LowercaseFilter < OceanNLP.Text.ITokenFilter
    % LOWERCASEFILTER  Case folding of a token stream (Lucene LowerCaseFilter).
    % Uses MATLAB lower(), which is Unicode-aware for cased letters.
    properties (Constant)
        Reference = "Lucene LowerCaseFilter"
        InDomain = "token-sequence"
        OutDomain = "token-sequence"
    end
    methods
        function out = forward(~, tokens)
            arguments
                ~
                tokens (1,:) string
            end
            out = lower(tokens);
        end
        function s = name(~), s = "lowercase"; end
    end
end
