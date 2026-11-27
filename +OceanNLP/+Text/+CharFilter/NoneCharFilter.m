classdef NoneCharFilter < OceanNLP.Text.ICharFilter
    % NONECHARFILTER  Identity (the "basic version": leave the text untouched).
    properties (Constant)
        Reference = "identity: no text cleanup"
        InDomain = "raw-text"
        OutDomain = "raw-text"
    end
    methods
        function out = forward(~, text)
            arguments
                ~
                text (1,1) string
            end
            out = text;
        end
        function s = name(~), s = "none"; end
    end
end
