classdef AsciiFoldingCharFilter < OceanNLP.Text.ICharFilter
    % ASCIIFOLDINGCHARFILTER  Accent stripping: NFD-decompose, drop combining
    % marks (Unicode category M*), recompose (NFC). Also folds a small set of
    % non-decomposable Latin letters (ss, ae, oe, o-stroke, l-stroke, d-stroke).
    % This is BERT's `_run_strip_accents` (uncased models) plus the extra folds;
    % it is NOT Lucene's full ASCIIFoldingFilter (a ~1000-entry table).
    properties (Constant)
        Reference = "BERT strip_accents (NFD + drop Mn) + small extra Latin folds"
        InDomain = "raw-text"
        OutDomain = "raw-text"
    end
    properties (Constant, Access = private)
        Extra = {char(223), 'ss'; char(230), 'ae'; char(198), 'AE'; char(339), 'oe'; char(338), 'OE'; ...
                 char(248), 'o'; char(216), 'O'; char(322), 'l'; char(321), 'L'; char(273), 'd'; char(272), 'D'}
    end
    methods
        function out = forward(obj, text)
            arguments
                obj
                text (1,1) string
            end
            d = OceanNLP.Text.Unicode.normalize(text, "NFD"); %Decompose into base letter and floating accent mask
            cps = OceanNLP.Text.Unicode.codepoints(d);
            keep = OceanNLP.Text.Unicode.category(cps) ~= 6;      % only Mn (nonspacing marks); Mc/Me such as Indic vowel signs are kept
            s = char(OceanNLP.Text.Unicode.fromCodepoints(cps(keep)));
            e = obj.Extra;
            for i = 1:size(e, 1)
                s = strrep(s, e{i, 1}, e{i, 2});
            end
            out = OceanNLP.Text.Unicode.normalize(string(s), "NFC");
        end
        function s = name(~), s = "asciifolding"; end
    end
    methods (Access = protected)
        function a = assumptions(~)
            a = {'accents removed by decomposition, so only letters that decompose into base + mark are folded', ...
                 'non-Latin scripts pass through unchanged', 'requires the MATLAB-bundled JVM'};
        end
    end
end
