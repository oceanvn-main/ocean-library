classdef UnicodeNormalizeCharFilter < OceanNLP.Text.ICharFilter
    % UNICODENORMALIZECHARFILTER  Unicode normalization (NFC/NFD/NFKC/NFKD) via
    % the JVM bundled with MATLAB (java.text.Normalizer). Default NFKC folds
    % compatibility characters (full-width letters, ligatures, superscripts).
    properties (SetAccess = private)
        Form (1,1) string = "NFKC"
    end
    properties (Constant)
        Reference = "Unicode Standard Annex #15 normalization forms (java.text.Normalizer)"
        InDomain = "raw-text"
        OutDomain = "raw-text"
    end
    methods
        function obj = UnicodeNormalizeCharFilter(options)
            arguments
                options.Form (1,1) string {mustBeMember(options.Form, ["NFC","NFD","NFKC","NFKD"])} = "NFKC"
            end
            obj.Form = options.Form;
        end
        function out = forward(obj, text)
            arguments
                obj
                text (1,1) string
            end
            out = OceanNLP.Text.Unicode.normalize(text, obj.Form);
        end
        function s = name(obj), s = "unicodenormalize:" + lower(obj.Form); end
    end
    methods (Access = protected)
        function a = assumptions(obj)
            a = {char("normalization form " + obj.Form), 'requires the MATLAB-bundled JVM'};
        end
    end
end
