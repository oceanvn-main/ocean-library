classdef HtmlStripCharFilter < OceanNLP.Text.ICharFilter
    % HTMLSTRIPCHARFILTER  Removes markup (Lucene HTMLStripCharFilter, simplified):
    % <script>/<style> blocks are dropped with their content, other tags are
    % replaced by a space, and the common named/numeric entities are decoded.
    % Not a full HTML parser: malformed markup is handled leniently, not validated.

    % Example: <h1>Hello &amp; Welcome!</h1><script>console.log("test");</script><p>Caf&#233;</p>
    % ->  Hello & Welcome!   Café 
    properties (Constant)
        Reference = "simplified Lucene HTMLStripCharFilter: tags -> space, script/style dropped, common entities decoded"
        InDomain = "raw-text"
        OutDomain = "raw-text"
    end
    methods
        function out = forward(~, text)
            arguments
                ~
                text (1,1) string
            end
            s = char(text);
            s = regexprep(s, '<(script|style)[^>]*>.*?</\1\s*>', ' ', 'ignorecase');
            s = regexprep(s, '<!--.*?-->', ' ');
            s = regexprep(s, '<[^>]*>', ' ');
            s = regexprep(s, '&#(\d+);', '${OceanNLP.Text.CharFilter.HtmlStripCharFilter.entity_($1, 10)}');
            s = regexprep(s, '&#[xX]([0-9a-fA-F]+);', '${OceanNLP.Text.CharFilter.HtmlStripCharFilter.entity_($1, 16)}');
            s = strrep(s, '&nbsp;', ' ');
            s = strrep(s, '&lt;', '<');
            s = strrep(s, '&gt;', '>');
            s = strrep(s, '&quot;', '"');
            s = strrep(s, '&apos;', '''');
            s = strrep(s, '&amp;', '&');              % last, so "&amp;lt;" -> "&lt;" (not "<")
            out = string(s);
        end
        function s = name(~), s = "htmlstrip"; end
    end

    methods (Static)
        function c = entity_(digits, base)
            % ENTITY_  numeric character reference -> character. Handles code points above U+FFFF
            % (surrogate pair) and maps invalid ones (0, surrogates, > U+10FFFF, absurdly long) to U+FFFD.
            if numel(digits) > 8
                cp = Inf;
            elseif base == 16
                cp = hex2dec(digits);
            else
                cp = str2double(digits);
            end
            if cp == 0 || cp > 1114111 || (cp >= 55296 && cp <= 57343)
                cp = 65533;
            end
            c = char(OceanNLP.Text.Unicode.fromCodepoints(cp));
        end
    end
    methods (Access = protected)
        function a = assumptions(~)
            a = {'tags replaced by a single space so adjacent words do not fuse', ...
                 'only nbsp, lt, gt, quot, apos, amp and numeric entities are decoded', ...
                 'not a validating HTML parser'};
        end
    end
end
