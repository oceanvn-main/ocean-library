classdef Unicode
    % UNICODE  Unicode helpers for the text-processing strategies.
    %
    % CHARACTER CATEGORIES come from a generated table (unicode_category_table.mat, Unicode 15.0,
    % produced from Python's unicodedata by +OceanTest/+OceanNLP/+Text/generate_unicode_table.py),
    % NOT from the JVM: MATLAB's bundled JVM is Java 8 (Unicode 6.2) and misclassifies tens of
    % thousands of newer letters/digits/punctuation marks, and MATLAB's own regexp cannot evaluate
    % \p{L}/\p{Mn} beyond ASCII (both verified). Category codes follow java.lang.Character:
    %   letters 1-5, marks 6-8, numbers 9-11, separators 12-14, other/control 0,15,16,18,19,
    %   punctuation 20-24 and 29-30, symbols 25-28.
    %
    % NORMALIZATION (NFC/NFD/NFKC/NFKD) still uses java.text.Normalizer (no MATLAB alternative),
    % so its decomposition data are those of the JVM's Unicode version (Java 8: 6.2). This only
    % affects characters added after that version. normalize() raises a clear error if the JVM
    % is unavailable (matlab -nojvm).
    %
    % Code points are handled as doubles; MATLAB chars are UTF-16 units, so surrogate pairs are
    % combined/split here. Functional core only (all static).

    methods (Static)
        function v = tableVersion()
            % TABLEVERSION  Unicode version of the category table, e.g. "15.0.0".
            v = string(OceanNLP.Text.Unicode.table_().version);
        end

        function out = normalize(text, form)
            % NORMALIZE  Unicode normalization; form in "NFC","NFD","NFKC","NFKD" (JVM-backed).
            arguments
                text (1,1) string
                form (1,1) string {mustBeMember(form, ["NFC","NFD","NFKC","NFKD"])} = "NFC"
            end
            if ~usejava('jvm')
                error('OceanNLP:Text:Unicode:NoJvm', ...
                    'Unicode normalization needs the MATLAB JVM (java.text.Normalizer); it is disabled (-nojvm).');
            end
            f = javaMethod('valueOf', 'java.text.Normalizer$Form', char(form));
            out = string(char(java.text.Normalizer.normalize(java.lang.String(char(text)), f)));
        end

        function cps = codepoints(text)
            % CODEPOINTS  string -> 1xN double code points (surrogate pairs combined).
            arguments
                text (1,1) string
            end
            u = double(char(text));
            cps = zeros(1, numel(u));
            k = 0; i = 1;
            while i <= numel(u)
                k = k + 1;
                if u(i) >= 55296 && u(i) <= 56319 && i < numel(u) && u(i+1) >= 56320 && u(i+1) <= 57343
                    cps(k) = 65536 + (u(i) - 55296) * 1024 + (u(i+1) - 56320);
                    i = i + 2;
                else
                    cps(k) = u(i);
                    i = i + 1;
                end
            end
            cps = cps(1:k);
        end

        function text = fromCodepoints(cps)
            % FROMCODEPOINTS  inverse of codepoints().
            arguments
                cps (1,:) double
            end
            u = zeros(1, 0);
            for c = cps
                if c >= 65536
                    c2 = c - 65536;
                    u(end+1:end+2) = [55296 + floor(c2 / 1024), 56320 + mod(c2, 1024)]; %#ok<AGROW>
                else
                    u(end+1) = c; %#ok<AGROW>
                end
            end
            text = string(char(u));
        end

        function t = category(cps)
            % CATEGORY  general-category code per code point (table lookup, Unicode 15).
            arguments
                cps (1,:) double
            end
            T = OceanNLP.Text.Unicode.table_();
            t = zeros(size(cps));
            ok = cps >= 0 & cps <= 1114111;
            if any(ok)
                bin = discretize(cps(ok), [double(T.starts); 1114112]);
                t(ok) = double(T.cats(bin));
            end
        end

        function out = lowerText(text)
            % LOWERTEXT  Per-character full Unicode lowercase using the Unicode 15 table (what HF
            % tokenizers' BertNormalizer does: char-wise, no final-sigma context rule). MATLAB's
            % lower() is not used because it lacks newer scripts (e.g. Adlam) and, unlike this,
            % is not tied to a stated Unicode version.
            arguments
                text (1,1) string
            end
            cps = OceanNLP.Text.Unicode.codepoints(text);
            if all(cps < 128)
                out = lower(text);
                return
            end
            T = OceanNLP.Text.Unicode.table_();
            [hit, loc] = ismember(cps, T.lowerSrc);
            if ~any(hit), out = text; return; end
            pieces = cell(1, numel(cps));
            for i = 1:numel(cps)
                if hit(i)
                    j = loc(i);
                    pieces{i} = double(T.lowerFlat(T.lowerStart(j) + (1:T.lowerLen(j))));
                else
                    pieces{i} = cps(i);
                end
            end
            out = OceanNLP.Text.Unicode.fromCodepoints([pieces{:}]);
        end

        function tf = isLetter(cps),  c = OceanNLP.Text.Unicode.category(cps); tf = c >= 1 & c <= 5;   end
        function tf = isMark(cps),    c = OceanNLP.Text.Unicode.category(cps); tf = c >= 6 & c <= 8;   end
        function tf = isNumber(cps),  c = OceanNLP.Text.Unicode.category(cps); tf = c >= 9 & c <= 11;  end

        function tf = isOther(cps)
            % ISOTHER  Unicode categories C*: Cc, Cf, Co, Cs and unassigned (Cn).
            c = OceanNLP.Text.Unicode.category(cps);
            tf = c == 0 | c == 15 | c == 16 | c == 18 | c == 19;
        end

        function tf = isSpace(cps)
            % ISSPACE  Unicode White_Space property (what regex \s and Rust \s match; exactly the
            % set HuggingFace's tokenizers treat as whitespace): U+0009-000D, 0020, 0085, 00A0,
            % 1680, 2000-200A, 2028, 2029, 202F, 205F, 3000. NOT U+001C-001F.
            arguments
                cps (1,:) double
            end
            tf = (cps >= 9 & cps <= 13) | cps == 32 | cps == 133 | cps == 160 | cps == 5760 | ...
                 (cps >= 8192 & cps <= 8202) | cps == 8232 | cps == 8233 | cps == 8239 | cps == 8287 | cps == 12288;
        end

        function tf = isPunctuation(cps)
            % ISPUNCTUATION  BERT is_punctuation: ASCII non-alphanumeric symbols (33-47, 58-64,
            % 91-96, 123-126) OR Unicode category P*.
            arguments
                cps (1,:) double
            end
            c = OceanNLP.Text.Unicode.category(cps);
            asciiSym = (cps >= 33 & cps <= 47) | (cps >= 58 & cps <= 64) | (cps >= 91 & cps <= 96) | (cps >= 123 & cps <= 126);
            tf = asciiSym | (c >= 20 & c <= 24) | c == 29 | c == 30;
        end
    end

    methods (Static, Access = private)
        function T = table_()
            persistent TT
            if isempty(TT)
                f = fullfile(fileparts(mfilename('fullpath')), 'unicode_category_table.mat');
                if ~isfile(f)
                    error('OceanNLP:Text:Unicode:TableMissing', ...
                        ['unicode_category_table.mat not found next to Unicode.m. Regenerate it with ' ...
                         '+OceanTest/+OceanNLP/+Text/generate_unicode_table.py.']);
                end
                d = load(f);
                TT = struct('starts', d.starts(:), 'cats', d.cats(:), 'version', char(d.unicode_version), ...
                    'lowerSrc', double(d.lower_src(:))', 'lowerStart', double(d.lower_start(:))', ...
                    'lowerLen', double(d.lower_len(:))', 'lowerFlat', double(d.lower_flat(:))');
            end
            T = TT;
        end
    end
end
