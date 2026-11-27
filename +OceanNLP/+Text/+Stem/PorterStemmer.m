classdef PorterStemmer < OceanNLP.Text.IStemmer
    % PORTERSTEMMER  The original Porter (1980) suffix-stripping algorithm
    % ("An algorithm for suffix stripping", Program 14(3)), hand-rolled.
    %
    % Faithful to the PAPER, not to later "improved" variants: step 2 uses
    % "abli"->"able" (not "bli"->"ble"), there is no "logi"->"log" rule, step 1c
    % turns y->i only when the stem contains a vowel, and words of any length
    % are processed (no short-word shortcut). This is what NLTK calls
    % PorterStemmer(mode="ORIGINAL_ALGORITHM"), which is the validation reference.
    %
    % Rule semantics (the paper's): within a step only the rule with the
    % LONGEST matching suffix is considered; if its condition fails the word is
    % left unchanged by that step (no fall-through to shorter suffixes).
    %
    % Conditions, on the stem left after removing the suffix:
    %   m>0 / m>1 : "measure" of [C](VC)^m[V]        *v* : stem has a vowel
    %   *d : ends in a double consonant               *o : ends cvc, last c not w/x/y
    %
    % Input is lowercased. Functional core: PorterStemmer.stemWord(word).
    %
    % See also: OceanNLP.Text.Filter.StemFilter

    properties (Constant)
        Reference = "Porter, M.F. (1980) An algorithm for suffix stripping, Program 14(3):130-137 (original algorithm)"
        InDomain = "token-sequence"
        OutDomain = "token-sequence"
    end

    methods
        function s = stem(~, word)
            arguments
                ~
                word (1,1) string
            end
            s = string(OceanNLP.Text.Stem.PorterStemmer.stemWord(char(word)));
        end

        function s = name(~), s = "porter"; end
    end

    methods (Static)
        function w = stemWord(w)
            % STEMWORD  Pure function: char row -> stemmed char row.
            w = lower(w);
            w = OceanNLP.Text.Stem.PorterStemmer.step1a_(w);
            w = OceanNLP.Text.Stem.PorterStemmer.step1b_(w);
            w = OceanNLP.Text.Stem.PorterStemmer.step1c_(w);
            w = OceanNLP.Text.Stem.PorterStemmer.step2_(w);
            w = OceanNLP.Text.Stem.PorterStemmer.step3_(w);
            w = OceanNLP.Text.Stem.PorterStemmer.step4_(w);
            w = OceanNLP.Text.Stem.PorterStemmer.step5a_(w);
            w = OceanNLP.Text.Stem.PorterStemmer.step5b_(w);
        end
    end

    methods (Static, Access = private)
        function tf = isCons_(w, i)
            c = w(i);
            if any(c == 'aeiou'), tf = false; return; end
            if c == 'y'
                if i == 1, tf = true; else, tf = ~OceanNLP.Text.Stem.PorterStemmer.isCons_(w, i - 1); end
                return
            end
            tf = true;
        end

        function m = measure_(stem)
            n = numel(stem);
            cv = repmat('v', 1, n);
            for i = 1:n
                if OceanNLP.Text.Stem.PorterStemmer.isCons_(stem, i), cv(i) = 'c'; end
            end
            m = numel(strfind(cv, 'vc'));
        end

        function tf = hasVowel_(stem)
            tf = false;
            for i = 1:numel(stem)
                if ~OceanNLP.Text.Stem.PorterStemmer.isCons_(stem, i), tf = true; return; end
            end
        end

        function tf = endsDouble_(w)
            n = numel(w);
            tf = n >= 2 && w(n) == w(n-1) && OceanNLP.Text.Stem.PorterStemmer.isCons_(w, n);
        end

        function tf = endsCvc_(w)
            n = numel(w);
            tf = n >= 3 && OceanNLP.Text.Stem.PorterStemmer.isCons_(w, n-2) && ~OceanNLP.Text.Stem.PorterStemmer.isCons_(w, n-1) && OceanNLP.Text.Stem.PorterStemmer.isCons_(w, n) && ~any(w(n) == 'wxy');
        end

        function w = applyRules_(w, rules)
            % rules: Nx3 cell {suffix, replacement, conditionFcn or []}; suffix '*d' = double consonant.
            for r = 1:size(rules, 1)
                suf = rules{r, 1};  rep = rules{r, 2};  cond = rules{r, 3};
                if strcmp(suf, '*d')
                    if OceanNLP.Text.Stem.PorterStemmer.endsDouble_(w)
                        stem = w(1:end-2);
                        if isempty(cond) || cond(stem), w = [stem rep]; end
                        return
                    end
                elseif endsWithChar_(w, suf)
                    stem = w(1:numel(w)-numel(suf));
                    if isempty(cond) || cond(stem), w = [stem rep]; end
                    return
                end
            end
        end

        function w = step1a_(w)
            w = OceanNLP.Text.Stem.PorterStemmer.applyRules_(w, {'sses','ss',[]; 'ies','i',[]; 'ss','ss',[]; 's','',[]});
        end

        function w = step1b_(w)
            if endsWithChar_(w, 'eed')
                stem = w(1:end-3);
                if OceanNLP.Text.Stem.PorterStemmer.measure_(stem) > 0, w = [stem 'ee']; end
                return
            end
            ok = false;  inter = '';
            for suf = {'ed', 'ing'}
                if endsWithChar_(w, suf{1})
                    inter = w(1:numel(w)-numel(suf{1}));
                    if OceanNLP.Text.Stem.PorterStemmer.hasVowel_(inter), ok = true; break; end
                end
            end
            if ~ok, return; end
            last = inter(end);
            rules = {'at','ate',[]; 'bl','ble',[]; 'iz','ize',[]; ...
                     '*d', last, @(stem) ~any(last == 'lsz'); ...
                     '', 'e', @(stem) OceanNLP.Text.Stem.PorterStemmer.measure_(stem) == 1 && OceanNLP.Text.Stem.PorterStemmer.endsCvc_(stem)};
            w = OceanNLP.Text.Stem.PorterStemmer.applyRules_(inter, rules);
        end

        function w = step1c_(w)
            w = OceanNLP.Text.Stem.PorterStemmer.applyRules_(w, {'y','i',@(stem) OceanNLP.Text.Stem.PorterStemmer.hasVowel_(stem)});
        end

        function w = step2_(w)
            c = @(stem) OceanNLP.Text.Stem.PorterStemmer.measure_(stem) > 0;
            w = OceanNLP.Text.Stem.PorterStemmer.applyRules_(w, {'ational','ate',c; 'tional','tion',c; 'enci','ence',c; 'anci','ance',c; ...
                'izer','ize',c; 'abli','able',c; 'alli','al',c; 'entli','ent',c; 'eli','e',c; 'ousli','ous',c; ...
                'ization','ize',c; 'ation','ate',c; 'ator','ate',c; 'alism','al',c; 'iveness','ive',c; ...
                'fulness','ful',c; 'ousness','ous',c; 'aliti','al',c; 'iviti','ive',c; 'biliti','ble',c});
        end

        function w = step3_(w)
            c = @(stem) OceanNLP.Text.Stem.PorterStemmer.measure_(stem) > 0;
            w = OceanNLP.Text.Stem.PorterStemmer.applyRules_(w, {'icate','ic',c; 'ative','',c; 'alize','al',c; 'iciti','ic',c; 'ical','ic',c; 'ful','',c; 'ness','',c});
        end

        function w = step4_(w)
            g = @(stem) OceanNLP.Text.Stem.PorterStemmer.measure_(stem) > 1;
            ion = @(stem) OceanNLP.Text.Stem.PorterStemmer.measure_(stem) > 1 && ~isempty(stem) && any(stem(end) == 'st');
            w = OceanNLP.Text.Stem.PorterStemmer.applyRules_(w, {'al','',g; 'ance','',g; 'ence','',g; 'er','',g; 'ic','',g; 'able','',g; ...
                'ible','',g; 'ant','',g; 'ement','',g; 'ment','',g; 'ent','',g; 'ion','',ion; 'ou','',g; ...
                'ism','',g; 'ate','',g; 'iti','',g; 'ous','',g; 'ive','',g; 'ize','',g});
        end

        function w = step5a_(w)
            if endsWithChar_(w, 'e')
                stem = w(1:end-1);
                m = OceanNLP.Text.Stem.PorterStemmer.measure_(stem);
                if m > 1 || (m == 1 && ~OceanNLP.Text.Stem.PorterStemmer.endsCvc_(stem)), w = stem; end
            end
        end

        function w = step5b_(w)
            if endsWithChar_(w, 'll') && OceanNLP.Text.Stem.PorterStemmer.measure_(w(1:end-1)) > 1
                w = w(1:end-1);
            end
        end
    end
end

function tf = endsWithChar_(w, suf)
    n = numel(suf);
    tf = numel(w) >= n && strcmp(w(numel(w)-n+1:end), suf);
    if n == 0, tf = true; end
end
