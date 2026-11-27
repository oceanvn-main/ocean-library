classdef WhitespacePreTokenizer < OceanNLP.Text.IPreTokenizer
    % WHITESPACEPRETOKENIZER  Split on runs of Unicode whitespace; nothing else
    % (no lowercasing, no punctuation handling). The simplest pre-tokenizer
    % (the "basic version").
    properties (Constant)
        Reference = "whitespace split (HF WhitespaceSplit pre-tokenizer)"
        InDomain = "raw-text"
        OutDomain = "token-sequence"
    end
    methods
        function words = forward(~, text)
            arguments
                ~
                text (1,1) string
            end
            cps = OceanNLP.Text.Unicode.codepoints(text);
            if isempty(cps), words = strings(1, 0); return; end
            sp = OceanNLP.Text.Unicode.isSpace(cps);
            words = strings(1, 0);
            i = 1;  n = numel(cps);
            while i <= n
                if sp(i), i = i + 1; continue; end
                j = i;
                while j < n && ~sp(j + 1), j = j + 1; end
                words(end+1) = OceanNLP.Text.Unicode.fromCodepoints(cps(i:j)); %#ok<AGROW>
                i = j + 1;
            end
        end
        function s = name(~), s = "whitespace"; end
    end
end
