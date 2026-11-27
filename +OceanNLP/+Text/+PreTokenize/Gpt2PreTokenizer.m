classdef Gpt2PreTokenizer < OceanNLP.Text.IPreTokenizer
    % GPT2PRETOKENIZER  GPT-2's pre-tokenization regex, implemented as an explicit
    % scanner over Unicode code points (MATLAB's regexp cannot evaluate \p{L}/\p{N}
    % beyond ASCII, so the pattern is realised by hand with JVM character classes):
    %
    %   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    %
    % Alternatives are tried in this order at each position (leftmost match,
    % first alternative wins). The optional " ?" is a literal U+0020 only.
    % \s is Unicode whitespace; \p{L} = categories L*, \p{N} = N*.
    % `\s+(?!\S)` matches a whitespace run but leaves its LAST character for the
    % next token when a non-space follows (so " word" keeps its leading space).
    %
    % Output pieces are raw text; OceanNLP.Subword.ByteLevel maps their UTF-8
    % bytes to the printable alphabet before BPE.
    properties (Constant)
        Reference = "GPT-2 pre-tokenization pattern (Radford et al. 2019; HF ByteLevel pre-tokenizer)"
        InDomain = "raw-text"
        OutDomain = "token-sequence"
    end
    methods
        function words = forward(~, text)
            arguments
                ~
                text (1,1) string
            end
            U = OceanNLP.Text.Unicode();
            cps = U.codepoints(text);
            n = numel(cps);
            words = strings(1, 0);
            if n == 0, return; end
            isL = U.isLetter(cps);
            isN = U.isNumber(cps);
            isS = U.isSpace(cps);
            isO = ~(isS | isL | isN);                       % [^\s\p{L}\p{N}]
            i = 1;
            while i <= n
                e = 0;                                       % end index of the match (0 = no match yet)
                % 1. contractions
                if cps(i) == 39
                    e = Gpt2PreTokenizerContraction_(cps, i);
                end
                % 2-4. optional single space + run of one class
                if e == 0
                    j = i;
                    if cps(j) == 32 && j < n, j = j + 1; end
                    for cls = {isL, isN, isO}
                        c = cls{1};
                        if c(j)
                            k = j;
                            while k < n && c(k + 1), k = k + 1; end
                            e = k;
                            break
                        end
                    end
                end
                % 5-6. whitespace: \s+(?!\S) then \s+
                if e == 0 && isS(i)
                    k = i;
                    while k < n && isS(k + 1), k = k + 1; end
                    if k == n
                        e = k;                               % run reaches the end
                    elseif k > i
                        e = k - 1;                           % backtrack: leave last space for next token
                    else
                        e = k;                               % single space before non-space (\s+)
                    end
                end
                if e == 0, e = i; end                        % unreachable safety net
                words(end+1) = U.fromCodepoints(cps(i:e)); %#ok<AGROW>
                i = e + 1;
            end
        end
        function s = name(~), s = "gpt2"; end
    end
end

function e = Gpt2PreTokenizerContraction_(cps, i)
    % Returns the end index if a contraction 's 't 're 've 'm 'll 'd starts at i, else 0.
    n = numel(cps);
    e = 0;
    if i + 1 > n, return; end
    c1 = cps(i + 1);
    if any(c1 == double('stmd'))
        e = i + 1;
    elseif i + 2 <= n
        c2 = cps(i + 2);
        if (c1 == double('r') && c2 == double('e')) || (c1 == double('v') && c2 == double('e')) || ...
           (c1 == double('l') && c2 == double('l'))
            e = i + 2;
        end
    end
end
