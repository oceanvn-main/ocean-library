classdef CharacterErrorRate
    % CHARACTERERRORRATE  Corpus character error rate (jiwer.cer convention).
    %
    %   r = OceanNLP.Speech.Evaluation.CharacterErrorRate.compute(refs, hyps)
    %   r = OceanNLP.Speech.Evaluation.CharacterErrorRate.compute(refs, hyps, 'Normalizer', n)
    %   c = OceanNLP.Speech.Evaluation.CharacterErrorRate.tokenize(text)
    %
    %     CER = sum_u (S_u + D_u + I_u) / sum_u N_u,   N_u = characters of reference u
    %
    %   CHARACTER CONVENTION (identical to jiwer.transformations.cer_default):
    %   strip leading/trailing whitespace, then EVERY remaining character is a
    %   token -- internal spaces count as characters and runs of spaces are NOT
    %   collapsed (jiwer's cer_default does not collapse them). Characters are
    %   Unicode code points (UTF-16 surrogate pairs are combined), matching
    %   Python str iteration. Case and punctuation are kept unless a Normalizer
    %   (applied to both sides first) removes them.
    %
    %   sum_u N_u = 0 raises '...:CharacterErrorRate:EmptyReference' (jiwer 4.0
    %   returns I instead). Per-utterance empty references are flagged, not hidden.
    %   Output struct: as WordErrorRate.compute.

    methods (Static)
        function r = compute(refs, hyps, opts)
            arguments
                refs string
                hyps string
                opts.Normalizer = []
            end
            [refs, hyps] = OceanNLP.Speech.Evaluation.WordErrorRate.prepare(refs, hyps, opts.Normalizer, 'CharacterErrorRate');
            tok = @OceanNLP.Speech.Evaluation.CharacterErrorRate.tokenize;
            refT = arrayfun(tok, refs, 'UniformOutput', false);
            hypT = arrayfun(tok, hyps, 'UniformOutput', false);
            r = OceanNLP.Speech.Evaluation.WordErrorRate.summarize(refs, hyps, refT, hypT, ...
                'CharacterErrorRate', "jiwer.cer_default: strip, every code point (spaces included) is a token; corpus rate = sum(S+D+I)/sum(N)");
        end

        function cp = tokenize(text)
            % TOKENIZE  Code points of strip(text) as a 1 x L double row.
            arguments
                text (1,1) string
            end
            u = double(char(strip(text)));
            if isempty(u)
                cp = zeros(1, 0);
                return;
            end
            hi = u >= 55296 & u <= 56319;                         % high surrogates 0xD800-0xDBFF
            lo = [u(2:end) >= 56320 & u(2:end) <= 57343, false];   % low surrogates 0xDC00-0xDFFF
            pair = hi & lo;
            cp = u;
            k = find(pair);
            cp(k) = (u(k) - 55296) * 1024 + (u(k+1) - 56320) + 65536;   % 0xD800, 0x400, 0xDC00, 0x10000
            cp(k + 1) = [];
        end
    end
end
