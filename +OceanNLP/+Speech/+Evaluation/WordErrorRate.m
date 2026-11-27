classdef WordErrorRate
    % WORDERRORRATE  Corpus word error rate from Levenshtein alignments.
    %
    %   r = OceanNLP.Speech.Evaluation.WordErrorRate.compute(refs, hyps)
    %   r = OceanNLP.Speech.Evaluation.WordErrorRate.compute(refs, hyps, 'Normalizer', n)
    %   w = OceanNLP.Speech.Evaluation.WordErrorRate.tokenize(text)
    %
    %   refs, hyps: string arrays with one utterance per element (same count).
    %   Normalizer: optional object with forward(stringArray) (e.g.
    %   OceanNLP.Speech.Data.TranscriptNormalizer), applied to BOTH sides first.
    %
    %   For utterance u with reference length N_u and alignment counts
    %   (S_u, D_u, I_u, H_u) from EditAlignment (N_u = S_u + D_u + H_u):
    %
    %     WER = sum_u (S_u + D_u + I_u) / sum_u N_u        (corpus level, micro average)
    %
    %   -- NOT the mean of per-utterance rates. sum_u N_u = 0 raises
    %   'OceanNLP:Speech:Evaluation:WordErrorRate:EmptyReference' (deliberate:
    %   jiwer 4.0 instead returns WER = I in that case, a count, not a rate).
    %   An individual empty reference is kept in the corpus sums (its hypothesis
    %   words are insertions) and flagged in PerUtterance.EmptyReference, with
    %   PerUtterance.Rate = NaN for that row.
    %
    %   TOKENIZATION (identical to jiwer.transformations.wer_default):
    %   runs of >= 2 whitespace characters -> one space, strip leading/trailing
    %   whitespace, split on " ", drop empty pieces. (A single tab between two
    %   words therefore does not split them, exactly as in jiwer; use a
    %   TranscriptNormalizer to collapse all whitespace.)
    %
    %   OUTPUT (struct): Rate, S, D, I, H, N, NumUtterances, NumEmptyReferences,
    %   PerUtterance (table Reference, Hypothesis -- after normalization -- and the
    %   EditAlignment.corpusCounts columns), Convention.

    methods (Static)
        function r = compute(refs, hyps, opts)
            arguments
                refs string
                hyps string
                opts.Normalizer = []
            end
            [refs, hyps] = OceanNLP.Speech.Evaluation.WordErrorRate.prepare(refs, hyps, opts.Normalizer, 'WordErrorRate');
            tok = @OceanNLP.Speech.Evaluation.WordErrorRate.tokenize;
            refT = arrayfun(tok, refs, 'UniformOutput', false);
            hypT = arrayfun(tok, hyps, 'UniformOutput', false);
            r = OceanNLP.Speech.Evaluation.WordErrorRate.summarize(refs, hyps, refT, hypT, ...
                'WordErrorRate', "jiwer.wer_default word tokens; corpus rate = sum(S+D+I)/sum(N)");
        end

        function w = tokenize(text)
            % TOKENIZE  jiwer wer_default: collapse >=2 whitespace, strip, split on " ".
            arguments
                text (1,1) string
            end
            t = strip(regexprep(text, '\s\s+', ' '));
            if t == ""
                w = strings(1, 0);
                return;
            end
            w = split(t, " ").';
            w = w(strlength(w) > 0);
        end
    end

    methods (Static, Hidden)
        function [refs, hyps] = prepare(refs, hyps, normalizer, className)
            % Shared input checks and normalization (also used by CharacterErrorRate).
            id = "OceanNLP:Speech:Evaluation:" + className;
            refs = reshape(refs, 1, []); hyps = reshape(hyps, 1, []);
            if numel(refs) ~= numel(hyps)
                error(id + ":CountMismatch", '%d references but %d hypotheses.', numel(refs), numel(hyps));
            end
            if isempty(refs)
                error(id + ":EmptyReference", 'No reference utterances were given.');
            end
            if any(ismissing(refs)) || any(ismissing(hyps))
                error(id + ":Missing", 'References and hypotheses must not be <missing>.');
            end
            if ~isempty(normalizer)
                if ~ismethod(normalizer, 'forward')
                    error(id + ":Normalizer", 'Normalizer must provide forward(stringArray).');
                end
                refs = reshape(normalizer.forward(refs), 1, []);
                hyps = reshape(normalizer.forward(hyps), 1, []);
            end
        end

        function r = summarize(refs, hyps, refT, hypT, className, convention)
            c = OceanNLP.Speech.Evaluation.EditAlignment.corpusCounts(refT, hypT);
            if c.N == 0
                error("OceanNLP:Speech:Evaluation:" + className + ":EmptyReference", ...
                    ['All %d references are empty after normalization/tokenization; ', ...
                    'the error rate sum(S+D+I)/sum(N) is undefined.'], numel(refs));
            end
            per = [table(refs(:), hyps(:), 'VariableNames', ["Reference", "Hypothesis"]), c.PerUtterance];
            r = struct('Rate', (c.S + c.D + c.I) / c.N, 'S', c.S, 'D', c.D, 'I', c.I, ...
                'H', c.H, 'N', c.N, 'NumUtterances', numel(refs), ...
                'NumEmptyReferences', nnz(c.PerUtterance.EmptyReference), ...
                'PerUtterance', per, 'Convention', convention);
        end
    end
end
