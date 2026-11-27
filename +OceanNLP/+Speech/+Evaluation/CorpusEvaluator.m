classdef CorpusEvaluator < handle
    % CORPUSEVALUATOR  WER + CER with per-utterance, per-speaker and worst-case breakdowns.
    %
    %   ev = OceanNLP.Speech.Evaluation.CorpusEvaluator('Normalizer', n, 'NumWorst', 10)
    %   r  = ev.evaluate(refs, hyps, 'Ids', ids, 'Speakers', spk)
    %
    %   r.WER, r.CER   structs from WordErrorRate.compute / CharacterErrorRate.compute
    %                  (corpus level: sum(S+D+I) / sum(N), never a mean of rates)
    %   r.PerUtterance table Id, Speaker, Reference, Hypothesis (normalized),
    %                  WordErrors, Words, WER, CharErrors, Chars, CER, EmptyReference
    %   r.PerSpeaker   table Speaker, NumUtterances, WordErrors, Words, WER,
    %                  CharErrors, Chars, CER -- WER_s = sum_{u in s} E_u / sum_{u in s} N_u
    %                  (NaN when the speaker's references are all empty)
    %   r.Worst        the NumWorst utterances with the highest WordErrors
    %                  (ties: higher utterance WER, then original order)
    %
    %   Speakers defaults to "" for every utterance (one anonymous group); Ids to
    %   "1".."U". The normalizer is applied once, inside WER/CER, to both sides.

    properties (SetAccess = immutable)
        Normalizer
        NumWorst (1,1) double
    end

    methods
        function obj = CorpusEvaluator(opts)
            arguments
                opts.Normalizer = []
                opts.NumWorst (1,1) double {mustBeInteger, mustBeNonnegative} = 10
            end
            if ~isempty(opts.Normalizer) && ~ismethod(opts.Normalizer, 'forward')
                error('OceanNLP:Speech:Evaluation:CorpusEvaluator:Normalizer', ...
                    'Normalizer must provide forward(stringArray).');
            end
            obj.Normalizer = opts.Normalizer;
            obj.NumWorst = opts.NumWorst;
        end

        function r = evaluate(obj, refs, hyps, opts)
            arguments
                obj
                refs string
                hyps string
                opts.Ids string = string.empty
                opts.Speakers string = string.empty
            end
            U = numel(refs);
            ids = reshape(opts.Ids, [], 1);
            spk = reshape(opts.Speakers, [], 1);
            if isempty(ids), ids = string((1:U)'); end
            if isempty(spk), spk = repmat("", U, 1); end
            if numel(ids) ~= U || numel(spk) ~= U
                error('OceanNLP:Speech:Evaluation:CorpusEvaluator:CountMismatch', ...
                    '%d utterances but %d ids and %d speakers.', U, numel(ids), numel(spk));
            end
            wer = OceanNLP.Speech.Evaluation.WordErrorRate.compute(refs, hyps, 'Normalizer', obj.Normalizer);
            cer = OceanNLP.Speech.Evaluation.CharacterErrorRate.compute(refs, hyps, 'Normalizer', obj.Normalizer);
            pw = wer.PerUtterance; pc = cer.PerUtterance;

            per = table(ids, spk, pw.Reference, pw.Hypothesis, pw.Errors, pw.N, pw.Rate, ...
                pc.Errors, pc.N, pc.Rate, pw.EmptyReference, 'VariableNames', ...
                ["Id", "Speaker", "Reference", "Hypothesis", "WordErrors", "Words", "WER", ...
                "CharErrors", "Chars", "CER", "EmptyReference"]);

            [g, names] = findgroups(spk);
            sumBy = @(x) splitapply(@sum, x, g);
            nu = splitapply(@numel, ids, g);
            we = sumBy(per.WordErrors); wn = sumBy(per.Words);
            ce = sumBy(per.CharErrors); cn = sumBy(per.Chars);
            perSpeaker = table(names, nu, we, wn, safeRatio(we, wn), ce, cn, safeRatio(ce, cn), ...
                'VariableNames', ["Speaker", "NumUtterances", "WordErrors", "Words", "WER", ...
                "CharErrors", "Chars", "CER"]);

            ord = (1:U)';
            werKey = per.WER; werKey(isnan(werKey)) = Inf;     % empty-ref rows with insertions rank high
            [~, idx] = sortrows([-per.WordErrors, -werKey, ord]);
            worst = per(idx(1:min(obj.NumWorst, U)), :);

            r = struct('WER', wer, 'CER', cer, 'PerUtterance', per, ...
                'PerSpeaker', perSpeaker, 'Worst', worst);
        end

        function cfg = config(obj)
            cfg = struct('Type', "CorpusEvaluator", 'NumWorst', obj.NumWorst, 'Normalizer', struct());
            if ~isempty(obj.Normalizer) && ismethod(obj.Normalizer, 'config')
                cfg.Normalizer = obj.Normalizer.config();
            end
        end
    end
end

function r = safeRatio(e, n)
% SAFERATIO  e ./ n with NaN where n = 0 (no reference tokens: rate undefined).
r = e ./ n;
r(n == 0) = NaN;
end
