classdef CorpusEvaluatorTest < OceanTest.BaseTestCase
    % CORPUSEVALUATORTEST  Per-utterance / per-speaker / worst breakdowns are consistent.

    methods (Test)
        function testBreakdownsSumToCorpus(testCase)
            f = fullfile(fileparts(mfilename('fullpath')), 'fixtures', 'jiwer_cases.json');
            fx = jsondecode(fileread(f));
            C = fx.cases;
            refs = string({C.ref}); hyps = string({C.hyp}); tags = string({C.tag});
            ev = OceanNLP.Speech.Evaluation.CorpusEvaluator('NumWorst', 7);
            r = ev.evaluate(refs, hyps, 'Speakers', tags, 'Ids', "u" + (1:numel(C)));

            testCase.verifyEqual(sum(r.PerUtterance.WordErrors), r.WER.S + r.WER.D + r.WER.I);
            testCase.verifyEqual(sum(r.PerSpeaker.Words), r.WER.N);
            testCase.verifyEqual(sum(r.PerSpeaker.CharErrors), r.CER.S + r.CER.D + r.CER.I);
            testCase.verifyEqual(sum(r.PerSpeaker.NumUtterances), numel(C));

            % each speaker's WER equals an independent WordErrorRate.compute on its rows
            for k = 1:height(r.PerSpeaker)
                s = r.PerSpeaker.Speaker(k);
                sel = tags == s;
                if all(r.PerUtterance.Words(sel) == 0)
                    testCase.verifyTrue(isnan(r.PerSpeaker.WER(k)));
                    continue;
                end
                w = OceanNLP.Speech.Evaluation.WordErrorRate.compute(refs(sel), hyps(sel));
                testCase.verifyEqual(r.PerSpeaker.WER(k), w.Rate, 'AbsTol', 1e-15, s);
            end

            testCase.verifyEqual(height(r.Worst), 7);
            testCase.verifyTrue(issorted(-r.Worst.WordErrors));
            testCase.verifyEqual(max(r.PerUtterance.WordErrors), r.Worst.WordErrors(1));
        end

        function testDefaultsAndMismatch(testCase)
            ev = OceanNLP.Speech.Evaluation.CorpusEvaluator('Normalizer', OceanNLP.Speech.Data.TranscriptNormalizer());
            r = ev.evaluate(["One!", "two three"], ["one", "two"]);
            testCase.verifyEqual(r.PerUtterance.Id, ["1"; "2"]);
            testCase.verifyEqual(height(r.PerSpeaker), 1);
            testCase.verifyEqual(r.WER.Rate, 1 / 3);
            testCase.verifyError(@() ev.evaluate("a", "a", 'Ids', ["x" "y"]), ...
                'OceanNLP:Speech:Evaluation:CorpusEvaluator:CountMismatch');
            testCase.verifyEqual(ev.config().Normalizer.Type, "TranscriptNormalizer");
        end
    end
end
