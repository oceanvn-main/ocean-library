classdef ErrorRateTest < OceanTest.BaseTestCase
    % ERRORRATETEST  WordErrorRate / CharacterErrorRate vs jiwer 4 (fixture) + edge cases.

    properties
        Fixture
    end

    methods (TestClassSetup)
        function loadFixture(testCase)
            f = fullfile(fileparts(mfilename('fullpath')), 'fixtures', 'jiwer_cases.json');
            testCase.Fixture = jsondecode(fileread(f));
        end
    end

    methods (Test)
        function testCorpusTotalsEqualJiwer(testCase)
            K = testCase.Fixture.corpora;
            for k = 1:numel(K)
                refs = string(K(k).refs)'; hyps = string(K(k).hyps)';
                w = OceanNLP.Speech.Evaluation.WordErrorRate.compute(refs, hyps);
                c = OceanNLP.Speech.Evaluation.CharacterErrorRate.compute(refs, hyps);
                testCase.verifyEqual([w.S w.D w.I w.H], [K(k).word.S K(k).word.D K(k).word.I K(k).word.H], K(k).name);
                testCase.verifyEqual([c.S c.D c.I c.H], [K(k).char.S K(k).char.D K(k).char.I K(k).char.H], K(k).name);
                testCase.verifyEqual(w.Rate, K(k).wer, 'AbsTol', 1e-15, K(k).name);
                testCase.verifyEqual(c.Rate, K(k).cer, 'AbsTol', 1e-15, K(k).name);
                testCase.verifyEqual(w.N, w.S + w.D + w.H);
            end
        end

        function testRateIsMicroAverageNotMeanOfRates(testCase)
            refs = ["one", "two three four five"];
            hyps = ["two", "two three four five"];
            w = OceanNLP.Speech.Evaluation.WordErrorRate.compute(refs, hyps);
            per = w.PerUtterance;
            testCase.verifyEqual(w.Rate, sum(per.Errors) / sum(per.N));
            testCase.verifyNotEqual(w.Rate, mean(per.Rate));
        end

        function testEmptyTotalReferenceErrors(testCase)
            e = testCase.Fixture.empty_total_reference;
            refs = string(e.refs)'; hyps = string(e.hyps)';
            testCase.verifyError(@() OceanNLP.Speech.Evaluation.WordErrorRate.compute(refs, hyps), ...
                'OceanNLP:Speech:Evaluation:WordErrorRate:EmptyReference');
            testCase.verifyError(@() OceanNLP.Speech.Evaluation.CharacterErrorRate.compute(refs, hyps), ...
                'OceanNLP:Speech:Evaluation:CharacterErrorRate:EmptyReference');
            % jiwer 4 returns the insertion count instead of raising (documented difference)
            testCase.verifyEqual(e.wer, e.word.I);
        end

        function testPerUtteranceEmptyReferenceIsReported(testCase)
            w = OceanNLP.Speech.Evaluation.WordErrorRate.compute(["", "one two"], ["three", "one two"]);
            testCase.verifyEqual(w.NumEmptyReferences, 1);
            testCase.verifyTrue(w.PerUtterance.EmptyReference(1));
            testCase.verifyTrue(isnan(w.PerUtterance.Rate(1)));
            testCase.verifyEqual([w.I w.N], [1 2]);
            testCase.verifyEqual(w.Rate, 0.5);
        end

        function testCountMismatchAndNormalizer(testCase)
            testCase.verifyError(@() OceanNLP.Speech.Evaluation.WordErrorRate.compute(["a" "b"], "a"), ...
                'OceanNLP:Speech:Evaluation:WordErrorRate:CountMismatch');
            n = OceanNLP.Speech.Data.TranscriptNormalizer();
            raw = OceanNLP.Speech.Evaluation.WordErrorRate.compute("One, two.", "one two");
            normd = OceanNLP.Speech.Evaluation.WordErrorRate.compute("One, two.", "one two", 'Normalizer', n);
            testCase.verifyEqual(raw.S, 2);
            testCase.verifyEqual(normd.Rate, 0);
            testCase.verifyEqual(normd.PerUtterance.Reference, "one two");
        end

        function testCharacterConventionCountsSpacesAndCodePoints(testCase)
            T = @OceanNLP.Speech.Evaluation.CharacterErrorRate.tokenize;
            testCase.verifyEqual(numel(T("  a  b ")), 4);           % strip ends, keep inner run
            testCase.verifyEqual(T("a" + char([55357 56832])), [97 128512]);   % U+1F600 as a surrogate pair
            c = OceanNLP.Speech.Evaluation.CharacterErrorRate.compute("ab cd", "abcd");
            testCase.verifyEqual([c.D c.N], [1 5]);
        end
    end
end
