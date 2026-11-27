classdef SpeechBatcherTest < OceanTest.BaseTestCase
    % SPEECHBATCHERTEST  Padding/mask invariants, coverage, determinism per (Seed, epoch).

    properties
        DS
    end

    methods (TestClassSetup)
        function build(testCase)
            s = OceanTest.OceanNLP.Speech.Data.fsddTestManifest().canonicalFSDDSplit();
            va = s.bySplit("validation");
            m = va.subset(round(linspace(1, va.height(), 23)));
            vocab = OceanNLP.Speech.Data.SpeechVocabulary.fit(s.bySplit("train").Entries.Transcript);
            testCase.DS = OceanNLP.Speech.Data.AudioDataset(m, ...
                OceanTest.OceanNLP.Speech.Data.FrameEnergyCountingTestTransform(8000), 'Vocabulary', vocab);
        end
    end

    methods (Test)
        function testPaddingMaskAndContents(testCase)
            b = OceanNLP.Speech.Data.SpeechBatcher(testCase.DS, 'BatchSize', 5, 'Seed', 1);
            testCase.verifyEqual(b.numBatches(), 5);
            seen = [];
            for k = 1:b.numBatches()
                B = b.batch(k);
                N = numel(B.Indices);
                T = size(B.X, 3);
                testCase.verifyEqual(size(B.X, [1 2]), [2 N]);
                testCase.verifyEqual(T, max(B.Lengths));
                testCase.verifyEqual(B.Mask, (1:T) <= B.Lengths(:));
                for n = 1:N
                    it = testCase.DS.item(B.Indices(n));
                    L = B.Lengths(n);
                    testCase.verifyEqual(L, it.Features.numFrames());
                    testCase.verifyEqual(reshape(B.X(:, n, 1:L), 2, L), it.Features.Values);
                    testCase.verifyEqual(nnz(B.X(:, n, L+1:end)), 0);         % right zero-padding
                    testCase.verifyEqual(B.Targets{n}, it.TokenIds);
                    testCase.verifyEqual(B.Ids(n), it.Id);
                    testCase.verifyEqual(B.Text(n), it.Text);
                end
                seen = [seen, B.Indices]; %#ok<AGROW>
            end
            testCase.verifyEqual(sort(seen), 1:testCase.DS.length());
        end

        function testDeterministicPerSeedAndEpoch(testCase)
            g = rng;
            a = OceanNLP.Speech.Data.SpeechBatcher(testCase.DS, 'BatchSize', 4, 'Seed', 7);
            b = OceanNLP.Speech.Data.SpeechBatcher(testCase.DS, 'BatchSize', 4, 'Seed', 7);
            testCase.verifyEqual(a.Batches, b.Batches);
            e0 = a.Batches;
            a.setEpoch(1);
            testCase.verifyNotEqual(a.Batches, e0);
            a.setEpoch(0);
            testCase.verifyEqual(a.Batches, e0);
            testCase.verifyEqual(rng, g);                                  % global RNG untouched
            c = OceanNLP.Speech.Data.SpeechBatcher(testCase.DS, 'BatchSize', 4, 'Shuffle', false);
            testCase.verifyEqual([c.Batches{:}], 1:testCase.DS.length());
        end

        function testSortByLength(testCase)
            L = testCase.DS.lengths();
            s = OceanNLP.Speech.Data.SpeechBatcher(testCase.DS, 'BatchSize', 4, 'Shuffle', false, 'SortByLength', true);
            order = [s.Batches{:}];
            testCase.verifyTrue(issorted(L(order)));
            sh = OceanNLP.Speech.Data.SpeechBatcher(testCase.DS, 'BatchSize', 4, 'Shuffle', true, 'SortByLength', true, 'Seed', 3);
            testCase.verifyEqual(sort(cellfun(@(c) c(1), sh.Batches)), sort(cellfun(@(c) c(1), s.Batches)));
            for k = 1:sh.numBatches()                                     % batches stay length-homogeneous
                testCase.verifyTrue(any(cellfun(@(c) isequal(c, sh.Batches{k}), s.Batches)));
            end
        end
    end
end
