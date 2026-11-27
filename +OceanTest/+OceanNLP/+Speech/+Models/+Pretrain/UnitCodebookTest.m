classdef UnitCodebookTest < OceanTest.BaseTestCase
    % UNITCODEBOOKTEST  k-means++/Lloyd codebook: reproducibility + Version, Lloyd fixed point
    %   (brute-force nearest centroid, centroid = cluster mean), recovery of separated clusters,
    %   CMVN invariance, state round trip / tamper detection, real FSDD log-mel frames.

    methods (Test)
        function testReproducibleAndVersioned(testCase)
            [seqs, ids] = testCase.blobs(1);
            a = OceanNLP.Speech.Models.Pretrain.UnitCodebook(4, 'Seed', 3, 'Normalize', "none").fit(seqs, 'Ids', ids);
            b = OceanNLP.Speech.Models.Pretrain.UnitCodebook(4, 'Seed', 3, 'Normalize', "none").fit(seqs, 'Ids', ids);
            testCase.verifyEqual(b.Centroids, a.Centroids);
            testCase.verifyEqual(b.Version, a.Version);
            testCase.verifyEqual(strlength(a.Version), 16);
            % id order does not matter (sorted), the id SET does
            p = numel(seqs):-1:1;
            c = OceanNLP.Speech.Models.Pretrain.UnitCodebook(4, 'Seed', 3, 'Normalize', "none").fit(seqs(p), 'Ids', ids(p));
            testCase.verifyEqual(c.Centroids, a.Centroids, 'AbsTol', 1e-12);
            % different data -> different Version
            d = OceanNLP.Speech.Models.Pretrain.UnitCodebook(4, 'Seed', 3, 'Normalize', "none").fit(seqs(1:end-1), 'Ids', ids(1:end-1));
            testCase.verifyNotEqual(d.Version, a.Version);
            % same frames, different ids -> different Version
            e = OceanNLP.Speech.Models.Pretrain.UnitCodebook(4, 'Seed', 3, 'Normalize', "none").fit(seqs, 'Ids', ids + "x");
            testCase.verifyEqual(e.Centroids, a.Centroids);
            testCase.verifyNotEqual(e.Version, a.Version);
            % different feature config -> different Version
            f = OceanNLP.Speech.Models.Pretrain.UnitCodebook(4, 'Seed', 3, 'Normalize', "none").fit(seqs, 'Ids', ids, ...
                'FeatureConfig', struct('Type', "Other"));
            testCase.verifyNotEqual(f.Version, a.Version);
        end

        function testLloydFixedPointBruteForce(testCase)
            rs = RandStream('mt19937ar', 'Seed', 11);
            seqs = arrayfun(@(i) randn(rs, 3, 40), 1:5, 'UniformOutput', false);
            cb = OceanNLP.Speech.Models.Pretrain.UnitCodebook(6, 'Seed', 0, 'Tol', 0, 'MaxIter', 500, 'Normalize', "none");
            cb.fit(seqs, 'Ids', "s" + (1:5));
            X = [seqs{:}];
            C = cb.Centroids;
            % brute-force nearest centroid (explicit loops)
            z = zeros(1, size(X, 2));
            for m = 1:size(X, 2)
                best = inf;
                for k = 1:6
                    d = sum((X(:, m) - C(:, k)).^2);
                    if d < best, best = d; z(m) = k; end
                end
            end
            u = cb.assign(seqs);
            testCase.verifyEqual([u{:}], z);
            for k = 1:6                                  % converged: every centroid = mean of its frames
                testCase.verifyEqual(C(:, k), mean(X(:, z == k), 2), 'AbsTol', 1e-12);
            end
            testCase.verifyTrue(all(diff(cb.Inertia) <= 1e-12), 'Lloyd inertia must not increase');
            Jfinal = mean(sum((X - C(:, z)).^2, 1));
            testCase.verifyEqual(cb.Inertia(end), Jfinal, 'RelTol', 1e-10);
        end

        function testRecoversSeparatedClusters(testCase)
            [seqs, ids, labels] = testCase.blobs(2);
            cb = OceanNLP.Speech.Models.Pretrain.UnitCodebook(4, 'Seed', 0, 'Normalize', "none").fit(seqs, 'Ids', ids);
            u = cb.assign(seqs);
            u = [u{:}]; y = [labels{:}];
            % purity 1: each true cluster maps to exactly one unit and vice versa
            testCase.verifyEqual(size(unique([u; y].', 'rows'), 1), 4);
            s = cb.usage(seqs);
            testCase.verifyEqual(s.UsedUnits, 4);
            p = accumarray(y(:), 1).' / numel(y);
            testCase.verifyEqual(s.Entropy, -sum(p .* log2(p)), 'AbsTol', 1e-12);
        end

        function testUtteranceNormalizationInvariance(testCase)
            [seqs, ids] = testCase.blobs(3);
            cb = OceanNLP.Speech.Models.Pretrain.UnitCodebook(4, 'Seed', 1).fit(seqs, 'Ids', ids);
            shifted = 3 * seqs{2} + 7;                  % gain + offset (channel) change
            testCase.verifyEqual(cb.assign(shifted), cb.assign(seqs{2}));
        end

        function testStateRoundTripAndTamper(testCase)
            [seqs, ids] = testCase.blobs(4);
            cb = OceanNLP.Speech.Models.Pretrain.UnitCodebook(3, 'Seed', 2).fit(seqs, 'Ids', ids);
            sd = cb.stateDict();
            cb2 = OceanNLP.Speech.Models.Pretrain.UnitCodebook.fromStateDict(sd);
            testCase.verifyEqual(cb2.Version, cb.Version);
            testCase.verifyEqual(cb2.assign(seqs{1}), cb.assign(seqs{1}));
            sd.Centroids(1) = sd.Centroids(1) + 1e-9;
            testCase.verifyError(@() OceanNLP.Speech.Models.Pretrain.UnitCodebook.fromStateDict(sd), ...
                'OceanNLP:Speech:Models:UnitCodebook:VersionMismatch');
        end

        function testRealFSDDLogMel(testCase)
            % real audio: features from FSDD recordings, ids from the FeatureSequence sources
            seqs = testCase.fsddFeatures(["0_jackson_0", "1_nicolas_0", "2_yweweler_0", "3_jackson_1", "4_nicolas_1", "5_yweweler_1"]);
            cb = OceanNLP.Speech.Models.Pretrain.UnitCodebook(8, 'Seed', 0).fit(seqs);
            testCase.verifyEqual(cb.TrainingIds, sort(cellfun(@(s) s.Source, seqs)).');
            testCase.verifyEqual(cb.FeatureConfig.Type, "LogMel");
            testCase.verifyFalse(isfield(cb.FeatureConfig, 'InputHistory'));
            u = cb.assign(seqs{1});
            testCase.verifyEqual(numel(u), seqs{1}.numFrames());
            testCase.verifyTrue(all(u >= 1 & u <= 8));
            cb2 = OceanNLP.Speech.Models.Pretrain.UnitCodebook(8, 'Seed', 0).fit(seqs);
            testCase.verifyEqual(cb2.Version, cb.Version);
            % a feature sequence of a different configuration is refused
            mf = OceanNLP.Speech.Features.LogMel(8000, 'NumMels', 40, 'FMax', 3000);
            w = OceanNLP.Speech.Core.Waveform.read("recordings/6_theo_0.wav", 'Root', fullfile(OceanNLP.Speech.Data.SpeechManifest.defaultRoot(), "FSDD"));
            testCase.verifyError(@() cb.assign(mf.forward(w)), 'OceanNLP:Speech:Models:UnitCodebook:FeatureMismatch');
        end

        function testErrors(testCase)
            testCase.verifyError(@() OceanNLP.Speech.Models.Pretrain.UnitCodebook(3).fit({randn(2, 5)}), ...
                'OceanNLP:Speech:Models:UnitCodebook:Ids');
            testCase.verifyError(@() OceanNLP.Speech.Models.Pretrain.UnitCodebook(5).fit({ones(2, 9)}, 'Ids', "a"), ...
                'OceanNLP:Speech:Models:UnitCodebook:TooFewDistinct');
            testCase.verifyError(@() OceanNLP.Speech.Models.Pretrain.UnitCodebook(2).assign(randn(2, 3)), ...
                'OceanNLP:Speech:Models:UnitCodebook:NotFitted');
        end
    end

    methods (Static)
        function [seqs, ids, labels] = blobs(seed)
            % Four well-separated Gaussian blobs in 2-D, split into 6 "utterances".
            rs = RandStream('mt19937ar', 'Seed', seed);
            mu = [0 0; 10 0; 0 10; 10 10].';
            seqs = cell(1, 6); labels = cell(1, 6);
            for i = 1:6
                y = randi(rs, 4, 1, 30);
                seqs{i} = mu(:, y) + 0.3 * randn(rs, 2, 30);
                labels{i} = y;
            end
            ids = "utt" + (1:6);
        end

        function seqs = fsddFeatures(names)
            root = fullfile(OceanNLP.Speech.Data.SpeechManifest.defaultRoot(), "FSDD");
            lm = OceanNLP.Speech.Features.LogMel(8000, 'NumMels', 40);
            seqs = arrayfun(@(n) lm.forward(OceanNLP.Speech.Core.Waveform.read("recordings/" + n + ".wav", 'Root', root)), ...
                names, 'UniformOutput', false);
        end
    end
end
