classdef HiddenLayerSelectorTest < OceanTest.BaseTestCase
    % HIDDENLAYERSELECTORTEST  NMI against a brute-force double loop + invariances; layer choice
    %   by clustering stability (well-separated structure beats isotropic noise, independent of
    %   layer order), bit-identical reproducibility, reference diagnostic, real hidden features
    %   of a HuBERTModel on FSDD audio, input validation.

    methods (Test)
        function testNMIBruteForce(testCase)
            rs = RandStream('mt19937ar', 'Seed', 5);
            a = randi(rs, 4, 1, 300); b = mod(a + (rand(rs, 1, 300) < 0.3) .* randi(rs, 3, 1, 300), 5) + 1;
            % brute force: explicit counts, natural logs
            n = numel(a); ua = unique(a); ub = unique(b);
            I = 0; Ha = 0; Hb = 0;
            for i = ua
                pi_ = sum(a == i) / n; Ha = Ha - pi_ * log(pi_);
                for j = ub
                    pj = sum(b == j) / n; pij = sum(a == i & b == j) / n;
                    if pij > 0, I = I + pij * log(pij / (pi_ * pj)); end
                end
            end
            for j = ub
                pj = sum(b == j) / n; Hb = Hb - pj * log(pj);
            end
            nmi = @OceanNLP.Speech.Models.Pretrain.HiddenLayerSelector.nmi;
            testCase.verifyEqual(nmi(a, b), I / sqrt(Ha * Hb), 'AbsTol', 1e-12);
            testCase.verifyEqual(nmi(b, a), nmi(a, b), 'AbsTol', 1e-12);              % symmetric
            perm = [3 1 4 2];
            testCase.verifyEqual(nmi(a, perm(a)), 1, 'AbsTol', 1e-12);                % relabelling invariant
            testCase.verifyEqual(nmi(a, 10 * a + 7), 1, 'AbsTol', 1e-12);
            testCase.verifyEqual(nmi(ones(1, 9), 2 * ones(1, 9)), 1);                 % both constant
            testCase.verifyEqual(nmi(ones(1, 9), [1 2 1 2 1 2 1 2 1]), 0);
            x = randi(rs, 5, 1, 2e5); y = randi(rs, 5, 1, 2e5);                       % independent
            testCase.verifyLessThan(nmi(x, y), 1e-3);
            testCase.verifyError(@() nmi([1 2], [1 2 3]), 'OceanNLP:Speech:Models:HiddenLayerSelector:Length');
        end

        function testPrefersStableLayerAndIsReproducible(testCase)
            [trS, vaS] = testCase.blobs(true);
            [trN, vaN] = testCase.blobs(false);
            ids = "u" + (1:numel(trS));
            sel = OceanNLP.Speech.Models.Pretrain.HiddenLayerSelector(4, 'Seeds', [0 1]);
            sel.fit({trN, trS}, {vaN, vaS}, 'Layers', [1 2], 'Ids', ids);
            testCase.verifyEqual(sel.Layer, 2);
            testCase.verifyGreaterThan(sel.Scores.Stability(2), 0.95);
            testCase.verifyLessThan(sel.Scores.Stability(1), sel.Scores.Stability(2));
            testCase.verifyEqual(sel.Codebook.Version, sel.Codebooks{2, 1}.Version);
            % layer order does not matter
            sel2 = OceanNLP.Speech.Models.Pretrain.HiddenLayerSelector(4, 'Seeds', [0 1]);
            sel2.fit({trS, trN}, {vaS, vaN}, 'Layers', [2 1], 'Ids', ids);
            testCase.verifyEqual(sel2.Layer, 2);
            % identical inputs + seeds -> identical scores, layer, codebook
            sel3 = OceanNLP.Speech.Models.Pretrain.HiddenLayerSelector(4, 'Seeds', [0 1]);
            sel3.fit({trN, trS}, {vaN, vaS}, 'Layers', [1 2], 'Ids', ids);
            testCase.verifyEqual(sel3.Scores, sel.Scores);
            testCase.verifyEqual(sel3.Codebook.Centroids, sel.Codebook.Centroids);
            testCase.verifyEqual(sel3.Codebook.Version, sel.Codebook.Version);
            % relative held-out inertia of separated blobs is small, of noise large
            testCase.verifyLessThan(sel.Scores.ValRelInertia(2), 0.05);
            testCase.verifyGreaterThan(sel.Scores.ValRelInertia(1), 0.2);
        end

        function testReferenceDiagnostic(testCase)
            [trS, vaS, labels] = testCase.blobs(true);
            sel = OceanNLP.Speech.Models.Pretrain.HiddenLayerSelector(4, 'Seeds', [0 1 2]);
            sel.fit({trS}, {vaS}, 'Layers', 5, 'Ids', "u" + (1:numel(trS)), 'Reference', labels);
            testCase.verifyEqual(sel.Layer, 5);
            testCase.verifyGreaterThan(sel.Scores.NMIWithReference, 0.95);   % recovers the true blobs
            testCase.verifyError(@() OceanNLP.Speech.Models.Pretrain.HiddenLayerSelector(4).fit({trS}, {vaS}, ...
                'Ids', "u" + (1:numel(trS)), 'Reference', {1:3}), 'OceanNLP:Speech:Models:HiddenLayerSelector:Reference');
        end

        function testRealHiddenFeatures(testCase)
            names = ["0_jackson_0", "1_jackson_0", "2_nicolas_0", "3_nicolas_0", "4_yweweler_0", "5_yweweler_0"];
            seqs = OceanTest.OceanNLP.Speech.Models.Pretrain.UnitCodebookTest.fsddFeatures(names);
            val = OceanTest.OceanNLP.Speech.Models.Pretrain.UnitCodebookTest.fsddFeatures(["3_jackson_46", "5_nicolas_47"]);
            hub = OceanNLP.Speech.Models.Pretrain.HuBERTModel(seqs{1}.numBins(), 'Dim', 16, 'Depth', 2, 'Heads', 2, ...
                'FFNDim', 32, 'ConvChannels', [16 16], 'NumUnits', 6, 'ProjDim', 8, 'Seed', 3);
            tr = OceanNLP.Speech.Models.Pretrain.MaskedUnitTrainer(hub, seqs, 'ValInputs', val, 'Verbose', false);
            trH = {tr.hiddenFeatures(1, "train"), tr.hiddenFeatures(2, "train")};
            vaH = {tr.hiddenFeatures(1, "val"), tr.hiddenFeatures(2, "val")};
            a = OceanNLP.Speech.Models.Pretrain.HiddenLayerSelector(6).fit(trH, vaH, 'Layers', [1 2]);
            b = OceanNLP.Speech.Models.Pretrain.HiddenLayerSelector(6).fit(trH, vaH, 'Layers', [1 2]);
            testCase.verifyEqual(b.Scores, a.Scores);
            testCase.verifyEqual(b.Layer, a.Layer);
            testCase.verifyTrue(all(a.Scores.Stability >= 0 & a.Scores.Stability <= 1 + 1e-12));
            [~, k] = max(a.Scores.Stability);
            testCase.verifyEqual(a.Layer, a.Scores.Layer(k));
            u = a.Codebook.assign(vaH{k});
            testCase.verifyEqual(cellfun(@numel, u), cellfun(@(s) s.numFrames(), vaH{k}));
        end

        function testValidation(testCase)
            testCase.verifyError(@() OceanNLP.Speech.Models.Pretrain.HiddenLayerSelector(4, 'Seeds', [1 1]), ...
                'OceanNLP:Speech:Models:HiddenLayerSelector:Seeds');
            [trS, vaS] = testCase.blobs(true);
            testCase.verifyError(@() OceanNLP.Speech.Models.Pretrain.HiddenLayerSelector(4).fit({trS, trS}, {vaS}, ...
                'Ids', "u" + (1:numel(trS))), 'OceanNLP:Speech:Models:HiddenLayerSelector:Count');
        end
    end

    methods (Static)
        function [tr, va, valLabels] = blobs(separated)
            % 4 well-separated Gaussian blobs in 3-D (separated) or one isotropic Gaussian.
            rs = RandStream('mt19937ar', 'Seed', 2 + separated);
            C = 12 * [1 0 0; 0 1 0; 0 0 1; -1 -1 -1].';
            make = @(n) arrayfun(@(i) frames(n), 1:6, 'UniformOutput', false);
            tr = make(50);
            [va, valLabels] = deal(cell(1, 3));
            for i = 1:3
                [va{i}, valLabels{i}] = frames(40);
            end
            function [X, z] = frames(n)
                z = randi(rs, 4, 1, n);
                if separated
                    X = C(:, z) + 0.5 * randn(rs, 3, n);
                else
                    X = 3 * randn(rs, 3, n);
                end
            end
        end
    end
end
