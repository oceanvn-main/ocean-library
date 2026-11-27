classdef VisionTransformerTest < OceanTest.BaseTestCase
    % VISIONTRANSFORMERTEST  ViT and its training infrastructure: patch
    %   embedding identity, resolution flexibility, gradients, attention
    %   rollout, AdamW / schedule / label smoothing / augmentation, a real
    %   learning run on USPS digits, and the AffectNet data-quality reader.

    methods (Static)
        function h = fileHashes(files)
            md = java.security.MessageDigest.getInstance('MD5');
            h = strings(numel(files), 1);
            for k = 1:numel(files)
                fid = fopen(files(k), 'r'); b = fread(fid, Inf, '*uint8'); fclose(fid);
                md.reset(); h(k) = string(sprintf('%02x', typecast(md.digest(b), 'uint8')));
            end
        end
    end

    methods (Test)
        function testPatchEmbeddingIsPatchifyPlusMatrixProduct(testCase)
            s = RandStream('mt19937ar', 'Seed', 1);
            pe = OceanDL.Vision.PatchEmbedding(2, 5, 3, 'Seed', 1);
            X = randn(s, 6, 9, 2, 2);                                    % 2 x 3 grid of 3x3 patches
            [T, grid] = pe.forward(X);
            testCase.verifyEqual(grid, [2 3]);
            testCase.verifySize(T.Data, [5 2 6]);
            E = reshape(pe.Proj.Weight.Data, [], 5).';                  % (D x p*p*C), column-major (kh,kw,c)
            r = 2; c = 3; n = 2;                                         % patch (row 2, col 3) of image 2
            patch = X((r-1)*3 + (1:3), (c-1)*3 + (1:3), :, n);
            t = r + 2 * (c - 1);                                         % column-major token order
            testCase.verifyEqual(T.Data(:, n, t), E * patch(:) + pe.Proj.Bias.Data, 'AbsTol', 1e-12);
        end

        function testViTShapesPoolingPositionsAndResolution(testCase)
            X = randn(16, 16, 1, 3);
            for pos = ["learned", "sincos", "none"]
                for pool = ["cls", "mean"]
                    vit = OceanDL.Vision.ViT('ImageSize', [16 16], 'Channels', 1, 'PatchSize', 4, 'Dim', 16, ...
                        'Depth', 2, 'Heads', 2, 'NumClasses', 10, 'Positions', pos, 'Pooling', pool, 'Seed', 1);
                    testCase.verifySize(vit.forward(X).Data, [10 3], char(pos + "/" + pool));
                end
            end
            vit = OceanDL.Vision.ViT('ImageSize', [16 16], 'Channels', 1, 'PatchSize', 4, 'Dim', 16, ...
                'Depth', 1, 'Heads', 2, 'Positions', "sincos", 'Seed', 2);
            testCase.verifySize(vit.forward(randn(32, 24, 1, 2)).Data, [10 2]);  % other resolution works
            learned = OceanDL.Vision.ViT('ImageSize', [16 16], 'Channels', 1, 'PatchSize', 4, 'Dim', 16, 'Depth', 1, 'Heads', 2);
            testCase.verifyError(@() learned.forward(randn(32, 32, 1, 1)), ...
                'OceanDL:Transformer:LearnedPositionalEmbedding:Length');
        end

        function testViTInputGradient(testCase)
            vit = OceanDL.Vision.ViT('ImageSize', [8 8], 'Channels', 1, 'PatchSize', 4, 'Dim', 8, ...
                'Depth', 2, 'Heads', 2, 'NumClasses', 3, 'Seed', 3);
            r = OceanDL.Autograd.GradCheck.check(@(x) vit.forward(x), {randn(8, 8, 1, 2)});
            testCase.verifyTrue(r.Passed, sprintf('max err %.3g', r.MaxAbsError));
        end

        function testAttentionRolloutIsADistributionOverPatches(testCase)
            vit = OceanDL.Vision.ViT('ImageSize', [16 16], 'Channels', 1, 'PatchSize', 4, 'Dim', 16, ...
                'Depth', 3, 'Heads', 4, 'Seed', 4);
            maps = OceanDL.Vision.AttentionRollout.compute(vit, randn(16, 16, 1, 5));
            testCase.verifySize(maps, [4 4 5]);
            testCase.verifyTrue(all(maps(:) >= 0));
            testCase.verifyEqual(squeeze(sum(maps, [1 2])), ones(5, 1), 'AbsTol', 1e-12);
            testCase.verifyEmpty(vit.Encoder.Layers{1}.Attention.LastAttention(:, :, :, []));  % storage is off again
            testCase.verifyFalse(vit.Encoder.Layers{1}.Attention.StoreAttention);
        end

        function testAdamWDecayIsDecoupled(testCase)
            w = [1 -2 3];
            adamw = OceanStats.Optim.AdamWOptimizer('LR', 0.1, 'WeightDecay', 0.5);
            testCase.verifyEqual(adamw.step(w, zeros(1, 3)), w * (1 - 0.1 * 0.5), 'AbsTol', 1e-15);   % zero grad: pure decay
            a = OceanStats.Optim.AdamOptimizer('LR', 0.01); b = OceanStats.Optim.AdamWOptimizer('LR', 0.01, 'WeightDecay', 0);
            g = [0.3 -1 2];
            testCase.verifyEqual(b.step(w, g), a.step(w, g), 'AbsTol', 1e-15);                        % wd = 0 is Adam
        end

        function testForModuleExcludesNormBiasPositionAndCls(testCase)
            vit = OceanDL.Vision.ViT('ImageSize', [8 8], 'Channels', 1, 'PatchSize', 4, 'Dim', 8, 'Depth', 1, 'Heads', 2);
            opt = OceanDL.optim.ModuleOptimizer.forModule(vit, @(wd) OceanStats.Optim.AdamWOptimizer('WeightDecay', wd), ...
                'WeightDecay', 0.05);
            named = vit.namedParameters();
            wd = cellfun(@(o) o.WeightDecay, opt.Optimizers);
            names = string({named.Name});
            testCase.verifyEqual(wd(names == "ClsToken"), 0);
            testCase.verifyEqual(wd(names == "PosEmbed.Weight"), 0);
            testCase.verifyTrue(all(wd(endsWith(names, "Bias")) == 0));
            testCase.verifyTrue(all(wd(contains(names, "Norm")) == 0));
            testCase.verifyEqual(wd(names == "Head.Weight"), 0.05);
            testCase.verifyEqual(wd(names == "Encoder.Layers{1}.Attention.Query.Weight"), 0.05);
        end

        function testWarmupCosineSchedule(testCase)
            s = OceanDL.optim.WarmupCosine(1e-3, 'WarmupSteps', 10, 'TotalSteps', 110, 'MinLR', 1e-5);
            testCase.verifyEqual(s.at(5), 5e-4, 'AbsTol', 1e-15);
            testCase.verifyEqual(s.at(10), 1e-3, 'AbsTol', 1e-15);
            testCase.verifyEqual(s.at(60), 1e-5 + (1e-3 - 1e-5) / 2, 'AbsTol', 1e-15);   % half-way: cos = 0
            testCase.verifyEqual(s.at(110), 1e-5, 'AbsTol', 1e-15);
        end

        function testLabelSmoothingValueAndGradient(testCase)
            F = OceanDL.Autograd.F;
            z = [2; 0; -1]; e = 0.1;
            logp = z - log(sum(exp(z)));
            expected = -((1 - e) * logp(1) + e * mean(logp));
            L = F.crossEntropy(OceanDL.Autograd.Tensor(z), 1, 'LabelSmoothing', e);
            testCase.verifyEqual(L.Data, expected, 'AbsTol', 1e-14);
            r = OceanDL.Autograd.GradCheck.check(@(zz) F.crossEntropy(zz, [1 3 2], 'LabelSmoothing', 0.2), {randn(3, 3)});
            testCase.verifyTrue(r.Passed);
        end

        function testImageAugment(testCase)
            X = reshape(1:4*5*1*3, 4, 5, 1, 3);
            flip = OceanDL.Vision.ImageAugment('FlipProb', 1, 'Seed', 1);
            testCase.verifyEqual(flip.apply(X), X(:, end:-1:1, :, :));
            crop = OceanDL.Vision.ImageAugment('Padding', 1, 'Seed', 2);
            Y = crop.apply(X);
            testCase.verifySize(Y, size(X));
            for n = 1:3                                              % each result is a shifted (edge-padded) copy
                ok = false;
                Xp = X([1 1:4 4], [1 1:5 5], 1, n);
                for dy = 0:2, for dx = 0:2
                    ok = ok || isequal(Y(:, :, 1, n), Xp(dy + (1:4), dx + (1:5)));
                end, end
                testCase.verifyTrue(ok);
            end
        end

        function testViTLearnsDigits(testCase)
            % Real data through the import contract; a tiny ViT must clearly learn.
            import OceanDL.Layout.DimFormat
            [rows, y, fmt] = OceanData.Adapters.UspsDigitsReader.read("train");
            X = DimFormat.toCanonical(DimFormat.fromRowMajor(reshape(rows.', 1, []), fmt.Shape), fmt.Format);
            y = y + 1;
            tr = 1:1500; te = 1501:2000;
            vit = OceanDL.Vision.ViT('ImageSize', [16 16], 'Channels', 1, 'PatchSize', 4, 'Dim', 32, ...
                'Depth', 2, 'Heads', 2, 'NumClasses', 10, 'Seed', 5);
            opt = OceanDL.optim.ModuleOptimizer.forModule(vit, @(wd) OceanStats.Optim.AdamWOptimizer('LR', 2e-3, 'WeightDecay', wd));
            order = RandStream('mt19937ar', 'Seed', 6);
            for ep = 1:10                                      % measured: ~87% at 8 epochs, 77% at 6
                perm = tr(randperm(order, numel(tr)));
                for b = 1:floor(numel(tr) / 50)
                    idx = perm((b-1)*50 + (1:50));
                    opt.zeroGrad();
                    L = OceanDL.Autograd.F.crossEntropy(vit.forward(X(:, :, :, idx)), y(idx));
                    L.backward();
                    opt.step();
                end
            end
            vit.eval();
            [~, pred] = max(OceanDL.Autograd.GradMode.noGrad(@() vit.forward(X(:, :, :, te))).Data, [], 1);
            testCase.verifyGreaterThan(mean(pred(:) == y(te)), 0.80);
        end

        function testAffectNetCleanResplitProtocol(testCase)
            six = ["anger" "contempt" "disgust" "fear" "sad" "surprise"];
            args = {'Split', "resplit", 'Sources', "affectnet", 'Labels', "agreeing", 'Classes', six, 'ImageSize', [8 8]};
            [~, ytr, itr] = OceanData.Adapters.AffectNetReader.read("train", args{:});
            [~, yte, ite] = OceanData.Adapters.AffectNetReader.read("test", args{:});
            testCase.verifyEqual(itr.Classes, six);
            testCase.verifyEqual(itr.Report.PerClassSourceEligible.ffhq, zeros(6, 1));            % no FFHQ
            md5 = @(f) OceanTest.OceanDL.Vision.VisionTransformerTest.fileHashes(f);
            testCase.verifyEmpty(intersect(md5(itr.Files), md5(ite.Files)));                     % no shared image CONTENT
            nTr = accumarray(ytr, 1, [6 1]); nTe = accumarray(yte, 1, [6 1]);
            testCase.verifyEqual(nTe ./ (nTr + nTe), 0.2 * ones(6, 1), 'AbsTol', 0.002);         % stratified 20%
            testCase.verifyGreaterThan(nTe(six == "sad"), 100);                                   % sad is testable again
        end

        function testAffectNetReaderDataQuality(testCase)
            [X, y, info] = OceanData.Adapters.AffectNetReader.read("test", 'ImageSize', [48 48], 'MaxImages', 120, 'Seed', 1);
            testCase.verifySize(X, [48 48 3 120]);
            testCase.verifyClass(X, 'single');
            testCase.verifyTrue(all(X(:) >= 0 & X(:) <= 1));
            testCase.verifyTrue(all(y >= 1 & y <= 8));
            testCase.verifyEqual(info.Report.DuplicatesRemoved, 1688);          % measured leakage removed
            testCase.verifyEqual(info.Format, "SSCB");
            [~, yf, infoF] = OceanData.Adapters.AffectNetReader.read("train", 'Labels', "folder", 'MaxImages', 10);
            [~, ~, infoA] = OceanData.Adapters.AffectNetReader.read("train", 'Labels', "agreeing", 'MaxImages', 10);
            testCase.verifyEqual(numel(yf), 10);
            testCase.verifyGreaterThan(infoF.Report.Listed, 0);
            testCase.verifyEqual(infoA.Report.FolderCsvDisagree, infoF.Report.FolderCsvDisagree);
            testCase.verifyEqual(infoF.Report.Listed, 16108);                   % jpg AND png files are listed
            comp = infoF.Report.PerClassSourceEligible;                         % happy/neutral are FFHQ (source confound)
            testCase.verifyGreaterThan(comp{"happy", "ffhq"}, 50 * comp{"happy", "affectnet"});
            [~, ~, infoNoFfhq] = OceanData.Adapters.AffectNetReader.read("train", 'Labels', "folder", 'Sources', "affectnet", 'MaxImages', 10);
            testCase.verifyEqual(infoNoFfhq.Report.PerClassSourceEligible{"happy", "ffhq"}, 0);
        end
    end
end
