classdef MaskedAutoencoderTest < OceanTest.BaseTestCase
    % MASKEDAUTOENCODERTEST  MAE mechanics: per-sample token gathering,
    %   patch <-> pixel layout consistency with PatchEmbedding, the encoder
    %   really sees only the visible tokens, the loss ignores visible patches,
    %   pre-training reduces reconstruction error, the backbone fine-tunes.

    methods (Test)
        function testGatherIsExactAndDifferentiable(testCase)
            X = reshape(1:3*2*5, 3, 2, 5);
            idx = [5 1; 2 4];                                          % sample 1 takes tokens 5,1; sample 2 takes 2,4
            Y = OceanDL.Vision.MaskedAutoencoder.gather(X, idx);
            testCase.verifyEqual(Y(:, 1, 1), X(:, 1, 5)); testCase.verifyEqual(Y(:, 1, 2), X(:, 1, 1));
            testCase.verifyEqual(Y(:, 2, 1), X(:, 2, 2)); testCase.verifyEqual(Y(:, 2, 2), X(:, 2, 4));
            r = OceanDL.Autograd.GradCheck.check(@(x) OceanDL.Vision.MaskedAutoencoder.gather(x, idx), {randn(3, 2, 5)});
            testCase.verifyTrue(r.Passed);
        end

        function testPatchifyMatchesPatchEmbeddingLayout(testCase)
            s = RandStream('mt19937ar', 'Seed', 1);
            X = randn(s, 8, 12, 2, 3);
            P = OceanDL.Vision.MaskedAutoencoder.patchify(X, [4 4]);
            testCase.verifySize(P, [32 3 6]);
            testCase.verifyEqual(OceanDL.Vision.MaskedAutoencoder.unpatchify(P, [4 4], 2, 3), X);
            pe = OceanDL.Vision.PatchEmbedding(2, 5, 4, 'Seed', 2);
            T = pe.forward(X);
            E = reshape(pe.Proj.Weight.Data, [], 5).';
            testCase.verifyEqual(T.Data, reshape(E * reshape(P, 32, []) + pe.Proj.Bias.Data, 5, 3, 6), 'AbsTol', 1e-12);
        end

        function testEncoderSeesOnlyVisibleTokensAndLossIsOnMaskedOnly(testCase)
            vit = OceanDL.Vision.ViT('ImageSize', [16 16], 'Channels', 1, 'PatchSize', 4, 'Dim', 16, 'Depth', 2, ...
                'Heads', 2, 'Positions', "sincos", 'Pooling', "mean", 'Seed', 3);
            mae = OceanDL.Vision.MaskedAutoencoder(vit, 'MaskRatio', 0.75, 'DecoderDim', 8, 'DecoderDepth', 1, ...
                'DecoderHeads', 2, 'NormPixLoss', false, 'Seed', 4);
            vit.Encoder.setStoreAttention(true);
            X = randn(16, 16, 1, 3);
            L = mae.forward(X);
            testCase.verifySize(vit.Encoder.Layers{1}.Attention.LastAttention, [4 4 2 3]);   % 16 tokens -> 4 visible
            testCase.verifyEqual(sum(mae.LastMask, 2), 12 * ones(3, 1));                      % 75% hidden per image
            % Recompute the loss by hand from the stored prediction: masked patches only.
            target = OceanDL.Vision.MaskedAutoencoder.patchify(X, [4 4]);
            e = squeeze(mean((mae.LastPrediction - target).^2, 1));                            % 3 x 16
            testCase.verifyEqual(L.Data, mean(e(mae.LastMask)), 'AbsTol', 1e-12);
            L.backward();
            testCase.verifyNotEmpty(vit.Patch.Proj.Weight.Grad);                               % gradient reaches the backbone
            testCase.verifyNotEmpty(mae.MaskToken.Grad);
        end

        function testPretrainingReducesReconstructionErrorOnDigits(testCase)
            import OceanDL.Layout.DimFormat
            [rows, ~, fmt] = OceanData.Adapters.UspsDigitsReader.read("test");
            X = DimFormat.toCanonical(DimFormat.fromRowMajor(reshape(rows.', 1, []), fmt.Shape), fmt.Format);
            vit = OceanDL.Vision.ViT('ImageSize', [16 16], 'Channels', 1, 'PatchSize', 4, 'Dim', 32, 'Depth', 2, ...
                'Heads', 2, 'Positions', "sincos", 'Pooling', "mean", 'NumClasses', 10, 'Seed', 5);
            mae = OceanDL.Vision.MaskedAutoencoder(vit, 'DecoderDim', 32, 'DecoderDepth', 1, 'Seed', 6, ...
                'NormPixLoss', false);        % 25% of USPS patches are flat: per-patch normalisation amplifies noise
            opt = OceanDL.optim.ModuleOptimizer(mae.parameters(), @() OceanStats.Optim.AdamWOptimizer('LR', 1e-3, 'WeightDecay', 0));
            rs = RandStream('mt19937ar', 'Seed', 7);
            losses = zeros(1, 300);
            for it = 1:300                                   % measured: 0.72 -> 0.43
                idx = randi(rs, size(X, 4), 1, 64);
                opt.zeroGrad(); L = mae.forward(X(:, :, :, idx)); L.backward(); opt.step();
                losses(it) = L.Data;
            end
            testCase.verifyLessThan(mean(losses(end-19:end)), 0.75 * mean(losses(1:20)));
            testCase.verifySize(vit.forward(X(:, :, :, 1:4)).Data, [10 4]);                    % backbone fine-tunes as a ViT
        end

        function testBackboneMustBeMeanPooledSincos(testCase)
            vit = OceanDL.Vision.ViT('ImageSize', [8 8], 'Channels', 1, 'PatchSize', 4, 'Dim', 8, 'Depth', 1, 'Heads', 2);
            testCase.verifyError(@() OceanDL.Vision.MaskedAutoencoder(vit), 'OceanDL:Vision:MaskedAutoencoder:Backbone');
        end
    end
end
