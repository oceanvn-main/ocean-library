classdef BuildingBlocksTest < OceanTest.BaseTestCase
    % BUILDINGBLOCKSTEST  Conv-autoencoder blocks against PyTorch (1e-10):
    %   GroupNorm (values + all gradients), nearest upsampling, a GN/SiLU
    %   residual block with a 1x1 skip followed by a stride-2 conv; gradient
    %   checks; encoder / decoder shapes, factors and self-attention.

    methods (Test)
        function testGroupNormMatchesTorch(testCase)
            fx = OceanTest.OceanDL.Generative.BuildingBlocksTest.fixtures();
            gn = OceanDL.nn.GroupNorm(3, 6);
            gn.Weight.Data = reshape(fx.raw('gn_weight'), 1, 1, []);
            gn.Bias.Data = reshape(fx.raw('gn_bias'), 1, 1, []);
            X = OceanDL.Autograd.Tensor(fx.img('gn_x'), 'RequiresGrad', true);
            Y = gn.forward(X);
            testCase.verifyEqual(Y.Data, fx.img('gn_y'), 'AbsTol', 1e-10);
            backward(sum(Y .* fx.img('gn_R'), 'all'));
            testCase.verifyEqual(X.Grad, fx.img('gn_dx'), 'AbsTol', 1e-10);
            testCase.verifyEqual(reshape(gn.Weight.Grad, [], 1), fx.col('gn_dweight'), 'AbsTol', 1e-10);
            testCase.verifyEqual(reshape(gn.Bias.Grad, [], 1), fx.col('gn_dbias'), 'AbsTol', 1e-10);
            r = OceanDL.Autograd.GradCheck.check(@(x) OceanDL.Autograd.F.groupNorm(x, 2, [], []), {randn(3, 2, 4, 2)});
            testCase.verifyTrue(r.Passed);
            testCase.verifyError(@() OceanDL.nn.GroupNorm(4, 6), 'OceanDL:nn:GroupNorm:Groups');
        end

        function testUpsampleMatchesTorch(testCase)
            fx = OceanTest.OceanDL.Generative.BuildingBlocksTest.fixtures();
            X = OceanDL.Autograd.Tensor(fx.img('up_x'), 'RequiresGrad', true);
            Y = OceanDL.nn.Upsample(2).forward(X);
            testCase.verifyEqual(Y.Data, fx.img('up_y'), 'AbsTol', 1e-12);
            backward(sum(Y .* fx.img('up_R'), 'all'));
            testCase.verifyEqual(X.Grad, fx.img('up_dx'), 'AbsTol', 1e-12);
            r = OceanDL.Autograd.GradCheck.check(@(x) OceanDL.Autograd.F.upsample(x, [2 3]), {randn(2, 3, 2, 2)});
            testCase.verifyTrue(r.Passed);
        end

        function testResBlockAndStridedConvMatchTorch(testCase)
            fx = OceanTest.OceanDL.Generative.BuildingBlocksTest.fixtures();
            blk = OceanDL.Vision.ConvResBlock(4, 6, 'Groups', 2);
            down = OceanDL.nn.Conv2d(6, 6, 3, 'Stride', 2, 'Padding', 1);
            conv = @(m, p) OceanTest.OceanDL.Generative.BuildingBlocksTest.loadConv(m, fx, p);
            norm = @(m, p) OceanTest.OceanDL.Generative.BuildingBlocksTest.loadNorm(m, fx, p);
            norm(blk.Norm1, 'rb_norm1'); norm(blk.Norm2, 'rb_norm2');
            conv(blk.Conv1, 'rb_conv1'); conv(blk.Conv2, 'rb_conv2'); conv(blk.Skip, 'rb_skip'); conv(down, 'rb_down');
            X = OceanDL.Autograd.Tensor(fx.img('rb_x'), 'RequiresGrad', true);
            Y = down.forward(blk.forward(X));
            testCase.verifyEqual(Y.Data, fx.img('rb_y'), 'AbsTol', 1e-10);
            backward(sum(Y .* fx.img('rb_R'), 'all'));
            testCase.verifyEqual(X.Grad, fx.img('rb_dx'), 'AbsTol', 1e-10);
            testCase.verifyEqual(blk.Conv1.Weight.Grad, permute(fx.raw('rb_dconv1_weight'), [3 4 2 1]), 'AbsTol', 1e-10);
        end

        function testEncoderDecoderShapesAndAttention(testCase)
            enc = OceanDL.Vision.ConvEncoder('InChannels', 1, 'Widths', [8 16 16], 'OutChannels', 6, 'Groups', 4, 'Seed', 1);
            dec = OceanDL.Vision.ConvDecoder('LatentChannels', 3, 'Widths', [8 16 16], 'OutChannels', 1, 'Groups', 4, 'Seed', 2);
            testCase.verifyEqual(enc.Factor, 4);
            z = enc.forward(randn(16, 12, 1, 2));
            testCase.verifySize(z.Data, [4 3 6 2]);
            x = dec.forward(randn(4, 3, 3, 2));
            testCase.verifySize(x.Data, [16 12 1 2]);
            testCase.verifyError(@() enc.forward(randn(10, 12, 1, 1)), 'OceanDL:Vision:ConvEncoder:Size');
            % Seeds: equal seeds -> equal models, and layers do not share weights.
            enc2 = OceanDL.Vision.ConvEncoder('InChannels', 1, 'Widths', [8 16 16], 'OutChannels', 6, 'Groups', 4, 'Seed', 1);
            testCase.verifyEqual(enc2.forward(ones(8, 8, 1, 1)).Data, enc.forward(ones(8, 8, 1, 1)).Data);
            res = enc.Body.Layers{2}; mid = enc.Body.Layers{end - 3};
            testCase.verifyNotEqual(res.Conv2.Weight.Data(1:4), mid.Conv2.Weight.Data(1:4));
            % Attention sees every position: perturbing one pixel of the input
            % map changes the attention output everywhere.
            att = OceanDL.Vision.SpatialSelfAttention(4, 'Groups', 2, 'Seed', 3);
            X = randn(3, 3, 4, 1); X2 = X; X2(1, 1, :) = X2(1, 1, :) + 1;
            d = abs(att.forward(X2).Data - att.forward(X).Data);
            testCase.verifyTrue(all(squeeze(sum(d, 3)) > 1e-8, 'all'));
            r = OceanDL.Autograd.GradCheck.check(@(x) att.forward(x), {randn(2, 2, 4, 2)});
            testCase.verifyTrue(r.Passed);
        end
    end

    methods (Static)
        function fx = fixtures()
            dir_ = fullfile(fileparts(mfilename('fullpath')), 'fixtures');
            raw = @(n) double(OceanData.Adapters.NpyReader.read(fullfile(dir_, [n '.npy'])));
            fx = struct('raw', raw, 'col', @(n) reshape(raw(n), [], 1), ...
                'img', @(n) OceanDL.Layout.DimFormat.toCanonical(raw(n), "BCSS"));
        end

        function loadConv(m, fx, prefix)
            m.Weight.Data = permute(fx.raw([prefix '_weight']), [3 4 2 1]);   % torch (Cout,Cin,KH,KW)
            m.Bias.Data = fx.col([prefix '_bias']);
        end

        function loadNorm(m, fx, prefix)
            m.Weight.Data = reshape(fx.raw([prefix '_weight']), 1, 1, []);
            m.Bias.Data = reshape(fx.raw([prefix '_bias']), 1, 1, []);
        end
    end
end
