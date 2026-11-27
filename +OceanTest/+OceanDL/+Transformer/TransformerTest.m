classdef TransformerTest < OceanTest.BaseTestCase
    % TRANSFORMERTEST  OceanDL.Transformer validated four ways:
    %   1. finite differences for the new ops (BatchMatMul, GELU, LayerNorm);
    %   2. PyTorch nn.MultiheadAttention / nn.TransformerEncoderLayer (Pre-LN
    %      and Post-LN): outputs, attention weights and gradients (.npy fixtures
    %      imported through NpyReader + DimFormat);
    %   3. the legacy forward-only OceanNLP BERT attention / Post-LN block with
    %      the same weights (shared core must reproduce it);
    %   4. structural properties: masked keys are ignored, attention is
    %      permutation-equivariant without positions, positional encodings.

    methods (Test)
        function testBatchMatMulAllTransposesWithBroadcastPages(testCase)
            s = RandStream('mt19937ar', 'Seed', 1);
            A = randn(s, 3, 4, 2, 2); B = randn(s, 4, 5, 1, 2);            % B broadcast over page dim 3
            testCase.gc(@(a, b) pagemtimes(a, b), {A, B});
            testCase.gc(@(a, b) pagemtimes(a, 'transpose', b, 'none'), {randn(s, 4, 3, 2), randn(s, 4, 5, 2)});
            testCase.gc(@(a, b) pagemtimes(a, 'none', b, 'transpose'), {randn(s, 3, 4, 2), randn(s, 5, 4, 1)});
            testCase.gc(@(a, b) pagemtimes(a, 'transpose', b, 'transpose'), {randn(s, 4, 3, 2), randn(s, 5, 4, 2)});
        end

        function testGeluAndLayerNorm(testCase)
            s = RandStream('mt19937ar', 'Seed', 2);
            F = OceanDL.Autograd.F;
            testCase.gc(@(x) F.gelu(x), {randn(s, 4, 5)});
            testCase.gc(@(x) F.gelu(x, "tanh"), {randn(s, 4, 5)});
            testCase.gc(@(x, g, b) F.layerNorm(x, g, b), {randn(s, 6, 3, 4), randn(s, 6, 1), randn(s, 6, 1)});
            y = F.layerNorm(randn(s, 6, 3, 4) * 5 + 2, ones(6, 1), zeros(6, 1));
            testCase.verifyEqual(mean(y.Data, 1), zeros(1, 3, 4), 'AbsTol', 1e-12);          % every token normalised
            testCase.verifyEqual(mean(y.Data.^2, 1), ones(1, 3, 4), 'AbsTol', 1e-4);
        end

        function testMultiHeadAttentionMatchesPyTorch(testCase)
            fx = OceanTest.OceanDL.Transformer.TransformerTest.fixtureReader();
            X = fx.seq('x_TND'); R = fx.seq('R_TND');
            mha = OceanDL.Transformer.MultiHeadAttention(8, 2);
            OceanTest.OceanDL.Transformer.TransformerTest.loadAttention(mha, fx, 'mha_in_proj_weight', ...
                'mha_in_proj_bias', 'mha_out_weight', 'mha_out_bias');
            ignore = logical(fx.raw('mha_key_padding_mask'));                    % PyTorch: true = ignore
            mha.StoreAttention = true;
            x = OceanDL.Autograd.Tensor(X, 'RequiresGrad', true);
            Y = mha.forward(x, 'KeyMask', ~ignore);                              % ours: true = attend
            backward(sum(Y .* R, 'all'));
            testCase.verifyEqual(Y.Data, fx.seq('mha_y_TND'), 'AbsTol', 1e-12);
            Wref = permute(fx.raw('mha_w_NHQK'), [3 4 2 1]);                    % (N,H,Tq,Tk) -> Tq x Tk x H x N
            testCase.verifyEqual(mha.LastAttention, Wref, 'AbsTol', 1e-12);
            testCase.verifyEqual(x.Grad, fx.seq('mha_dx_TND'), 'AbsTol', 1e-12);
            dWin = fx.raw('mha_dWin');
            testCase.verifyEqual(mha.Query.Weight.Grad, dWin(1:8, :), 'AbsTol', 1e-12);
            testCase.verifyEqual(mha.Value.Weight.Grad, dWin(17:24, :), 'AbsTol', 1e-12);
        end

        function testTransformerBlockMatchesPyTorchPreAndPostLN(testCase)
            fx = OceanTest.OceanDL.Transformer.TransformerTest.fixtureReader();
            X = fx.seq('x_TND'); R = fx.seq('R_TND');
            for tag = ["pre", "post"]
                p = @(n) char("enc_" + tag + "_" + n);
                blk = OceanDL.Transformer.TransformerBlock(8, 2, 16, 'Norm', tag, 'Activation', "gelu");
                OceanTest.OceanDL.Transformer.TransformerTest.loadAttention(blk.Attention, fx, ...
                    p('self_attn_in_proj_weight'), p('self_attn_in_proj_bias'), p('self_attn_out_proj_weight'), p('self_attn_out_proj_bias'));
                blk.MLP.Fc1.Weight.Data = fx.raw(p('linear1_weight')); blk.MLP.Fc1.Bias.Data = fx.col(p('linear1_bias'));
                blk.MLP.Fc2.Weight.Data = fx.raw(p('linear2_weight')); blk.MLP.Fc2.Bias.Data = fx.col(p('linear2_bias'));
                blk.Norm1.Weight.Data = fx.col(p('norm1_weight')); blk.Norm1.Bias.Data = fx.col(p('norm1_bias'));
                blk.Norm2.Weight.Data = fx.col(p('norm2_weight')); blk.Norm2.Bias.Data = fx.col(p('norm2_bias'));
                x = OceanDL.Autograd.Tensor(X, 'RequiresGrad', true);
                Y = blk.forward(x);
                backward(sum(Y .* R, 'all'));
                testCase.verifyEqual(Y.Data, fx.seq(p('y_TND')), 'AbsTol', 1e-12, char(tag));
                testCase.verifyEqual(x.Grad, fx.seq(p('dx_TND')), 'AbsTol', 1e-12, char(tag));
                testCase.verifyEqual(blk.MLP.Fc1.Weight.Grad, fx.raw(p('dW1')), 'AbsTol', 1e-12, char(tag));
                testCase.verifyEqual(blk.Norm1.Weight.Grad, fx.col(p('dnorm1')), 'AbsTol', 1e-12, char(tag));
            end
        end

        function testSharedCoreReproducesLegacyBertAttention(testCase)
            s = RandStream('mt19937ar', 'Seed', 3);
            d = 8; T = 6;
            W = arrayfun(@(k) randn(s, d, d) / sqrt(d), 1:4, 'UniformOutput', false);
            b = arrayfun(@(k) 0.1 * randn(s, d, 1), 1:4, 'UniformOutput', false);
            legacy = OceanNLP.Embedding.Modules.MultiHeadAttention(W{1}, b{1}, W{2}, b{2}, W{3}, b{3}, W{4}, b{4}, 2);
            mha = OceanDL.Transformer.MultiHeadAttention(d, 2);
            lins = {mha.Query, mha.Key, mha.Value, mha.Output};
            for k = 1:4, lins{k}.Weight.Data = W{k}; lins{k}.Bias.Data = b{k}; end
            X = randn(s, d, T);
            keyMask = logical([1 1 1 1 0 0]);
            Yold = legacy.forward(X, keyMask);
            Ynew = mha.forward(reshape(X, d, 1, T), 'KeyMask', keyMask);
            testCase.verifyEqual(reshape(Ynew.Data, d, T), Yold, 'AbsTol', 1e-12);
        end

        function testSharedCoreReproducesLegacyBertPostLnBlock(testCase)
            s = RandStream('mt19937ar', 'Seed', 4);
            d = 8; T = 5; ff = 16; epsLN = 1e-12;                              % BERT's eps: std+eps vs sqrt(var+eps) agree
            W = arrayfun(@(k) randn(s, d, d) / sqrt(d), 1:4, 'UniformOutput', false);
            b = arrayfun(@(k) 0.1 * randn(s, d, 1), 1:4, 'UniformOutput', false);
            W1 = randn(s, ff, d) / sqrt(d); b1 = 0.1 * randn(s, ff, 1);
            W2 = randn(s, d, ff) / sqrt(ff); b2 = 0.1 * randn(s, d, 1);
            g1 = 1 + 0.1 * randn(s, d, 1); be1 = 0.1 * randn(s, d, 1);
            g2 = 1 + 0.1 * randn(s, d, 1); be2 = 0.1 * randn(s, d, 1);
            att = OceanNLP.Embedding.Modules.MultiHeadAttention(W{1}, b{1}, W{2}, b{2}, W{3}, b{3}, W{4}, b{4}, 2);
            ln1 = OceanDL.Norm.LayerNormLayer(d, 'Epsilon', epsLN); ln1.Gamma = g1; ln1.Beta = be1;
            ln2 = OceanDL.Norm.LayerNormLayer(d, 'Epsilon', epsLN); ln2.Gamma = g2; ln2.Beta = be2;
            legacy = OceanNLP.Embedding.Modules.TransformerEncoderBlock(att, ln1, ln2, W1, b1, W2, b2);
            blk = OceanDL.Transformer.TransformerBlock(d, 2, ff, 'Norm', "post", 'Activation', "gelu", 'Epsilon', epsLN);
            lins = {blk.Attention.Query, blk.Attention.Key, blk.Attention.Value, blk.Attention.Output};
            for k = 1:4, lins{k}.Weight.Data = W{k}; lins{k}.Bias.Data = b{k}; end
            blk.MLP.Fc1.Weight.Data = W1; blk.MLP.Fc1.Bias.Data = b1;
            blk.MLP.Fc2.Weight.Data = W2; blk.MLP.Fc2.Bias.Data = b2;
            blk.Norm1.Weight.Data = g1; blk.Norm1.Bias.Data = be1;
            blk.Norm2.Weight.Data = g2; blk.Norm2.Bias.Data = be2;
            X = randn(s, d, T);
            testCase.verifyEqual(reshape(blk.forward(reshape(X, d, 1, T)).Data, d, T), legacy.forward(X), 'AbsTol', 1e-9);
        end

        function testMaskedKeysAreIgnored(testCase)
            s = RandStream('mt19937ar', 'Seed', 5);
            mha = OceanDL.Transformer.MultiHeadAttention(8, 4, 'Seed', 1);
            mha.StoreAttention = true;
            X = randn(s, 8, 2, 6);
            mask = logical([1 1 1 1 1 1; 1 1 1 0 0 0]);
            Y1 = mha.forward(X, 'KeyMask', mask);
            A = mha.LastAttention;
            testCase.verifyEqual(A(:, 4:6, :, 2), zeros(6, 3, 4));                % no weight on masked keys
            testCase.verifyEqual(sum(A, 2), ones(6, 1, 4, 2), 'AbsTol', 1e-12);
            X(:, 2, 4:6) = 100 * randn(s, 8, 1, 3);                               % change the masked tokens
            Y2 = mha.forward(X, 'KeyMask', mask);
            testCase.verifyEqual(Y2.Data(:, 2, 1:3), Y1.Data(:, 2, 1:3), 'AbsTol', 1e-12);   % unmasked queries unaffected
        end

        function testAttentionIsPermutationEquivariantWithoutPositions(testCase)
            % Why positional encodings are needed: shuffling tokens only shuffles outputs.
            enc = OceanDL.Transformer.TransformerEncoder(2, 8, 2, 16, 'Seed', 3);
            X = randn(8, 2, 5);
            perm = [3 1 5 2 4];
            Y = enc.forward(X);
            Yp = enc.forward(X(:, :, perm));
            testCase.verifyEqual(Yp.Data, Y.Data(:, :, perm), 'AbsTol', 1e-12);
        end

        function testPositionalEncodings(testCase)
            P = OceanDL.Transformer.PositionalEncoding.sincos1d(4, 6);
            testCase.verifySize(P, [6 1 4]);
            testCase.verifyEqual(P(:, 1, 1), repmat([0; 1], 3, 1));                 % t = 0: sin 0, cos 0
            testCase.verifyEqual(P(1, 1, 3), sin(2), 'AbsTol', 1e-15);              % i = 0 -> wavelength 2*pi
            P2 = OceanDL.Transformer.PositionalEncoding.sincos2d(3, 4, 8);
            testCase.verifySize(P2, [8 1 12]);
            t = 2 + 3 * (3 - 1);                                                     % row 2, column 3 (column-major)
            rowPE = OceanDL.Transformer.PositionalEncoding.sincos1d(3, 4);
            colPE = OceanDL.Transformer.PositionalEncoding.sincos1d(4, 4);
            testCase.verifyEqual(P2(1:4, 1, t), rowPE(:, 1, 2));
            testCase.verifyEqual(P2(5:8, 1, t), colPE(:, 1, 3));
            pe = OceanDL.Transformer.LearnedPositionalEmbedding(5, 4, 'Seed', 1);
            y = pe.forward(ones(4, 3, 5));
            backward(sum(y, 'all'));
            testCase.verifyEqual(pe.Weight.Grad, 3 * ones(4, 1, 5));                % summed over the batch of 3
        end

        function testEncoderStackAndEvalDeterminism(testCase)
            enc = OceanDL.Transformer.TransformerEncoder(3, 8, 2, 16, 'Dropout', 0.3, 'DropPath', 0.2, 'Seed', 1);
            testCase.verifyNotEmpty(enc.FinalNorm);                                  % pre-LN stacks end with a norm
            testCase.verifyEqual(enc.Layers{1}.DropPath, 0);
            testCase.verifyEqual(enc.Layers{3}.DropPath, 0.2, 'AbsTol', 1e-15);      % linear stochastic-depth schedule
            X = randn(8, 2, 4);
            enc.eval();
            testCase.verifyEqual(enc.forward(X).Data, enc.forward(X).Data);
            post = OceanDL.Transformer.TransformerEncoder(2, 8, 2, 16, 'Norm', "post");
            testCase.verifyEmpty(post.FinalNorm);
        end
    end

    methods (Access = private)
        function gc(testCase, fn, inputs)
            r = OceanDL.Autograd.GradCheck.check(fn, inputs);
            testCase.verifyTrue(r.Passed, sprintf('%s: max abs err %.3g', func2str(fn), r.MaxAbsError));
        end
    end

    methods (Static)
        function fx = fixtureReader()
            dir_ = fullfile(fileparts(mfilename('fullpath')), 'fixtures');
            raw = @(n) double(OceanData.Adapters.NpyReader.read(fullfile(dir_, [n '.npy'])));
            fx = struct('raw', raw, ...
                'col', @(n) reshape(raw(n), [], 1), ...
                'seq', @(n) OceanDL.Layout.DimFormat.toCanonical(raw(n), "TBC"));
        end

        function loadAttention(mha, fx, inW, inB, outW, outB)
            % PyTorch packs [Wq; Wk; Wv] into in_proj_weight (3D x D).
            Win = fx.raw(inW); bin = fx.col(inB); D = size(Win, 2);
            lins = {mha.Query, mha.Key, mha.Value};
            for k = 1:3
                lins{k}.Weight.Data = Win((k-1)*D + (1:D), :);
                lins{k}.Bias.Data = bin((k-1)*D + (1:D));
            end
            mha.Output.Weight.Data = fx.raw(outW);
            mha.Output.Bias.Data = fx.col(outB);
        end
    end
end
