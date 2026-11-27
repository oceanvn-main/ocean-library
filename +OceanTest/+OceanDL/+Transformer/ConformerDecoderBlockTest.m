classdef ConformerDecoderBlockTest < OceanTest.BaseTestCase
    % CONFORMERDECODERBLOCKTEST  OceanDL.Transformer.DecoderBlock and ConformerBlock:
    %   1. PyTorch references with copied weights (fixtures/gen_torch_decoder_conformer_reference.py):
    %      DecoderBlock vs torch.nn.TransformerDecoderLayer(norm_first=True) -- outputs, input,
    %      memory and parameter gradients; ConformerBlock vs an independent torch implementation;
    %   2. finite-difference GradCheck (inputs and every parameter) on tiny sizes;
    %   3. structure: causal mask (future targets never change earlier outputs), masked memory
    %      frames ignored, padded batch == per-utterance, empty-memory guard.

    methods (Test)
        function testDecoderBlockMatchesPyTorch(testCase)
            fx = OceanTest.OceanDL.Transformer.ConformerDecoderBlockTest.fixtureReader();
            blk = OceanDL.Transformer.DecoderBlock(8, 2, 16, 'Activation', "relu");
            p = @(n) ['dec_' n];
            loadMha(blk.SelfAttention, fx, p('self_attn_in_proj_weight'), p('self_attn_in_proj_bias'), ...
                p('self_attn_out_proj_weight'), p('self_attn_out_proj_bias'));
            loadMha(blk.CrossAttention, fx, p('multihead_attn_in_proj_weight'), p('multihead_attn_in_proj_bias'), ...
                p('multihead_attn_out_proj_weight'), p('multihead_attn_out_proj_bias'));
            blk.MLP.Fc1.Weight.Data = fx.raw(p('linear1_weight')); blk.MLP.Fc1.Bias.Data = fx.col(p('linear1_bias'));
            blk.MLP.Fc2.Weight.Data = fx.raw(p('linear2_weight')); blk.MLP.Fc2.Bias.Data = fx.col(p('linear2_bias'));
            norms = {blk.Norm1, blk.Norm2, blk.Norm3};
            for k = 1:3
                norms{k}.Weight.Data = fx.col(p(sprintf('norm%d_weight', k))); norms{k}.Bias.Data = fx.col(p(sprintf('norm%d_bias', k)));
            end
            x = OceanDL.Autograd.Tensor(fx.seq('dec_x_TND'), 'RequiresGrad', true);
            mem = OceanDL.Autograd.Tensor(fx.seq('dec_mem_TND'), 'RequiresGrad', true);
            valid = ~logical(fx.raw('dec_mem_pad_mask'));                      % PyTorch: true = ignore
            y = blk.forward(x, mem, 'MemoryKeyMask', valid);
            backward(sum(y .* fx.seq('dec_R_TND'), 'all'));
            testCase.verifyEqual(y.Data, fx.seq('dec_y_TND'), 'AbsTol', 1e-12);
            testCase.verifyEqual(x.Grad, fx.seq('dec_dx_TND'), 'AbsTol', 1e-12);
            testCase.verifyEqual(mem.Grad, fx.seq('dec_dmem_TND'), 'AbsTol', 1e-12);
            dW = fx.raw('dec_grad_multihead_attn_in_proj_weight');
            testCase.verifyEqual(blk.CrossAttention.Key.Weight.Grad, dW(9:16, :), 'AbsTol', 1e-12);
            testCase.verifyEqual(blk.MLP.Fc1.Weight.Grad, fx.raw('dec_grad_linear1_weight'), 'AbsTol', 1e-12);
            testCase.verifyEqual(blk.Norm2.Weight.Grad, fx.col('dec_grad_norm2_weight'), 'AbsTol', 1e-12);
        end

        function testConformerBlockMatchesTorchReference(testCase)
            fx = OceanTest.OceanDL.Transformer.ConformerDecoderBlockTest.fixtureReader();
            blk = OceanDL.Transformer.ConformerBlock(8, 2, 'ConvKernel', 3, 'FFMult', 4);
            map = {'FF1In', 'ff1_in'; 'FF1Out', 'ff1_out'; 'FF2In', 'ff2_in'; 'FF2Out', 'ff2_out'};
            for i = 1:size(map, 1)
                blk.(map{i, 1}).Weight.Data = fx.raw(['conf_' map{i, 2} '_weight']);
                blk.(map{i, 1}).Bias.Data = fx.col(['conf_' map{i, 2} '_bias']);
            end
            att = {blk.Attention.Query, 'q'; blk.Attention.Key, 'k'; blk.Attention.Value, 'v'; blk.Attention.Output, 'o'};
            for i = 1:4
                att{i, 1}.Weight.Data = fx.raw(['conf_' att{i, 2} '_weight']); att{i, 1}.Bias.Data = fx.col(['conf_' att{i, 2} '_bias']);
            end
            conv = {blk.Pointwise1, 'pw1'; blk.Depthwise, 'dw'; blk.Pointwise2, 'pw2'};
            for i = 1:3
                conv{i, 1}.Weight.Data = fx.raw(['conf_' conv{i, 2} '_weight']); conv{i, 1}.Bias.Data = fx.col(['conf_' conv{i, 2} '_bias']);
            end
            ln = {blk.NormFF1, 'ln_ff1'; blk.NormAttn, 'ln_att'; blk.NormConv, 'ln_conv'; blk.ConvNorm, 'ln_cnorm'; ...
                  blk.NormFF2, 'ln_ff2'; blk.NormOut, 'ln_out'};
            for i = 1:size(ln, 1)
                ln{i, 1}.Weight.Data = fx.col(['conf_' ln{i, 2} '_weight']); ln{i, 1}.Bias.Data = fx.col(['conf_' ln{i, 2} '_bias']);
            end
            valid = logical(fx.raw('conf_valid'));
            x = OceanDL.Autograd.Tensor(fx.seq('conf_x_TND'), 'RequiresGrad', true);
            y = blk.forward(x, 'KeyMask', valid);
            backward(sum(y .* fx.seq('conf_R_TND'), 'all'));
            testCase.verifyEqual(y.Data, fx.seq('conf_y_TND'), 'AbsTol', 1e-12);
            testCase.verifyEqual(x.Grad, fx.seq('conf_dx_TND'), 'AbsTol', 1e-12);
            testCase.verifyEqual(blk.Depthwise.Weight.Grad, fx.raw('conf_grad_dw_weight'), 'AbsTol', 1e-12);
            testCase.verifyEqual(blk.Pointwise1.Weight.Grad, fx.raw('conf_grad_pw1_weight'), 'AbsTol', 1e-12);
            testCase.verifyEqual(blk.Attention.Query.Weight.Grad, fx.raw('conf_grad_q_weight'), 'AbsTol', 1e-12);
            testCase.verifyEqual(blk.ConvNorm.Weight.Grad, fx.col('conf_grad_ln_cnorm_weight'), 'AbsTol', 1e-12);
        end

        function testGradCheckConformerBlock(testCase)
            s = RandStream('mt19937ar', 'Seed', 4);
            for pos = ["rope", "none"]
                blk = OceanDL.Transformer.ConformerBlock(4, 2, 'ConvKernel', 3, 'FFMult', 2, 'PosEnc', pos, 'Seed', 1);
                blk.eval();
                M = [true(1, 5); true(1, 3) false(1, 2)];
                r = OceanDL.Autograd.GradCheck.check(@(x) blk.forward(x, 'KeyMask', M), {randn(s, 4, 2, 5)});
                testCase.verifyTrue(r.Passed, sprintf('input grad (%s): rel err %.3g', pos, r.MaxRelError));
                X = randn(s, 4, 2, 5);
                err = OceanTest.OceanNLP.Speech.Models.EncodersTest.paramGradError(blk, @(b) b.forward(X, 'KeyMask', M), 2);
                testCase.verifyLessThan(err, 1e-6, char(pos));
            end
        end

        function testGradCheckDecoderBlock(testCase)
            s = RandStream('mt19937ar', 'Seed', 5);
            blk = OceanDL.Transformer.DecoderBlock(4, 2, 6, 'Activation', "gelu", 'Seed', 3);
            mm = [true(1, 4); true(1, 2) false(1, 2)];
            r = OceanDL.Autograd.GradCheck.check(@(x, m) blk.forward(x, m, 'MemoryKeyMask', mm), {randn(s, 4, 2, 3), randn(s, 4, 2, 4)});
            testCase.verifyTrue(r.Passed, sprintf('input/memory grad rel err %.3g', r.MaxRelError));
            X = randn(s, 4, 2, 3); Mem = randn(s, 4, 2, 4);
            err = OceanTest.OceanNLP.Speech.Models.EncodersTest.paramGradError(blk, @(b) b.forward(X, Mem, 'MemoryKeyMask', mm), 3);
            testCase.verifyLessThan(err, 1e-6);
        end

        function testDecoderCausalAndMemoryMask(testCase)
            s = RandStream('mt19937ar', 'Seed', 6);
            blk = OceanDL.Transformer.DecoderBlock(6, 3, 12, 'Seed', 4);
            X = randn(s, 6, 2, 5); Mem = randn(s, 6, 2, 7);
            mm = [true(1, 7); true(1, 4) false(1, 3)];
            y = blk.forward(X, Mem, 'MemoryKeyMask', mm).Data;
            for u = 1:4                              % change positions u+1..U: outputs 1..u unchanged
                X2 = X; X2(:, :, u+1:end) = randn(s, 6, 2, 5 - u);
                y2 = blk.forward(X2, Mem, 'MemoryKeyMask', mm).Data;
                testCase.verifyEqual(y2(:, :, 1:u), y(:, :, 1:u), 'AbsTol', 1e-13);
                testCase.verifyGreaterThan(max(abs(y2(:, :, u+1:end) - y(:, :, u+1:end)), [], 'all'), 1e-3);
            end
            Mem2 = Mem; Mem2(:, 2, 5:end) = 1e3 * randn(s, 6, 1, 3);  % masked frames of utterance 2
            y3 = blk.forward(X, Mem2, 'MemoryKeyMask', mm).Data;
            testCase.verifyEqual(y3, y, 'AbsTol', 1e-12);
            % per-utterance (no padding) == padded batch
            y4 = blk.forward(X(:, 2, :), Mem(:, 2, 1:4)).Data;
            testCase.verifyEqual(y4, y(:, 2, :), 'AbsTol', 1e-12);
            % stored attention: rows sum to 1, masked frames get zero weight, causal self-attention
            blk.StoreAttention = true;
            blk.forward(X, Mem, 'MemoryKeyMask', mm);
            A = blk.LastCrossAttention;
            testCase.verifySize(A, [5 7 3 2]);
            testCase.verifyEqual(sum(A, 2), ones(5, 1, 3, 2), 'AbsTol', 1e-12);
            testCase.verifyEqual(nnz(A(:, 5:end, :, 2)), 0);
            S = blk.LastSelfAttention;
            testCase.verifyEqual(nnz(S .* ~tril(true(5))), 0);
            % empty memory row: finite (guard), no NaN
            y5 = blk.forward(X, Mem, 'MemoryKeyMask', [true(1, 7); false(1, 7)]).Data;
            testCase.verifyTrue(all(isfinite(y5), 'all'));
        end

        function testDecoderIncrementalStepMatchesForward(testCase)
            % forwardStep with the key/value cache == column u of the full causal forward,
            % including a masked memory, an all-masked (guarded) memory row and cache
            % reordering (rows duplicated / permuted between steps, as in beam search).
            s = RandStream('mt19937ar', 'Seed', 9);
            blk = OceanDL.Transformer.DecoderBlock(8, 2, 16, 'Seed', 2);
            blk.eval();
            U = 6; N = 3; T = 7;
            X = randn(s, 8, N, U); Mem = randn(s, 8, N, T);
            mm = [true(1, T); true(1, 4) false(1, 3); false(1, T)];
            y = blk.forward(X, Mem, 'MemoryKeyMask', mm).Data;
            mkv = blk.memoryKeyValues(Mem);
            kv = OceanDL.Transformer.DecoderBlock.emptyCache(4, 2, N);
            for u = 1:U
                [yu, kv] = blk.forwardStep(X(:, :, u), kv, mkv, 'MemoryKeyMask', mm);
                testCase.verifyEqual(yu, y(:, :, u), 'AbsTol', 1e-12);
                testCase.verifySize(kv.K, [4 u 2 N]);
            end
            % reorder: rows (3, 1, 1) continue from the cache of rows 3, 1, 1 after 3 steps
            perm = [3 1 1];
            kv = OceanDL.Transformer.DecoderBlock.emptyCache(4, 2, N);
            for u = 1:3
                [~, kv] = blk.forwardStep(X(:, :, u), kv, mkv, 'MemoryKeyMask', mm);
            end
            kv = struct('K', kv.K(:, :, :, perm), 'V', kv.V(:, :, :, perm));
            mk = struct('K', mkv.K(:, :, :, perm), 'V', mkv.V(:, :, :, perm));
            X2 = X(:, perm, :); X2(:, 3, 4) = randn(s, 8, 1);           % the duplicate diverges at u = 4
            y2 = blk.forward(X2, Mem(:, perm, :), 'MemoryKeyMask', mm(perm, :)).Data;
            for u = 4:U
                [yu, kv] = blk.forwardStep(X2(:, :, u), kv, mk, 'MemoryKeyMask', mm(perm, :));
                testCase.verifyEqual(yu, y2(:, :, u), 'AbsTol', 1e-12);
            end
            blk.train();
            testCase.verifyError(@() blk.forwardStep(X(:, :, 1), OceanDL.Transformer.DecoderBlock.emptyCache(4, 2, N), mkv), ...
                'OceanDL:Transformer:DecoderBlock:StepTraining');
        end

        function testConformerPaddingInvariance(testCase)
            s = RandStream('mt19937ar', 'Seed', 7);
            blk = OceanDL.Transformer.ConformerBlock(8, 2, 'ConvKernel', 5, 'Dropout', 0.3, 'Seed', 2);
            blk.eval();
            lens = [9 6 3];
            X = zeros(8, 3, 9);
            for n = 1:3, X(:, n, 1:lens(n)) = randn(s, 8, 1, lens(n)); end
            M = OceanDL.Layout.SequenceMask.fromLengths(lens, 9);
            Y = blk.forward(X, 'KeyMask', M).Data;
            Xj = X; Xj(:, 2, 7:end) = 50 * randn(s, 8, 1, 3);          % garbage in padding
            Yj = blk.forward(Xj, 'KeyMask', M).Data;
            testCase.verifyEqual(Yj, Y, 'AbsTol', 1e-12);
            for n = 1:3
                y1 = blk.forward(X(:, n, 1:lens(n))).Data;
                testCase.verifyEqual(Y(:, n, 1:lens(n)), y1, 'AbsTol', 1e-12);
                testCase.verifyEqual(nnz(Y(:, n, lens(n)+1:end)), 0);
            end
            testCase.verifyError(@() OceanDL.Transformer.ConformerBlock(8, 2, 'ConvKernel', 4), ...
                'OceanDL:Transformer:ConformerBlock:EvenKernel');
        end
    end

    methods (Static)
        function fx = fixtureReader()
            dir_ = fullfile(fileparts(mfilename('fullpath')), 'fixtures');
            raw = @(n) double(OceanData.Adapters.NpyReader.read(fullfile(dir_, [n '.npy'])));
            fx = struct('raw', raw, 'col', @(n) reshape(raw(n), [], 1), ...
                'seq', @(n) OceanDL.Layout.DimFormat.toCanonical(raw(n), "TBC"));
        end
    end
end

function loadMha(mha, fx, inW, inB, outW, outB)
W = fx.raw(inW); b = fx.col(inB); D = mha.EmbedDim;
lins = {mha.Query, mha.Key, mha.Value};
for k = 1:3
    lins{k}.Weight.Data = W((k-1)*D + (1:D), :);
    lins{k}.Bias.Data = b((k-1)*D + (1:D));
end
mha.Output.Weight.Data = fx.raw(outW);
mha.Output.Bias.Data = fx.col(outB);
end
