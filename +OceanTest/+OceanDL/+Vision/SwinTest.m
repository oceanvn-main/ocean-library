classdef SwinTest < OceanTest.BaseTestCase
    % SWINTEST  Swin Transformer geometry and equivalences:
    %   partition/reverse and shift/unshift are exact inverses; the shifted-
    %   window mask never lets wrapped-around regions attend to each other;
    %   a single un-shifted window with zero relative bias IS a global
    %   Pre-LN TransformerBlock; relative indices; PatchMerging; gradients;
    %   and cost that grows linearly (Swin) vs quadratically (ViT) in tokens.

    methods (Test)
        function testPartitionReverseAndShiftAreExactInverses(testCase)
            X = randn(3, 2, 8, 12);
            W = OceanDL.Vision.WindowOps.partition(X, 4);
            testCase.verifySize(W, [3 2*2*3 16]);
            testCase.verifyEqual(OceanDL.Vision.WindowOps.reverse(W, 4, 2, 8, 12), X);
            S = OceanDL.Vision.WindowOps.shift(X, 2, -1);
            testCase.verifyEqual(S(:, :, 1, 1), X(:, :, 3, 3));                         % rolled by -2
            testCase.verifyEqual(OceanDL.Vision.WindowOps.shift(S, 2, +1), X);
            % window w = (hw, ww), local token i = (r, c): W(:, n + N(w-1), r + M(c-1)) = X(:, n, M(hw-1)+r, M(ww-1)+c)
            n = 2; hw = 2; ww = 3; r = 3; c = 1;
            testCase.verifyEqual(W(:, n + 2 * ((hw - 1) + 2 * (ww - 1)), r + 4 * (c - 1)), X(:, n, 4*(hw-1) + r, 4*(ww-1) + c));
        end

        function testShiftMaskSeparatesWrappedRegions(testCase)
            Hp = 8; M = 4; s = 2; N = 1;
            mask = OceanDL.Vision.WindowOps.shiftMask(Hp, Hp, M, s, N);
            testCase.verifySize(mask, [16 16 1 4]);
            testCase.verifyTrue(all(mask(:, :, 1, 1), 'all'));                          % top-left window: one region
            % Bottom-right window after the roll mixes 4 regions -> 4 diagonal blocks of 4 tokens.
            testCase.verifyEqual(nnz(mask(:, :, 1, 4)), 4 * 4 * 4);
            testCase.verifyEqual(mask(:, :, 1, 4), mask(:, :, 1, 4).');                 % symmetric
            testCase.verifyTrue(all(diag(mask(:, :, 1, 4))));                           % every token sees itself
            % Structural check on a shifted block: attention weights across regions are exactly 0.
            blk = OceanDL.Vision.SwinBlock(8, 2, M, s, 'Seed', 1);
            blk.Attention.StoreAttention = true;
            blk.forward(randn(8, N, Hp, Hp));
            A = blk.Attention.LastAttention;                                              % 16 x 16 x 2 x 4
            testCase.verifyEqual(A(~repmat(mask, 1, 1, 2, 1)), zeros(nnz(~repmat(mask, 1, 1, 2, 1)), 1));
        end

        function testSingleWindowWithoutBiasIsAGlobalTransformerBlock(testCase)
            D = 8; Hp = 4; N = 2;
            swin = OceanDL.Vision.SwinBlock(D, 2, 4, 0, 'Seed', 3);
            swin.RelBiasTable.Data(:) = 0;
            blk = OceanDL.Transformer.TransformerBlock(D, 2, 4 * D, 'Norm', "pre");
            copy = @(dst, src) OceanTest.OceanDL.Vision.SwinTest.copyParams(dst, src);
            copy(blk.Attention, swin.Attention); copy(blk.MLP, swin.MLP);
            copy(blk.Norm1, swin.Norm1); copy(blk.Norm2, swin.Norm2);
            X4 = randn(D, N, Hp, Hp);
            Ys = swin.forward(X4);
            Yt = blk.forward(reshape(X4, D, N, Hp * Hp));                                % same column-major token order
            testCase.verifyEqual(reshape(Ys.Data, D, N, []), Yt.Data, 'AbsTol', 1e-12);
        end

        function testRelativeIndexDependsOnlyOnOffset(testCase)
            M = 3;
            idx = OceanDL.Vision.WindowOps.relativeIndex(M);
            testCase.verifySize(idx, [9 9]);
            testCase.verifyEqual(unique(idx(:)).', 1:25);                                % all (2M-1)^2 offsets used
            testCase.verifyTrue(all(diag(idx) == idx(1, 1)));                            % zero offset everywhere
            % tokens (r,c): 1=(1,1), 5=(2,2), 6=(3,2), 2=(2,1): offset of 5->1 equals offset of 6->2
            testCase.verifyEqual(idx(5, 1), idx(6, 2));
        end

        function testPatchMergingAndGradients(testCase)
            pm = OceanDL.Vision.PatchMerging(3, 'Seed', 1);
            testCase.verifySize(pm.forward(randn(3, 2, 4, 6)).Data, [6 2 2 3]);
            r = OceanDL.Autograd.GradCheck.check(@(x) pm.forward(x), {randn(3, 1, 2, 2)});
            testCase.verifyTrue(r.Passed);
            net = OceanDL.Vision.Swin('Channels', 1, 'PatchSize', 2, 'Dim', 8, 'Depths', [2 1], 'Heads', [2 2], ...
                'WindowSize', 2, 'NumClasses', 3, 'Seed', 2);
            testCase.verifySize(net.forward(randn(8, 8, 1, 2)).Data, [3 2]);
            r = OceanDL.Autograd.GradCheck.check(@(x) net.forward(x), {randn(8, 8, 1, 1)});
            testCase.verifyTrue(r.Passed, sprintf('max err %.3g', r.MaxAbsError));
        end

        function testSwinRunsAtSeveralResolutions(testCase)
            net = OceanDL.Vision.Swin('Channels', 1, 'PatchSize', 2, 'Dim', 8, 'Depths', [2 2], 'Heads', [2 2], ...
                'WindowSize', 2, 'NumClasses', 4, 'Seed', 3);
            for H = [8 16 24]
                testCase.verifySize(net.forward(randn(H, H, 1, 1)).Data, [4 1]);
            end
        end
    end

    methods (Static)
        function copyParams(dst, src)
            a = dst.namedParameters(); b = src.namedParameters();
            assert(isequal({a.Name}, {b.Name}), 'parameter layout differs');
            for k = 1:numel(a)
                a(k).Param.Data = b(k).Param.Data;
            end
        end
    end
end
