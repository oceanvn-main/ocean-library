classdef QuantizerTest < OceanTest.BaseTestCase
    % QUANTIZERTEST  Discrete bottlenecks:
    %   VQ (gradient codebook) == PyTorch: indices, loss, straight-through
    %   output, dL/dz and dL/dcodebook; EMA with Decay 0 == one exact Lloyd
    %   (k-means) step; dead codes are re-seeded onto encoder outputs; buffers
    %   follow Module.to; FSQ == the paper's reference algorithm (values,
    %   indices, gradients) and its index <-> code map is a bijection.

    methods (Test)
        function testVectorQuantizerMatchesTorch(testCase)
            fx = OceanTest.OceanDL.Generative.BuildingBlocksTest.fixtures();
            vq = OceanDL.Generative.Quantizers.VectorQuantizer(7, 3, 'Update', "grad", 'Init', "uniform", ...
                'RestartDead', false, 'Seed', 1);
            vq.Codebook.Data = fx.raw('vq_E').';                                        % torch (K x D)
            z = OceanDL.Autograd.Tensor(fx.img('vq_z'), 'RequiresGrad', true);
            [zq, info] = vq.forward(z);
            testCase.verifyEqual(info.Indices, permute(fx.raw('vq_idx_NHW'), [2 3 1]));  % (h x w x N)
            testCase.verifyEqual(zq.Data, fx.img('vq_zq'), 'AbsTol', 1e-12);
            testCase.verifyEqual(info.Loss.Data, fx.raw('vq_loss'), 'AbsTol', 1e-12);
            backward(info.Loss + sum(zq .* fx.img('vq_R'), 'all'));
            testCase.verifyEqual(z.Grad, fx.img('vq_dz'), 'AbsTol', 1e-12);
            testCase.verifyEqual(vq.Codebook.Grad, fx.raw('vq_dE').', 'AbsTol', 1e-12);
        end

        function testStraightThroughPassesDecoderGradientUnchanged(testCase)
            vq = OceanDL.Generative.Quantizers.VectorQuantizer(5, 2, 'Beta', 0, 'Update', "ema", 'Layout', "CB", 'Seed', 2);
            vq.eval();
            z = OceanDL.Autograd.Tensor(randn(2, 9), 'RequiresGrad', true);
            G = randn(2, 9);
            [zq, ~] = vq.forward(z);
            backward(sum(zq .* G, 'all'));
            testCase.verifyEqual(z.Grad, G, 'AbsTol', 1e-15);                        % identity Jacobian
        end

        function testEmaWithZeroDecayIsOneLloydStep(testCase)
            s = RandStream('mt19937ar', 'Seed', 3);
            Z = [randn(s, 2, 20) + [4; 0], randn(s, 2, 20) - [4; 0], randn(s, 2, 20) + [0; 6]];
            vq = OceanDL.Generative.Quantizers.VectorQuantizer(3, 2, 'Update', "ema", 'Decay', 0, 'Epsilon', 1e-12, ...
                'Init', "uniform", 'RestartDead', false, 'Layout', "CB", 'Seed', 4);
            E0 = vq.codes();
            [~, a] = min(sum(E0 .^ 2, 1).' - 2 * E0.' * Z + sum(Z .^ 2, 1), [], 1);     % assignment to the OLD codes
            vq.forward(Z);
            E1 = vq.codes();
            for k = unique(a)
                testCase.verifyEqual(E1(:, k), mean(Z(:, a == k), 2), 'AbsTol', 1e-9);  % centroid of its cluster
            end
        end

        function testDeadCodesAreReseededOntoEncoderOutputs(testCase)
            Z = randn(RandStream('mt19937ar', 'Seed', 5), 4, 50) + 10;              % far from the uniform init
            vq = OceanDL.Generative.Quantizers.VectorQuantizer(16, 4, 'Update', "grad", 'Init', "uniform", ...
                'DeadThreshold', 1, 'Decay', 0.5, 'Layout', "CB", 'Seed', 6);
            [~, info1] = vq.forward(Z);
            used = round(info1.Usage * 16);
            testCase.verifyLessThanOrEqual(used, 3);                                % collapse: a few codes take all
            testCase.verifyEqual(info1.Restarted, 16 - used);                        % every unused code is dead
            E = vq.codes();
            dmin = min(sqrt(max(0, sum(E .^ 2, 1).' - 2 * E.' * Z + sum(Z .^ 2, 1))), [], 2);
            testCase.verifyEqual(nnz(dmin < 1e-2), 16 - used);                       % re-seeded codes sit on data
            [~, info2] = vq.forward(Z);
            testCase.verifyGreaterThan(info2.Usage, 0.5);
            % data init: codes start on encoder outputs
            vq2 = OceanDL.Generative.Quantizers.VectorQuantizer(8, 4, 'Init', "data", 'Layout', "CB", 'Seed', 7);
            [~, info3] = vq2.forward(Z);
            testCase.verifyGreaterThan(info3.Usage, 0.5);
        end

        function testBuffersFollowModuleTo(testCase)
            vq = OceanDL.Generative.Quantizers.VectorQuantizer(4, 3, 'Update', "ema", 'CodeDim', 2, 'Normalize', true, ...
                'Layout', "CB", 'Seed', 8).to('DataType', "single");
            testCase.verifyClass(vq.EmaCodebook, 'single');
            testCase.verifyClass(vq.ProjIn.Weight.Data, 'single');
            s = RandStream('mt19937ar', 'Seed', 9);
            vq.forward(randn(s, 3, 10, 'single'));                               % training step: data init + EMA
            vq.eval();                                                             % frozen codebook for the comparison
            [zq, info] = vq.forward(randn(s, 3, 10, 'single'));
            testCase.verifyClass(zq.Data, 'single');
            testCase.verifySize(zq.Data, [3 10]);
            testCase.verifyClass(vq.lookup(info.Indices), 'single');
            testCase.verifyEqual(double(vq.lookup(info.Indices)), double(zq.Data), 'AbsTol', 1e-6);   % single: z_e + (e - z_e) rounds
        end

        function testFsqMatchesPaperReference(testCase)
            fx = OceanTest.OceanDL.Generative.BuildingBlocksTest.fixtures();
            q = OceanDL.Generative.Quantizers.FSQ([8 5 5 5]);
            z = OceanDL.Autograd.Tensor(fx.img('fsq_z'), 'RequiresGrad', true);
            [zq, info] = q.forward(z);
            testCase.verifyEqual(zq.Data, fx.img('fsq_zhat'), 'AbsTol', 1e-12);
            testCase.verifyEqual(info.Indices, permute(fx.raw('fsq_idx_NHW'), [2 3 1]), 'AbsTol', 1e-9);
            backward(sum(zq .* fx.img('fsq_R'), 'all'));
            testCase.verifyEqual(z.Grad, fx.img('fsq_dz'), 'AbsTol', 1e-12);
            testCase.verifyEqual(q.lookup(info.Indices), zq.Data, 'AbsTol', 1e-12);
        end

        function testFsqIndexCodeMapIsABijection(testCase)
            q = OceanDL.Generative.Quantizers.FSQ([4 3 5], 'Layout', "CB");
            K = q.numCodes();
            testCase.verifyEqual(K, 60);
            codes = q.lookup((1:K).');                                             % 3 x 60 grid points
            testCase.verifyEqual(size(unique(codes.', 'rows'), 1), K);
            % Feeding a grid point back (pre-tanh) returns the same index:
            % invert the bound so that tanh(z + shift) * half - off = code * h.
            L = [4 3 5].'; h = floor(L / 2); half = (L - 1) * (1 + 1e-3) / 2; off = 0.5 * (mod(L, 2) == 0);
            z = atanh((codes .* h + off) ./ half) - atanh(off ./ half);
            [~, info] = q.forward(z);
            testCase.verifyEqual(info.Indices(:), (1:K).');
            testCase.verifyEqual(info.Usage, 1);
            testCase.verifyEqual(info.Perplexity, K, 'RelTol', 1e-12);
        end
    end
end
