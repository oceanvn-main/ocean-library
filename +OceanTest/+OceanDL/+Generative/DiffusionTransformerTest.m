classdef DiffusionTransformerTest < OceanTest.BaseTestCase
    % DIFFUSIONTRANSFORMERTEST  Transformer denoisers against PyTorch (1e-10):
    %   * DiT block: adaLN-Zero modulation, parameter-free LayerNorm, GELU MLP
    %     (values, input / condition gradients, adaLN weight gradient);
    %   * MMDiT block with every modern option at once: RMSNorm, SwiGLU,
    %     per-head RMS QK-norm, 2-D axial RoPE on the image stream, joint
    %     attention across the two streams;
    %   * RoPE property: scores depend only on RELATIVE 2-D offsets;
    %   * full DiT / MMDiT: shapes, zero-initialised output, conditioning,
    %     gradients, single precision, and short flow-matching training.

    methods (Test)
        function testDiTBlockMatchesTorch(testCase)
            fx = OceanTest.OceanDL.Generative.BuildingBlocksTest.fixtures();
            tok = @(n) permute(fx.raw(n), [3 1 2]);                                % torch (N,T,D) -> D x N x T
            blk = OceanDL.Vision.DiTBlock(8, 2);
            L = @OceanTest.OceanDL.Generative.DiffusionTransformerTest.loadLinear;
            L(blk.Attention.Query, fx, 'dit_attn_q'); L(blk.Attention.Key, fx, 'dit_attn_k');
            L(blk.Attention.Value, fx, 'dit_attn_v'); L(blk.Attention.Output, fx, 'dit_attn_o');
            L(blk.MLP.Fc1, fx, 'dit_fc1'); L(blk.MLP.Fc2, fx, 'dit_fc2'); L(blk.AdaLN, fx, 'dit_ada');
            X = OceanDL.Autograd.Tensor(tok('dit_x'), 'RequiresGrad', true);
            C = OceanDL.Autograd.Tensor(fx.raw('dit_c').', 'RequiresGrad', true);
            Y = blk.forward(X, C);
            testCase.verifyEqual(Y.Data, tok('dit_y'), 'AbsTol', 1e-10);
            backward(sum(Y .* tok('dit_R'), 'all'));
            testCase.verifyEqual(X.Grad, tok('dit_dx'), 'AbsTol', 1e-10);
            testCase.verifyEqual(C.Grad, fx.raw('dit_dc').', 'AbsTol', 1e-10);
            testCase.verifyEqual(blk.AdaLN.Weight.Grad, fx.raw('dit_dada_weight'), 'AbsTol', 1e-10);
        end

        function testMMDiTBlockWithModernOptionsMatchesTorch(testCase)
            fx = OceanTest.OceanDL.Generative.BuildingBlocksTest.fixtures();
            tok = @(n) permute(fx.raw(n), [3 1 2]);
            blk = OceanDL.Vision.MMDiTBlock(8, 2, 'MLPRatio', 1.5, 'Norm', "rms", 'FFN', "swiglu", 'QKNorm', "rms");
            L = @OceanTest.OceanDL.Generative.DiffusionTransformerTest.loadLinear;
            streams = {blk.Attention.A, 'mm_ax'; blk.Attention.B, 'mm_az'};
            for s = 1:2
                [m, p] = streams{s, :};
                L(m.Query, fx, [p '_q']); L(m.Key, fx, [p '_k']); L(m.Value, fx, [p '_v']); L(m.Output, fx, [p '_o']);
                m.QNorm.Weight.Data = fx.col([p '_qn']); m.KNorm.Weight.Data = fx.col([p '_kn']);
            end
            L(blk.MLPX.Fc1, fx, 'mm_x1'); L(blk.MLPX.Fc2, fx, 'mm_x2');
            L(blk.MLPZ.Fc1, fx, 'mm_z1'); L(blk.MLPZ.Fc2, fx, 'mm_z2');
            L(blk.AdaX, fx, 'mm_adax'); L(blk.AdaZ, fx, 'mm_adaz');
            X = OceanDL.Autograd.Tensor(tok('mm_x'), 'RequiresGrad', true);
            Z = OceanDL.Autograd.Tensor(tok('mm_z'), 'RequiresGrad', true);
            C = OceanDL.Autograd.Tensor(fx.raw('mm_c').', 'RequiresGrad', true);
            [Yx, Yz] = blk.forward(X, Z, C, OceanDL.Transformer.RoPE.axial2d(2, 3, 4));
            testCase.verifyEqual(Yx.Data, tok('mm_yx'), 'AbsTol', 1e-10);
            testCase.verifyEqual(Yz.Data, tok('mm_yz'), 'AbsTol', 1e-10);
            backward(sum(Yx .* tok('mm_Rx'), 'all') + sum(Yz .* tok('mm_Rz'), 'all'));
            testCase.verifyEqual(X.Grad, tok('mm_dx'), 'AbsTol', 1e-10);
            testCase.verifyEqual(Z.Grad, tok('mm_dz'), 'AbsTol', 1e-10);
            testCase.verifyEqual(C.Grad, fx.raw('mm_dc').', 'AbsTol', 1e-10);
            testCase.verifyEqual(blk.Attention.A.QNorm.Weight.Grad, fx.col('mm_dqn'), 'AbsTol', 1e-10);
        end

        function testRopeScoresDependOnlyOnRelativeOffsets(testCase)
            d = 8; Hp = 5; Wp = 6;
            r = OceanDL.Transformer.RoPE.axial2d(Hp, Wp, d);
            s = RandStream('mt19937ar', 'Seed', 1);
            q = randn(s, d, 1); k = randn(s, d, 1);
            at = @(row, col) sub2ind([Hp Wp], row, col);                              % column-major token index
            Q = OceanDL.Transformer.RoPE.apply(repmat(q, 1, Hp * Wp), r);
            K = OceanDL.Transformer.RoPE.apply(repmat(k, 1, Hp * Wp), r);
            score = @(a, b) Q(:, a).' * K(:, b);
            testCase.verifyEqual(score(at(1, 1), at(3, 2)), score(at(2, 4), at(4, 5)), 'AbsTol', 1e-12);   % same (2, 1) offset
            testCase.verifyEqual(score(at(5, 6), at(4, 3)), score(at(2, 4), at(1, 1)), 'AbsTol', 1e-12);   % same (-1, -3) offset
            testCase.verifyGreaterThan(abs(score(at(1, 1), at(3, 2)) - score(at(1, 1), at(2, 3))), 1e-6);  % different offsets differ
            testCase.verifyEqual(vecnorm(Q), vecnorm(repmat(q, 1, Hp * Wp)), 'AbsTol', 1e-12);             % a rotation
            testCase.verifyError(@() OceanDL.Transformer.RoPE.axial2d(2, 2, 6), 'OceanDL:Transformer:RoPE:Dim');
        end

        function testDiTAndMMDiTModels(testCase)
            s = RandStream('mt19937ar', 'Seed', 2);
            X = randn(s, 8, 8, 1, 2);
            nets = {OceanDL.Vision.DiT('InChannels', 1, 'PatchSize', 2, 'Dim', 16, 'Depth', 2, 'Heads', 2, 'NumClasses', 3, 'Seed', 1)
                    OceanDL.Vision.DiT('InChannels', 1, 'PatchSize', 2, 'Dim', 16, 'Depth', 2, 'Heads', 2, 'Positions', "rope", ...
                        'Norm', "rms", 'FFN', "swiglu", 'QKNorm', "rms", 'Seed', 2)
                    OceanDL.Vision.MMDiT('InChannels', 1, 'PatchSize', 2, 'Dim', 16, 'Depth', 2, 'Heads', 2, 'NumClasses', 3, ...
                        'ContextTokens', 2, 'Positions', "rope", 'QKNorm', "rms", 'Seed', 3)};
            for k = 1:numel(nets)
                net = nets{k}; name = sprintf('%s #%d', class(net), k);
                y0 = net.forward(X, [0.2 0.8], [1; 2]);
                testCase.verifySize(y0.Data, [8 8 1 2], name);
                testCase.verifyEqual(y0.Data, zeros(8, 8, 1, 2), name);                 % adaLN-Zero + zero final layer
                % Give every zero-initialised modulation / output weight a value.
                P = net.namedParameters();
                for i = 1:numel(P)
                    if all(P(i).Param.Data == 0, 'all'), P(i).Param.Data = 0.1 * randn(s, size(P(i).Param.Data)); end
                end
                base = net.forward(X, [0.2 0.8], [1; 2]).Data;
                testCase.verifyGreaterThan(max(abs(base - net.forward(X, [0.6 0.8], [1; 2]).Data), [], 'all'), 1e-8, name);
                if net.NumClasses > 0
                    testCase.verifyGreaterThan(max(abs(base - net.forward(X, [0.2 0.8], [3; 2]).Data), [], 'all'), 1e-8, name);
                end
                r = OceanDL.Autograd.GradCheck.check(@(x) net.forward(x, 0.3, []), {randn(s, 4, 4, 1, 1)});
                testCase.verifyTrue(r.Passed, sprintf('%s max err %.3g', name, r.MaxAbsError));
                net.to('DataType', "single");
                testCase.verifyClass(net.forward(single(X), [0.2 0.8], []).Data, 'single', name);
            end
        end

        function testDiffusionTransformersTrainOnDigits(testCase)
            import OceanDL.Generative.Diffusion.*
            X = single(OceanTest.OceanDL.Generative.VQVAETest.digits());
            [~, y] = OceanData.Adapters.UspsDigitsReader.read("test"); y = y + 1;
            nets = {OceanDL.Vision.DiT('InChannels', 1, 'PatchSize', 4, 'Dim', 32, 'Depth', 2, 'Heads', 4, 'NumClasses', 10, ...
                        'Positions', "rope", 'QKNorm', "rms", 'Seed', 1)
                    OceanDL.Vision.MMDiT('InChannels', 1, 'PatchSize', 4, 'Dim', 32, 'Depth', 2, 'Heads', 4, 'NumClasses', 10, ...
                        'ContextTokens', 2, 'Seed', 2)};
            for k = 1:numel(nets)
                m = DiffusionModel(nets{k}, Schedules.RectifiedFlow(), 'Target', "velocity", 'NumClasses', 10, 'Seed', 3);
                m.to('DataType', "single");
                opt = OceanDL.optim.ModuleOptimizer.forModule(m, @(wd) OceanStats.Optim.AdamWOptimizer('LR', 1e-3, 'WeightDecay', wd), ...
                    'WeightDecay', 0);
                rs = RandStream('mt19937ar', 'Seed', 4); losses = zeros(1, 120);
                for it = 1:120
                    idx = randi(rs, size(X, 4), 1, 32);
                    opt.zeroGrad(); L = m.forward(X(:, :, :, idx), y(idx)); L.backward(); opt.step();
                    losses(it) = L.Data;
                end
                testCase.verifyLessThan(mean(losses(end-19:end)), 0.8 * mean(losses(1:20)), class(nets{k}));
                S = m.generate(2, [16 16 1], Samplers.HeunODE('Steps', 3), 'Labels', [5; 6], 'Guidance', 1.5);
                testCase.verifySize(S, [16 16 1 2]);
                testCase.verifyTrue(all(isfinite(S), 'all'));
            end
        end
    end

    methods (Static)
        function loadLinear(m, fx, prefix)
            m.Weight.Data = fx.raw([prefix '_weight']);
            m.Bias.Data = fx.col([prefix '_bias']);
        end
    end
end
