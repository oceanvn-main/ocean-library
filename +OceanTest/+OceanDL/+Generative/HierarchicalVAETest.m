classdef HierarchicalVAETest < OceanTest.BaseTestCase
    % HIERARCHICALVAETEST  Configurable hierarchical VAE:
    %   * KL between two diagonal Gaussians == Monte Carlo; KL to N(0,I) and
    %     KL(q||q) = 0 limits;
    %   * VDVAE bottleneck block and one top-down latent layer (prior net,
    %     residual posterior, KL, latent injection) == PyTorch (1e-10);
    %   * a ONE-latent hierarchy IS the VAE: OceanDL.Generative.VAE run on the
    %     same network gives exactly the same loss;
    %   * any configuration builds and runs (levels, latents per level,
    %     latent channels per level, block type, top prior), with per-layer
    %     latent shapes, sampling, reconstruction; invalid configs rejected;
    %   * per-layer KL bookkeeping and per-layer free bits; gradients;
    %   * costEstimate == an independent hand count of conv FLOPs;
    %   * 2-layer training on USPS improves the ELBO with both layers in use.

    methods (Test)
        function testKlToMatchesMonteCarloAndLimits(testCase)
            s = RandStream('mt19937ar', 'Seed', 1);
            q = OceanDL.Generative.DiagonalGaussian(randn(s, 3, 2), 0.5 * randn(s, 3, 2));
            p = OceanDL.Generative.DiagonalGaussian(randn(s, 3, 2), 0.5 * randn(s, 3, 2));
            kl = q.klTo(p).Data;
            z = q.Mu.Data + exp(q.LogVar.Data / 2) .* randn(s, 3, 2, 4e5);
            testCase.verifyEqual(mean(q.logProb(z).Data - p.logProb(z).Data, 3), kl, 'AbsTol', 0.02);
            testCase.verifyTrue(all(kl >= 0, 'all'));
            std0 = OceanDL.Generative.DiagonalGaussian(zeros(3, 2), zeros(3, 2));
            testCase.verifyEqual(q.klTo(std0).Data, q.klStandardNormal().Data, 'AbsTol', 1e-14);
            testCase.verifyEqual(q.klTo(q).Data, zeros(3, 2), 'AbsTol', 1e-14);
        end

        function testBottleneckBlockMatchesTorch(testCase)
            fx = OceanTest.OceanDL.Generative.BuildingBlocksTest.fixtures();
            B = OceanTest.OceanDL.Generative.BuildingBlocksTest;
            blk = OceanDL.Vision.BottleneckBlock(6, 8, 'Ratio', 0.25);
            B.loadConv(blk.Conv1, fx, 'bn_c1'); B.loadConv(blk.Conv2, fx, 'bn_c2'); B.loadConv(blk.Conv3, fx, 'bn_c3');
            B.loadConv(blk.Conv4, fx, 'bn_c4'); B.loadConv(blk.Skip, fx, 'bn_skip');
            X = OceanDL.Autograd.Tensor(fx.img('bn_x'), 'RequiresGrad', true);
            Y = blk.forward(X);
            testCase.verifyEqual(Y.Data, fx.img('bn_y'), 'AbsTol', 1e-10);
            backward(sum(Y .* fx.img('bn_R'), 'all'));
            testCase.verifyEqual(X.Grad, fx.img('bn_dx'), 'AbsTol', 1e-10);
        end

        function testTopDownBlockMatchesTorch(testCase)
            fx = OceanTest.OceanDL.Generative.BuildingBlocksTest.fixtures();
            B = OceanTest.OceanDL.Generative.BuildingBlocksTest;
            td = OceanDL.Generative.TopDownBlock(4, 2, 'Groups', 2);
            res = @(m, p) OceanTest.OceanDL.Generative.HierarchicalVAETest.loadRes(m, fx, p);
            res(td.PriorBlock, 'td_prior_block'); res(td.PostBlock, 'td_post_block'); res(td.Block, 'td_block');
            B.loadConv(td.PriorHead, fx, 'td_prior_head'); B.loadConv(td.PostHead, fx, 'td_post_head');
            B.loadConv(td.ZProj, fx, 'td_zproj');
            H = OceanDL.Autograd.Tensor(fx.img('td_h'), 'RequiresGrad', true);
            E = OceanDL.Autograd.Tensor(fx.img('td_e'), 'RequiresGrad', true);
            [Ho, kl] = td.forward(H, E, "mode");
            testCase.verifyEqual(Ho.Data, fx.img('td_ho'), 'AbsTol', 1e-10);
            testCase.verifyEqual(kl.Data, fx.img('td_kl'), 'AbsTol', 1e-10);
            backward(sum(Ho .* fx.img('td_Rh'), 'all') + sum(kl .* fx.img('td_Rk'), 'all'));
            testCase.verifyEqual(H.Grad, fx.img('td_dh'), 'AbsTol', 1e-10);
            testCase.verifyEqual(E.Grad, fx.img('td_de'), 'AbsTol', 1e-10);
            testCase.verifyEqual(td.PriorHead.Weight.Grad, permute(fx.raw('td_dprior_head_weight'), [3 4 2 1]), 'AbsTol', 1e-10);
            fresh = OceanDL.Generative.TopDownBlock(4, 2, 'Groups', 2, 'Seed', 3);      % zero-init prior head
            [~, ~, ~, ~, p] = fresh.forward(fx.img('td_h'), fx.img('td_e'), "mode");
            testCase.verifyEqual([max(abs(p.Mu.Data), [], 'all') max(abs(p.LogVar.Data), [], 'all')], [0 0]);   % starts as N(0, I)
        end

        function testOneLatentHierarchyIsTheVAE(testCase)
            X = rand(RandStream('mt19937ar', 'Seed', 1), 8, 8, 1, 5);
            h = OceanDL.Generative.HierarchicalVAE('Widths', [8 8], 'LatentsPerLevel', [0 1], 'LatentChannels', 3, ...
                'Groups', 4, 'Seed', 10);
            vae = OceanDL.Generative.VAE(OceanTest.OceanDL.Generative.HvaeEncoderAdapter(h), ...
                OceanTest.OceanDL.Generative.HvaeDecoderAdapter(h, [8 8]), 'Likelihood', "bernoulli", 'Seed', 10 + 9001);
            Lh = h.forward(X);
            Lv = vae.forward(X);                                                   % same network, same noise draw
            testCase.verifyEqual(Lh.Data, Lv.Data, 'AbsTol', 1e-10);
            testCase.verifyEqual(h.LastStats.KL, vae.LastStats.KL, 'AbsTol', 1e-10);
            testCase.verifyEqual(h.LastStats.Recon, vae.LastStats.Recon, 'AbsTol', 1e-10);
        end

        function testAnyConfigurationBuildsAndRuns(testCase)
            X = rand(RandStream('mt19937ar', 'Seed', 2), 16, 16, 1, 3);
            cfgs = {
                {'Widths', [8 16],       'LatentsPerLevel', [0 1]}
                {'Widths', [8 16 16],    'LatentsPerLevel', [0 1 1]}
                {'Widths', [8 16 16],    'LatentsPerLevel', [1 0 2], 'BlockType', "bottleneck"}
                {'Widths', [8 8 8 8],    'LatentsPerLevel', [1 1 1 1], 'TopPrior', "learned", 'Posterior', "direct"}
                {'Widths', [8 16 16],    'LatentsPerLevel', [2 1 1], 'LatentChannels', [2 3 4], 'BlocksPerLevel', 0}};
            for k = 1:numel(cfgs)
                c = cfgs{k};
                h = OceanDL.Generative.HierarchicalVAE(c{:}, 'Groups', 4, 'Seed', k);
                name = sprintf('config %d', k);
                L = h.forward(X);
                testCase.verifyTrue(isfinite(L.Data), name);
                testCase.verifyEqual(numel(h.LastStats.KLPerLayer), h.numLatentLayers(), name);
                qs = h.encode(X);
                lvls = repelem(numel(h.Widths):-1:1, h.LatentsPerLevel(end:-1:1));     % top first
                for i = 1:numel(qs)
                    l = lvls(i); r = 16 / 2 ^ (l - 1);
                    testCase.verifySize(qs{i}.Mu.Data, [r r h.LatentChannels(l) 3], sprintf('%s layer %d', name, i));
                end
                testCase.verifySize(h.sample(2, [16 16], 'Temperature', 0.8), [16 16 1 2], name);
                testCase.verifySize(h.reconstruct(X), [16 16 1 3], name);
                testCase.verifySize(h.activeUnits(X), [1 h.numLatentLayers()], name);
            end
            testCase.verifyError(@() OceanDL.Generative.HierarchicalVAE('Widths', [8 8], 'LatentsPerLevel', [0 0]), ...
                'OceanDL:Generative:HierarchicalVAE:Config');
            testCase.verifyError(@() OceanDL.Generative.HierarchicalVAE('Widths', [8 8], 'LatentsPerLevel', [1 1 1]), ...
                'OceanDL:Generative:HierarchicalVAE:Config');
            testCase.verifyError(@() OceanDL.Generative.HierarchicalVAE('Widths', [8 8 8], 'LatentChannels', [2 3]), ...
                'OceanDL:Generative:HierarchicalVAE:Config');
        end

        function testPerLayerKlBookkeepingFreeBitsAndGradients(testCase)
            X = rand(RandStream('mt19937ar', 'Seed', 3), 8, 8, 1, 4);
            mk = @(fb, beta) OceanDL.Generative.HierarchicalVAE('Widths', [8 8 8], 'LatentsPerLevel', [0 1 1], ...
                'LatentChannels', 2, 'Groups', 4, 'FreeBits', fb, 'Beta', beta, 'Seed', 5);
            a = mk(100, 1); c = mk(0, 0);
            La = a.forward(X); Lc = c.forward(X);
            testCase.verifyEqual(La.Data - Lc.Data, 2 * 100, 'AbsTol', 1e-9);        % two layers, both below lambda
            testCase.verifyEqual(a.LastStats.KL, sum(a.LastStats.KLPerLayer), 'AbsTol', 1e-12);
            testCase.verifyEqual(a.LastStats.NegELBO, a.LastStats.Recon + a.LastStats.KL, 'AbsTol', 1e-12);
            La.backward(); Lc.backward();
            Pa = a.namedParameters(); Pc = c.namedParameters();
            for i = 1:numel(Pa)
                if isempty(Pa(i).Param.Grad) && isempty(Pc(i).Param.Grad), continue; end
                testCase.verifyEqual(Pa(i).Param.Grad, Pc(i).Param.Grad, 'AbsTol', 1e-10, Pa(i).Name);
            end
            % Deterministic path (posterior means) is differentiable end to end.
            m = mk(0, 1);
            f = @(x) OceanTest.OceanDL.Generative.HierarchicalVAETest.meanPath(m, x);
            r = OceanDL.Autograd.GradCheck.check(f, {rand(RandStream('mt19937ar', 'Seed', 4), 4, 4, 1, 1)});
            testCase.verifyTrue(r.Passed, sprintf('max err %.3g', r.MaxAbsError));
        end

        function testLevelsAboveTheTopLatentAreNotBuilt(testCase)
            % [8 16 16] + [0 1 0] is the model [8 16] + [0 1]: same parameters, and
            % every parameter is reached by backpropagation (nothing dead). (A conv
            % bias directly before a GroupNorm has an exactly-zero gradient by
            % construction, so "reached" = a gradient exists, not "non-zero".)
            a = OceanDL.Generative.HierarchicalVAE('Widths', [8 16 16], 'LatentsPerLevel', [0 1 0], 'Groups', 4, 'Seed', 1);
            b = OceanDL.Generative.HierarchicalVAE('Widths', [8 16], 'LatentsPerLevel', [0 1], 'Groups', 4, 'Seed', 1);
            testCase.verifyEqual(a.Widths, [8 16]);
            testCase.verifyEqual(a.LatentsPerLevel, [0 1]);
            testCase.verifyEqual(a.numParameters(), b.numParameters());
            L = a.forward(rand(RandStream('mt19937ar', 'Seed', 1), 16, 16, 1, 3));
            L.backward();
            P = a.namedParameters();
            for i = 1:numel(P)
                testCase.verifyNotEmpty(P(i).Param.Grad, P(i).Name);
            end
        end

        function testFreeBitsGranularity(testCase)
            % With lambda far above every KL, each group contributes exactly lambda:
            % loss(lambda) - loss(no KL) = lambda x number of groups.
            X = rand(RandStream('mt19937ar', 'Seed', 3), 8, 8, 1, 4);
            mk = @(fb, beta, per) OceanDL.Generative.HierarchicalVAE('Widths', [8 8 8], 'LatentsPerLevel', [0 1 1], ...
                'LatentChannels', 2, 'Groups', 4, 'FreeBits', fb, 'FreeBitsPer', per, 'Beta', beta, 'Seed', 5);
            base = mk(0, 0, "layer").forward(X).Data;
            % layers: z at 4x4x2 (level 2) and 2x2x2 (level 3, top)
            groups = struct('layer', 2, 'channel', 2 + 2, 'unit', 4 * 4 * 2 + 2 * 2 * 2);
            for per = ["layer", "channel", "unit"]
                h = mk(100, 1, per);
                testCase.verifyEqual(h.forward(X).Data - base, 100 * groups.(per), 'AbsTol', 1e-8, per);
                testCase.verifyEqual(h.LastStats.NegELBO, h.LastStats.Recon + h.LastStats.KL, 'AbsTol', 1e-12, per);  % true KL reported
            end
        end

        function testLogVarClamp(testCase)
            % Soft clamp: |log-variance| <= c for prior and posterior; c = 0 is off.
            td = OceanDL.Generative.TopDownBlock(4, 2, 'Groups', 2, 'Seed', 3, 'LogVarClamp', 1);
            td.PostHead.Bias.Data(3:4) = 50;                                   % posterior log-variance channels huge
            [~, ~, ~, q] = td.forward(randn(4, 4, 4, 2), randn(4, 4, 4, 2), "mode");
            lv = q.LogVar.Data;
            testCase.verifyLessThanOrEqual(max(abs(lv), [], 'all'), 1);
            testCase.verifyGreaterThan(min(lv, [], 'all'), 0.99);             % tanh saturates at +c
            off = OceanDL.Generative.TopDownBlock(4, 2, 'Groups', 2, 'Seed', 3);
            off.PostHead.Bias.Data(3:4) = 50;
            [~, ~, ~, q0] = off.forward(randn(4, 4, 4, 2), randn(4, 4, 4, 2), "mode");
            testCase.verifyGreaterThan(min(q0.LogVar.Data, [], 'all'), 40);   % unclamped
        end

        function testCostEstimateEqualsAHandCount(testCase)
            h = OceanDL.Generative.HierarchicalVAE('InChannels', 1, 'Widths', [8 8], 'LatentsPerLevel', [0 1], ...
                'LatentChannels', 2, 'Groups', 4);
            T = h.costEstimate([8 8], 'Batch', 10, 'GFLOPS', 5);
            c = @(hw, k, ci, co) 2 * hw * k * k * ci * co;                      % 2 * Ho * Wo * k^2 * Cin * Cout
            res = @(hw, ci, co) c(hw, 3, ci, co) + c(hw, 3, co, co) + (ci ~= co) * c(hw, 1, ci, co);
            enc1 = c(64, 3, 1, 8) + res(64, 8, 8) + c(16, 3, 8, 8);              % in, block, stride-2 down (4x4 out)
            enc2 = res(16, 8, 8);
            dec2 = res(16, 16, 8) + c(16, 3, 8, 4) + c(16, 1, 2, 8) + res(16, 8, 8) ...   % top latent (standard prior)
                 + res(16, 8, 8);                                                % extra block at 4x4
            dec1 = c(64, 3, 8, 8) + res(64, 8, 8) + c(64, 3, 8, 1);              % lift conv (runs at 8x8), block, output
            testCase.verifyEqual(T.EncoderMFLOP, [enc1; enc2] / 1e6, 'AbsTol', 1e-12);
            testCase.verifyEqual(T.DecoderMFLOP, [dec1; dec2] / 1e6, 'AbsTol', 1e-12);
            u = T.Properties.UserData;
            testCase.verifyEqual(u.StepGFLOP, 3 * (enc1 + enc2 + dec1 + dec2) * 10 / 1e9, 'RelTol', 1e-12);
            testCase.verifyEqual(u.SecondsPerStep, u.StepGFLOP / 5, 'RelTol', 1e-12);
        end

        function testTwoLayerTrainingOnDigits(testCase)
            X = (OceanTest.OceanDL.Generative.VQVAETest.digits() + 1) / 2;       % pixel intensities in [0, 1]
            h = OceanDL.Generative.HierarchicalVAE('Widths', [16 32 32], 'LatentsPerLevel', [0 1 1], 'LatentChannels', 4, ...
                'Groups', 8, 'FreeBits', 0.5, 'Seed', 1).to('DataType', "single");
            X = single(X);
            opt = OceanDL.optim.ModuleOptimizer(h.parameters(), @() OceanStats.Optim.AdamWOptimizer('LR', 2e-3, 'WeightDecay', 0));
            rs = RandStream('mt19937ar', 'Seed', 2); neg = zeros(1, 150);
            for it = 1:150
                idx = randi(rs, size(X, 4), 1, 32);
                opt.zeroGrad(); L = h.forward(X(:, :, :, idx)); L.backward(); opt.step();
                neg(it) = h.LastStats.NegELBO;
            end
            testCase.verifyLessThan(mean(neg(end-9:end)), 0.7 * mean(neg(1:10)));
            testCase.verifyTrue(all(h.LastStats.KLPerLayer > 0.1));                  % both layers carry information
            testCase.verifyGreaterThan(min(h.activeUnits(X(:, :, :, 1:500))), 0);
        end
    end

    methods (Static)
        function loadRes(m, fx, prefix)
            B = OceanTest.OceanDL.Generative.BuildingBlocksTest;
            B.loadNorm(m.Norm1, fx, [prefix '_norm1']); B.loadNorm(m.Norm2, fx, [prefix '_norm2']);
            B.loadConv(m.Conv1, fx, [prefix '_conv1']); B.loadConv(m.Conv2, fx, [prefix '_conv2']);
            if ~isempty(m.Skip), B.loadConv(m.Skip, fx, [prefix '_skip']); end
        end

        function y = meanPath(m, x)
            [out, kls] = m.topDown(m.bottomUp(x), size(x, 4), size(x, 1:2), "mode");
            y = sum(out, 'all');
            for i = 1:numel(kls), y = y + sum(kls{i}, 'all'); end
        end
    end
end
