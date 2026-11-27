classdef DiffusionTrainingTest < OceanTest.BaseTestCase
    % DIFFUSIONTRAININGTEST  Networks and the training objective:
    %   * timestep embedding and ADM scale-shift residual block == PyTorch;
    %   * U-Net: shapes, zero-initialised output, time / class / skip paths,
    %     gradients;
    %   * ModelEMA == its closed-form weighted average, swap in / restore;
    %   * classifier-free guidance limits (w = 1 conditional, w = 0 null) and
    %     CondDropout = 1 == training with the null label;
    %   * EDM preconditioning algebra;
    %   * a trained MLP reaches the BAYES-OPTIMAL flow-matching loss of
    %     Gaussian data (computed from the analytic denoiser), for velocity and
    %     eps targets; minimum-SNR weighting formula;
    %   * a class-conditional U-Net trains and samples on USPS digits.

    methods (Test)
        function testTimestepEmbeddingMatchesTorch(testCase)
            fx = OceanTest.OceanDL.Generative.BuildingBlocksTest.fixtures();
            te = OceanDL.nn.TimestepEmbedding(16, 'FreqDim', 8);
            te.Fc1.Weight.Data = fx.raw('te_fc1_weight'); te.Fc1.Bias.Data = fx.col('te_fc1_bias');
            te.Fc2.Weight.Data = fx.raw('te_fc2_weight'); te.Fc2.Bias.Data = fx.col('te_fc2_bias');
            t = fx.col('te_t').';
            testCase.verifyEqual(OceanDL.nn.TimestepEmbedding.sinusoid(t, 8, 10000, 1000), fx.raw('te_phi').', 'AbsTol', 1e-12);
            testCase.verifyEqual(te.forward(t).Data, fx.raw('te_e').', 'AbsTol', 1e-12);
        end

        function testFiLMResBlockMatchesTorch(testCase)
            fx = OceanTest.OceanDL.Generative.BuildingBlocksTest.fixtures();
            blk = OceanDL.Vision.ConvResBlock(4, 6, 'Groups', 2, 'EmbedDim', 5);
            B = OceanTest.OceanDL.Generative.BuildingBlocksTest;
            B.loadNorm(blk.Norm1, fx, 'film_norm1'); B.loadNorm(blk.Norm2, fx, 'film_norm2');
            B.loadConv(blk.Conv1, fx, 'film_conv1'); B.loadConv(blk.Conv2, fx, 'film_conv2'); B.loadConv(blk.Skip, fx, 'film_skip');
            blk.EmbProj.Weight.Data = fx.raw('film_emb_weight'); blk.EmbProj.Bias.Data = fx.col('film_emb_bias');
            X = OceanDL.Autograd.Tensor(fx.img('film_x'), 'RequiresGrad', true);
            E = OceanDL.Autograd.Tensor(fx.raw('film_emb').', 'RequiresGrad', true);
            Y = blk.forward(X, E);
            testCase.verifyEqual(Y.Data, fx.img('film_y'), 'AbsTol', 1e-10);
            backward(sum(Y .* fx.img('film_R'), 'all'));
            testCase.verifyEqual(X.Grad, fx.img('film_dx'), 'AbsTol', 1e-10);
            testCase.verifyEqual(E.Grad, fx.raw('film_demb').', 'AbsTol', 1e-10);
            % Without an embedding the block is the unconditioned one (P1 behaviour).
            plain = OceanDL.Vision.ConvResBlock(4, 6, 'Groups', 2);
            testCase.verifyEmpty(plain.EmbProj);
        end

        function testUNetStructureConditioningAndGradients(testCase)
            net = OceanDL.Vision.UNet2D('InChannels', 1, 'Widths', [8 16], 'ResPerLevel', 1, 'NumClasses', 3, ...
                'Groups', 4, 'Seed', 1);
            X = randn(RandStream('mt19937ar', 'Seed', 2), 8, 8, 1, 2);
            y0 = net.forward(X, [0.3 0.7], [1; 2]);
            testCase.verifySize(y0.Data, [8 8 1 2]);
            testCase.verifyEqual(y0.Data, zeros(8, 8, 1, 2));                     % zero-initialised output conv
            s = RandStream('mt19937ar', 'Seed', 3);
            net.ConvOut.Weight.Data = 0.1 * randn(s, size(net.ConvOut.Weight.Data));
            base = net.forward(X, [0.3 0.7], [1; 2]).Data;
            testCase.verifyGreaterThan(max(abs(base - net.forward(X, [0.9 0.7], [1; 2]).Data), [], 'all'), 1e-8);   % time
            testCase.verifyGreaterThan(max(abs(base - net.forward(X, [0.3 0.7], [3; 2]).Data), [], 'all'), 1e-8);   % class
            testCase.verifyEqual(net.forward(X, [0.3 0.7], []).Data, net.forward(X, [0.3 0.7], [4; 4]).Data);     % [] = null label
            testCase.verifyEqual(numel(net.UpOps(net.UpOps == "pop")), 1 + 1 + 1 + 1);                             % every skip consumed
            tiny = OceanDL.Vision.UNet2D('InChannels', 1, 'Widths', [4 4], 'Groups', 2, 'Seed', 4);
            tiny.ConvOut.Weight.Data = 0.1 * randn(s, size(tiny.ConvOut.Weight.Data));
            r = OceanDL.Autograd.GradCheck.check(@(x) tiny.forward(x, 0.4), {randn(s, 4, 4, 1, 1)});
            testCase.verifyTrue(r.Passed, sprintf('max err %.3g', r.MaxAbsError));
        end

        function testModelEmaIsTheClosedFormAverage(testCase)
            net = OceanDL.nn.Linear(2, 2, 'Seed', 1);
            p0 = net.Weight.Data;
            ema = OceanDL.optim.ModelEMA(net, 'Decay', 0.9);
            P = {p0 + 1, p0 - 2, p0 + 5};
            for k = 1:3
                net.Weight.Data = P{k}; ema.update();
            end
            expected = 0.9 ^ 3 * p0 + 0.1 * (0.9 ^ 2 * P{1} + 0.9 * P{2} + P{3});
            testCase.verifyEqual(ema.Shadow('Weight'), expected, 'AbsTol', 1e-12);
            ema.applyTo(net);
            testCase.verifyEqual(net.Weight.Data, expected, 'AbsTol', 1e-12);
            ema.restore(net);
            testCase.verifyEqual(net.Weight.Data, P{3});
            w = OceanDL.optim.ModelEMA(net, 'Decay', 0.999, 'Warmup', true);
            testCase.verifyEqual(w.currentDecay(), 0.1);                          % (1 + 0) / (10 + 0)
        end

        function testGuidanceLimitsAndConditionDropout(testCase)
            import OceanDL.Generative.Diffusion.*
            sch = Schedules.RectifiedFlow();
            mk = @(p) DiffusionModel(MLPDenoiser(2, 'NumClasses', 2, 'Seed', 1), sch, 'Target', "velocity", ...
                'NumClasses', 2, 'Layout', "CB", 'CondDropout', p, 'Seed', 2);
            m = mk(0.1);
            x = randn(RandStream('mt19937ar', 'Seed', 3), 2, 5); y = [1; 2; 1; 2; 2];
            [c1, ~] = feval(m.predictor('Labels', y, 'Guidance', 1), x, 0.6);
            [u0, ~] = feval(m.predictor('Labels', y, 'Guidance', 0), x, 0.6);
            [cc, ~] = feval(m.predictor('Labels', y), x, 0.6);
            [nn_, ~] = feval(m.predictor('Labels', 3 * ones(5, 1)), x, 0.6);
            testCase.verifyEqual(c1, cc, 'AbsTol', 1e-12);
            testCase.verifyEqual(u0, nn_, 'AbsTol', 1e-12);
            [g2, ~] = feval(m.predictor('Labels', y, 'Guidance', 2), x, 0.6);
            testCase.verifyEqual(g2, u0 + 2 * (c1 - u0), 'AbsTol', 1e-10);         % linear in the prediction
            X = randn(RandStream('mt19937ar', 'Seed', 4), 2, 16);
            a = mk(1).forward(X, randi(2, 16, 1));                                  % every label dropped
            b = mk(0).forward(X, 3 * ones(16, 1));                                  % explicit null label
            testCase.verifyEqual(a.Data, b.Data, 'AbsTol', 1e-12);
        end

        function testEdmPreconditioningAlgebra(testCase)
            import OceanDL.Generative.Diffusion.*
            sch = Schedules.EDM('SigmaData', 0.5);
            net = MLPDenoiser(3, 'Seed', 1);
            m = DiffusionModel(net, sch, 'Target', "edm", 'Layout', "CB");
            x = randn(RandStream('mt19937ar', 'Seed', 2), 3, 4); t = 0.6; s = sch.sigma(t);
            [x0, e] = feval(m.predictor(), x, t);
            Fout = net.forward(x / sqrt(s ^ 2 + 0.25), repmat(log(s) / 4, 1, 4)).Data;
            testCase.verifyEqual(x0, 0.25 / (s ^ 2 + 0.25) * x + s * 0.5 / sqrt(s ^ 2 + 0.25) * Fout, 'AbsTol', 1e-12);
            testCase.verifyEqual(x, x0 + s * e, 'AbsTol', 1e-10);
            testCase.verifyError(@() DiffusionModel(net, Schedules.CosineVP(), 'Target', "edm"), ...
                'OceanDL:Generative:Diffusion:DiffusionModel:Target');
            testCase.verifyError(@() DiffusionModel(net, Schedules.RectifiedFlow(), 'Target', "eps"), ...
                'OceanDL:Generative:Diffusion:DiffusionModel:Target');
        end

        function testMinSnrWeightMatchesPublishedForms(testCase)
            import OceanDL.Generative.Diffusion.*
            sch = Schedules.CosineVP(); t = [0.1 0.4 0.8]; g = 5;
            snr = sch.snr(t);
            T = Targets;
            w = @(kind) min(snr, g) .* T.x0ErrorFactor(kind, sch.alpha(t), sch.sigma(t), sch.dalpha(t), sch.dsigma(t)) .^ 2;
            testCase.verifyEqual(w("eps"), min(snr, g) ./ snr, 'RelTol', 1e-12);          % Hang et al. 2023, eps
            testCase.verifyEqual(w("v"), min(snr, g) ./ (snr + 1), 'RelTol', 1e-12);     % ... v-prediction
            testCase.verifyEqual(w("x0"), min(snr, g), 'RelTol', 1e-12);
        end

        function testTrainedNetworkReachesTheBayesOptimalLoss(testCase)
            % Gaussian data: the minimum achievable flow-matching loss is the
            % expected squared error of the ANALYTIC optimal predictor. A small
            % MLP trained with DiffusionModel must come close to it.
            import OceanDL.Generative.Diffusion.*
            mu = [1; -1]; s0 = 0.5;
            for kind = ["velocity", "eps"]
                if kind == "velocity", sch = Schedules.RectifiedFlow(); else, sch = Schedules.CosineVP(); end
                m = DiffusionModel(MLPDenoiser(2, 'Hidden', 64, 'Seed', 1), sch, 'Target', kind, 'Layout', "CB", ...
                    'TimeSampler', TimeSampler("uniform", 'TMin', 0.02), 'Seed', 2);
                opt = OceanDL.optim.ModuleOptimizer(m.parameters(), @() OceanStats.Optim.AdamWOptimizer('LR', 2e-3, 'WeightDecay', 0));
                rs = RandStream('mt19937ar', 'Seed', 3);
                for it = 1:1500
                    opt.zeroGrad(); L = m.forward(mu + s0 * randn(rs, 2, 256)); L.backward(); opt.step();
                end
                % Evaluate both on the same fresh draws (t, eps, x0).
                es = RandStream('mt19937ar', 'Seed', 9); M = 20000;
                t = TimeSampler("uniform", 'TMin', 0.02).sample(M, es, sch);
                X0 = mu + s0 * randn(es, 2, M); E = randn(es, 2, M);
                opt_ = zeros(1, M); net_ = zeros(1, M); bayes = Analytic.gaussian(mu, s0, sch);
                for j = 1:M
                    a = sch.alpha(t(j)); s = sch.sigma(t(j)); da = sch.dalpha(t(j)); ds = sch.dsigma(t(j));
                    x = a * X0(:, j) + s * E(:, j);
                    target = Targets.fromX0Eps(kind, X0(:, j), E(:, j), a, s, da, ds);
                    [x0h, eh] = bayes(x, t(j));
                    opt_(j) = mean((Targets.fromX0Eps(kind, x0h, eh, a, s, da, ds) - target) .^ 2);
                    p = m.Net.forward(x, t(j)).Data;
                    net_(j) = mean((p - target) .^ 2);
                end
                best = mean(opt_); got = mean(net_);
                testCase.verifyGreaterThanOrEqual(got, best * 0.99, kind);             % cannot beat the Bayes optimum
                testCase.verifyLessThan(got, best * 1.10, kind);                        % ... and gets within 10%
            end
        end

        function testClassConditionalUNetTrainsAndSamplesOnDigits(testCase)
            import OceanDL.Generative.Diffusion.*
            X = OceanTest.OceanDL.Generative.VQVAETest.digits();
            [~, y] = OceanData.Adapters.UspsDigitsReader.read("test"); y = y + 1;
            net = OceanDL.Vision.UNet2D('InChannels', 1, 'Widths', [16 32], 'NumClasses', 10, 'Groups', 8, 'Seed', 1);
            m = DiffusionModel(net, Schedules.RectifiedFlow(), 'Target', "velocity", 'NumClasses', 10, 'Seed', 2);
            m.to('DataType', "single");
            X = single(X);
            opt = OceanDL.optim.ModuleOptimizer(m.parameters(), @() OceanStats.Optim.AdamWOptimizer('LR', 2e-3, 'WeightDecay', 0));
            ema = OceanDL.optim.ModelEMA(net, 'Decay', 0.99, 'Warmup', true);
            rs = RandStream('mt19937ar', 'Seed', 3); losses = zeros(1, 80);
            for it = 1:80
                idx = randi(rs, size(X, 4), 1, 32);
                opt.zeroGrad(); L = m.forward(X(:, :, :, idx), y(idx)); L.backward(); opt.step(); ema.update();
                losses(it) = L.Data;
            end
            testCase.verifyLessThan(mean(losses(end-9:end)), 0.8 * mean(losses(1:10)));
            ema.applyTo(net);
            [S, info] = m.generate(4, [16 16 1], Samplers.EulerODE('Steps', 4), 'Labels', [1; 2; 3; 4], 'Guidance', 2, ...
                'Stream', RandStream('mt19937ar', 'Seed', 4));
            ema.restore(net);
            testCase.verifySize(S, [16 16 1 4]);
            testCase.verifyClass(S, 'single');
            testCase.verifyTrue(all(isfinite(S), 'all'));
            testCase.verifyEqual(info.NFE, 4);
        end
    end
end
