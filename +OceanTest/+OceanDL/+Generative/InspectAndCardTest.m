classdef InspectAndCardTest < OceanTest.BaseTestCase
    % INSPECTANDCARDTEST  Tools behind the Model Lab GUI:
    %   * ForwardProcess.at == Eq. (2.2.1); describe() reports the book's
    %     discrete quantities for LinearVP (step i, beta_i, alpha_i);
    %   * ReverseProcess run step by step (in chunks) ends at EXACTLY the
    %     sampler's result for DDPM, DDIM and Heun; frames hold x_i, x_phi,
    %     eps_phi with x_i = alphaBar x_phi + sigma eps_phi, and Mu / Var equal
    %     Eq. (2.2.4) from the discrete accessors; the stride keeps the last step;
    %   * textbook DDPM options: integer training steps i / L; x_phi clipping;
    %   * ModelCard rebuilds a model (nested recipes) with identical outputs.

    methods (Test)
        function testForwardProcess(testCase)
            import OceanDL.Generative.*
            sch = Diffusion.Schedules.LinearVP();
            s = RandStream('mt19937ar', 'Seed', 1);
            x0 = randn(s, 4, 4); e = randn(s, 4, 4);
            [xt, info] = Inspect.ForwardProcess.at(x0, sch, 0.25, e);
            testCase.verifyEqual(xt, sch.alphaBar(250) * x0 + sqrt(1 - sch.alphaBar(250) ^ 2) * e, 'AbsTol', 1e-12);
            testCase.verifyEqual([info.Step info.BetaSq info.AlphaStep], [250 sch.betaSq(250) sch.alphaStep(250)], 'AbsTol', 1e-15);
            testCase.verifyEqual(info.SNR, sch.alphaBar(250) ^ 2 / (1 - sch.alphaBar(250) ^ 2), 'RelTol', 1e-12);
            F = Inspect.ForwardProcess.trajectory(x0, sch, [0 0.5 1], e);
            testCase.verifyEqual(F(1).X, x0, 'AbsTol', 1e-15);                              % t = 0: clean
            testCase.verifyEqual(height(Inspect.ForwardProcess.scheduleTable(sch)), 1001);
        end

        function testReverseProcessMatchesSamplers(testCase)
            import OceanDL.Generative.*
            sch = Diffusion.Schedules.LinearVP('L', 200);
            pred = Diffusion.Analytic.gmm([-2 2], [0.3 0.3], [0.5 0.5], sch);
            X1 = randn(RandStream('mt19937ar', 'Seed', 2), 1, 7);
            smps = {Diffusion.Samplers.DDPMAncestral('Steps', 200), Diffusion.Samplers.DDIM('Steps', 20), ...
                    Diffusion.Samplers.HeunODE('Steps', 10)};
            for k = 1:numel(smps)
                ref = smps{k}.sample(pred, sch, [1 7], 'X1', X1, 'Stream', RandStream('mt19937ar', 'Seed', 3));
                rp = Inspect.ReverseProcess(pred, sch, smps{k}, [1 7], 'X1', X1, ...
                    'Stream', RandStream('mt19937ar', 'Seed', 3), 'Stride', 7);
                while ~rp.advance(3), end                                                    % in chunks, like the GUI
                testCase.verifyEqual(rp.X, ref, 'AbsTol', 1e-12, class(smps{k}));
                testCase.verifyEqual(rp.Frames(end).Step, rp.numSteps(), class(smps{k}));  % last step always kept
                testCase.verifyEqual(rp.progress(), 1);
            end
            % Frames of the DDPM run: consistency and Eq. (2.2.4).
            rp = Inspect.ReverseProcess(pred, sch, Diffusion.Samplers.DDPMAncestral('Steps', 200), [1 7], 'X1', X1, 'Stride', 50);
            rp.advance();
            for f = rp.Frames
                a = sch.alpha(f.T); sg = sch.sigma(f.T);
                testCase.verifyEqual(f.X, a * f.X0 + sg * f.Eps, 'AbsTol', 1e-10);
                i = round(f.T * 200);
                if i > 1
                    ab = sch.alphaBar(i); abm = sch.alphaBar(i - 1); b2 = sch.betaSq(i); al = sch.alphaStep(i);
                    testCase.verifyEqual(f.Mu, abm * b2 / (1 - ab ^ 2) * f.X0 + (1 - abm ^ 2) * al / (1 - ab ^ 2) * f.X, 'AbsTol', 1e-12);
                    testCase.verifyEqual(f.Var, (1 - abm ^ 2) / (1 - ab ^ 2) * b2, 'AbsTol', 1e-15);
                end
            end
        end

        function testTextbookDdpmOptions(testCase)
            import OceanDL.Generative.Diffusion.*
            ts = TimeSampler("uniform", 'Discrete', 1000);
            t = ts.sample(5000, RandStream('mt19937ar', 'Seed', 4), []);
            testCase.verifyEqual(t * 1000, round(t * 1000), 'AbsTol', 1e-9);                 % integer steps
            testCase.verifyGreaterThanOrEqual(min(t), 1e-3); testCase.verifyLessThanOrEqual(max(t), 1);
            testCase.verifyError(@() TimeSampler("logitnormal", 'Discrete', 10), 'OceanDL:Generative:Diffusion:TimeSampler:Discrete');
            sch = Schedules.LinearVP();
            wild = @(x, t) deal(5 * ones(size(x)), zeros(size(x)));                           % out-of-range clean estimate
            clip = Samplers.ISampler.clipped(@(x, t) OceanTest.OceanDL.Generative.InspectAndCardTest.twoOut(wild, x, t), sch, [-1 1]);
            [x0, e] = clip(0.3 * ones(2), 0.5);
            testCase.verifyEqual(x0, ones(2));
            testCase.verifyEqual(0.3 * ones(2), sch.alpha(0.5) * x0 + sch.sigma(0.5) * e, 'AbsTol', 1e-12);
        end

        function testModelCardRoundTrip(testCase)
            import OceanDL.Training.ModelCard
            dir_ = tempname; mkdir(dir_); cleanup = onCleanup(@() rmdir(dir_, 's'));
            R = @ModelCard.recipe;
            % (1) a hierarchical VAE (args only)
            hr = R("OceanDL.Generative.HierarchicalVAE", {}, {'Widths', [8 8 8], 'LatentsPerLevel', [0 1 1], ...
                'LatentChannels', 2, 'Groups', 4, 'Seed', 3});
            h = ModelCard.build(hr); h.to('DataType', "single");
            ModelCard.save(fullfile(dir_, 'h.mat'), h, hr, 'Meta', struct('Dataset', "toy"));
            [h2, card] = ModelCard.load(fullfile(dir_, 'h.mat'));
            X = rand(8, 8, 1, 3, 'single');
            testCase.verifyEqual(h2.reconstruct(X), h.reconstruct(X), 'AbsTol', 1e-6);
            testCase.verifyEqual([card.Meta.Dataset, card.DataType], ["toy", "single"]);
            % (2) a DDPM with nested recipes (U-Net + schedule + time sampler)
            netR = R("OceanDL.Vision.UNet2D", {}, {'InChannels', 1, 'Widths', [8 8], 'NumClasses', 2, 'Groups', 4, 'Seed', 1});
            dmR = R("OceanDL.Generative.Diffusion.DiffusionModel", ...
                {netR, R("OceanDL.Generative.Diffusion.Schedules.LinearVP", {}, {'L', 100})}, ...
                {'Target', "eps", 'NumClasses', 2, 'TimeSampler', R("OceanDL.Generative.Diffusion.TimeSampler", {"uniform"}, {'Discrete', 100})});
            dm = ModelCard.build(dmR);
            P = dm.namedParameters(); for k = 1:numel(P), P(k).Param.Data = P(k).Param.Data + 0.01 * k; end   % non-default weights
            ModelCard.save(fullfile(dir_, 'd.mat'), dm, dmR);
            dm2 = ModelCard.load(fullfile(dir_, 'd.mat'));
            testCase.verifyEqual(dm2.Schedule.L, 100);
            testCase.verifyEqual(dm2.Times.Discrete, 100);
            x = randn(8, 8, 1, 2);
            [a0, ~] = feval(dm.predictor('Labels', [1; 2]), x, 0.4);
            [b0, ~] = feval(dm2.predictor('Labels', [1; 2]), x, 0.4);
            testCase.verifyEqual(b0, a0, 'AbsTol', 1e-12);
        end

        function testCardStateIsSavedOnHost(testCase)
            % Weights trained on a GPU (a containers.Map of gpuArrays) must be
            % saved as host arrays, and the caller's live map must stay untouched.
            testCase.assumeTrue(OceanDL.Device.gpuAvailable(), 'needs a GPU');
            live = containers.Map({'w', 'b'}, {gpuArray(single([1 2; 3 4])), gpuArray(single(5))});
            host = OceanDL.Training.ModelCard.toHost(struct('S', {{live}}));
            m = host.S{1};
            testCase.verifyFalse(isa(m('w'), 'gpuArray'));
            testCase.verifyEqual(m('w'), single([1 2; 3 4]));
            testCase.verifyTrue(isa(live('w'), 'gpuArray'), 'the caller''s map must not be modified');
        end
    end

    methods (Static)
        function [a, b] = twoOut(fn, x, t)
            [a, b] = fn(x, t);
        end
    end
end
