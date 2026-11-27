classdef InterpretTest < matlab.unittest.TestCase
    % INTERPRETTEST  OceanDL.Interpret (feature visualisation, Szeliski 5.4.5)
    %   against closed-form answers (TinyTraceable: one known 3x3 conv + SiLU)
    %   and against the model's own encode / reconstruct / KL (small random HVAE).

    methods (Test)
        function testBackpropRules(tc)
            a = [-2 -0.5 0 0.3 1.5];
            g = [1 -1 2 -3 0.5];
            s = 1 ./ (1 + exp(-a)); dsilu = s + a .* s .* (1 - s);
            for rule = ["standard", "deconvnet", "guided"]
                x = OceanDL.Autograd.Tensor(a, 'RequiresGrad', true);
                OceanDL.Autograd.BackpropRule.run(rule, @() backward(OceanDL.Autograd.F.silu(x), g));
                switch rule
                    case "standard",  want = g .* dsilu;
                    case "deconvnet", want = max(g, 0);
                    case "guided",    want = max(g, 0) .* dsilu .* (a > 0);
                end
                tc.verifyEqual(x.Grad, want, 'AbsTol', 1e-12, rule);
            end
            tc.verifyEqual(OceanDL.Autograd.BackpropRule.current(), "standard");   % restored
        end

        function testTraceMatchesTheModel(tc)
            h = OceanTest.OceanDL.Interpret.InterpretTest.hvae();
            X = rand(RandStream('mt19937ar', 'Seed', 1), 8, 8, 1, 3);
            R = @OceanDL.Autograd.Tensor.raw;
            [acts, order] = OceanDL.Autograd.GradMode.noGrad(@() h.trace(X));
            qs = h.encode(X);
            for i = 1:numel(qs)
                tc.verifyEqual(R(acts(sprintf('z%d.mu', i))), OceanDL.Autograd.Tensor.raw(qs{i}.Mu), 'AbsTol', 1e-12);
            end
            tc.verifyEqual(1 ./ (1 + exp(-R(acts('dec.out')))), h.reconstruct(X), 'AbsTol', 1e-12);
            tc.verifyEqual(order(1), "enc.L1.in");
            T = h.traceNames([8 8 1]);
            tc.verifyEqual(T.Name, order(:));
            tc.verifyEqual(unique(T.Section).', ["decoder", "encoder", "latent"]);
            % 'Parts' encoder stops after the bottom-up pass
            enc = OceanDL.Autograd.GradMode.noGrad(@() h.trace(X, 'Parts', "encoder"));
            tc.verifyFalse(isKey(enc, 'dec.out'));
            tc.verifyEqual(R(enc('enc.L2.block1')), R(acts('enc.L2.block1')), 'AbsTol', 1e-12);
        end

        function testUnitGradientIsTheTrueGradient(tc)
            h = OceanTest.OceanDL.Interpret.InterpretTest.hvae();
            x = rand(RandStream('mt19937ar', 'Seed', 2), 8, 8, 1);
            layer = "enc.L2.block1"; c = 3; r = 2; k = 3;
            g = OceanDL.Interpret.Probe.unitGradient(h, x, layer, c, r, k);
            f = @(xx) OceanDL.Interpret.Occlusion.pickUnit(OceanDL.Interpret.Probe.activation(h, xx, layer), c, r, k);
            d = 1e-6;
            for p = [5 20 37 60]
                e = zeros(size(x)); e(p) = d;
                tc.verifyEqual(g(p), (f(x + e) - f(x - e)) / (2 * d), 'AbsTol', 1e-6, sprintf('pixel %d', p));
            end
        end

        function testProjectionRulesOnOneLayer(tc)
            % conv is linear: its projection under ANY rule is the weight pattern;
            % deconvnet(act) passes max(1, 0) = 1 through SiLU -> same pattern;
            % guided(act) additionally multiplies by silu'(pre) (pre > 0) or 0.
            W = [1 -1 1; -1 2 -1; 1 -1 1];
            m = OceanTest.OceanDL.Interpret.TinyTraceable(W);
            x = rand(RandStream('mt19937ar', 'Seed', 9), 5, 5);
            pattern = zeros(5); pattern(2:4, 2:4) = W;
            P = @(layer, rule) OceanDL.Interpret.Probe.unitGradient(m, x, layer, 1, 3, 3, rule);
            for rule = ["standard", "deconvnet", "guided"]
                tc.verifyEqual(P("enc.L1.conv", rule), pattern, 'AbsTol', 1e-12, rule);
            end
            pre = OceanDL.Interpret.Probe.activation(m, x, "enc.L1.conv"); pre = pre(3, 3);
            s = 1 / (1 + exp(-pre)); dsilu = s + pre * s * (1 - s);
            tc.verifyEqual(P("enc.L1.act", "standard"), pattern * dsilu, 'AbsTol', 1e-12);
            tc.verifyEqual(P("enc.L1.act", "deconvnet"), pattern, 'AbsTol', 1e-12);
            tc.verifyEqual(P("enc.L1.act", "guided"), pattern * dsilu * (pre > 0), 'AbsTol', 1e-12);
        end

        function testTopActivationsAgainstBruteForce(tc)
            m = OceanTest.OceanDL.Interpret.TinyTraceable();
            X = rand(RandStream('mt19937ar', 'Seed', 3), 6, 6, 1, 23);
            T = OceanDL.Interpret.TopActivations.scan(m, X, "enc.L1.act", 'K', 4, 'Batch', 5);
            a = OceanDL.Interpret.Probe.activation(m, X, "enc.L1.act");
            best = squeeze(max(a, [], [1 2]));                              % per image
            [v, idx] = sort(best, 'descend');
            tc.verifyEqual(T.Value, v(1:4).', 'AbsTol', 1e-12);
            tc.verifyEqual(T.Image, idx(1:4).');
            for j = 1:4
                tc.verifyEqual(a(T.Row(j), T.Col(j), 1, T.Image(j)), T.Value(j), 'AbsTol', 1e-12);
            end
        end

        function testReceptiveFieldAndCrop(tc)
            m = OceanTest.OceanDL.Interpret.TinyTraceable(ones(3));
            X = rand(RandStream('mt19937ar', 'Seed', 4), 7, 7, 1, 2);
            rf = OceanDL.Interpret.ReceptiveField.effective(m, X, "enc.L1.conv");
            tc.verifyEqual(rf.Size, [3 3]);
            tc.verifyEqual(rf.Stride, [1 1]);
            x = reshape(1:49, 7, 7);
            P = OceanDL.Interpret.ReceptiveField.crop(x, rf, 1, 1);               % corner: zero-padded
            tc.verifyEqual(P, [0 0 0; 0 1 8; 0 2 9]);
            tc.verifyEqual(OceanDL.Interpret.ReceptiveField.crop(x, rf, 4, 4), x(3:5, 3:5));
        end

        function testFeatureVisFindsTheWeightPattern(tc)
            % maximising a LINEAR unit sum_p w_p x_p over x in (0,1): x -> 1 where w > 0, -> 0 where w < 0.
            % Default (1/frequency prior): correct signs; this checkerboard is high-frequency, so
            % the prior slows it down (by design). Without the prior: the exact optimum.
            W = [1 -1 1; -1 2 -1; 1 -1 1];
            m = OceanTest.OceanDL.Interpret.TinyTraceable(W);
            [x, info] = OceanDL.Interpret.FeatureVis.maximize(m, "enc.L1.conv", 1, 'InputSize', [7 7 1], ...
                'Steps', 150, 'Jitter', 0, 'LR', 0.1);
            patch = x(3:5, 3:5);
            tc.verifyTrue(all(patch(W > 0) > 0.5) && all(patch(W < 0) < 0.5), 'signs follow the weights');
            tc.verifyGreaterThan(info.History(end), 3 * info.History(1));
            tc.verifyEqual(info.Activation, sum(W .* patch, 'all'), 'AbsTol', 1e-9);
            [x, info] = OceanDL.Interpret.FeatureVis.maximize(m, "enc.L1.conv", 1, 'InputSize', [7 7 1], ...
                'Steps', 300, 'Jitter', 0, 'LR', 0.1, 'Decorrelate', false);
            patch = x(3:5, 3:5);
            tc.verifyTrue(all(patch(W > 0) > 0.95) && all(patch(W < 0) < 0.05), 'plain pixels: the exact optimum');
            tc.verifyGreaterThan(info.Activation, 0.95 * sum(W(W > 0)));
        end

        function testFeatureVisFromAGivenImage(tc)
            % 'Init': the start image is the given image (clipped to [0.02, 0.98]); optimising from it
            % raises the objective above the start value.
            W = [1 -1 1; -1 2 -1; 1 -1 1];
            m = OceanTest.OceanDL.Interpret.TinyTraceable(W);
            x0 = double(rand(RandStream('mt19937ar', 'Seed', 11), 7, 7) > 0.5);
            [x, info] = OceanDL.Interpret.FeatureVis.maximize(m, "enc.L1.conv", 1, 'InputSize', [7 7 1], ...
                'Init', x0, 'Steps', 0);
            tc.verifyEqual(x, min(max(x0, 0.02), 0.98), 'AbsTol', 1e-9, 'Steps 0 returns the start image');
            tc.verifyEqual(info.Activation, info.StartActivation, 'AbsTol', 1e-12);
            [~, info] = OceanDL.Interpret.FeatureVis.maximize(m, "enc.L1.conv", 1, 'InputSize', [7 7 1], ...
                'Init', x0, 'Steps', 100, 'Jitter', 0, 'LR', 0.1);
            tc.verifyGreaterThan(info.Activation, info.StartActivation);
        end

        function testWeightsForTracedLayers(tc)
            h = OceanTest.OceanDL.Interpret.InterpretTest.hvae();
            R = @OceanDL.Autograd.Tensor.raw;
            tc.verifyEqual(h.weightsFor("enc.L1.in"), R(h.ConvIn.Weight.Data));
            tc.verifyEqual(h.weightsFor("enc.L2.in"), R(h.Down{1}.Weight.Data));
            tc.verifyEqual(h.weightsFor("enc.L1.block1"), R(h.Up{1}{1}.Conv2.Weight.Data));
            tc.verifyEqual(h.weightsFor("dec.out"), R(h.ConvOut.Weight.Data));
            tc.verifyEqual(h.weightsFor("dec.L1.lift"), R(h.Lift{1}.Weight.Data));
            tc.verifyEmpty(h.weightsFor("z1.mu"));
            tc.verifyEmpty(h.weightsFor("dec.L3.start"));
            tc.verifyEqual(h.weightsFor("dec.out.act"), R(h.Res{1}{end}.Conv2.Weight.Data));
            tc.verifyEqual(h.weightsFor("dec.L2.td1"), R(h.TD{2}{1}.Block.Conv2.Weight.Data));
            % the weights really produce the traced layer: ConvIn output == conv of x
            X = rand(RandStream('mt19937ar', 'Seed', 12), 8, 8, 1, 1);
            acts = OceanDL.Autograd.GradMode.noGrad(@() h.trace(X, 'Parts', "encoder"));
            tc.verifyEqual(R(acts('enc.L1.in')), R(h.ConvIn.forward(X)), 'AbsTol', 1e-12);
        end

        function testOcclusionOfALinearUnit(tc)
            % 1x1 occluder with fill 0 on a linear unit: drop at pixel p = w_p x_p inside the field, 0 outside
            W = [1 -1 1; -1 2 -1; 1 -1 1];
            m = OceanTest.OceanDL.Interpret.TinyTraceable(W);
            x = rand(RandStream('mt19937ar', 'Seed', 5), 7, 7);
            M = OceanDL.Interpret.Occlusion.map(m, x, OceanDL.Interpret.Occlusion.unitTarget("enc.L1.conv", 1, 4, 4), ...
                "enc.L1.conv", 'Size', 1, 'Stride', 1, 'Fill', 0);
            want = zeros(7); want(3:5, 3:5) = W .* x(3:5, 3:5);
            tc.verifyEqual(M, want, 'AbsTol', 1e-12);
        end

        function testKlPerUnitSumsToTheLayerKl(tc)
            h = OceanTest.OceanDL.Interpret.InterpretTest.hvae();
            X = rand(RandStream('mt19937ar', 'Seed', 6), 8, 8, 1, 7);
            kl = OceanDL.Interpret.LatentStats.klPerUnit(h, X, 'Batch', 3);
            [~, kls] = OceanDL.Autograd.GradMode.noGrad(@() h.topDown(h.bottomUp(X), 7, [8 8], "mode"));
            for i = 1:numel(kl)
                tc.verifyEqual(sum(kl{i}, 'all'), sum(OceanDL.Autograd.Tensor.raw(kls{i}), 'all') / 7, 'AbsTol', 1e-10);
                tc.verifyGreaterThanOrEqual(min(kl{i}, [], 'all'), 0);
            end
            tc.verifyEqual(OceanDL.Interpret.LatentStats.numLayers(h, [8 8 1]), 2);
        end

        function testDecoderAtomsAreJacobianColumns(tc)
            h = OceanTest.OceanDL.Interpret.InterpretTest.hvae();
            X = rand(RandStream('mt19937ar', 'Seed', 7), 8, 8, 1, 5);
            z0 = OceanDL.Interpret.LatentStats.meanCodes(h, X);
            layer = 2; ch = 1;
            [A, info] = OceanDL.Interpret.DecoderAtoms.atoms(h, z0, layer, ch, [8 8], 'Delta', 1e-5);
            p = 3; pix = 29;                                                   % one unit, one output pixel
            zs = cellfun(@(z) OceanDL.Autograd.Tensor(z, 'RequiresGrad', true), z0, 'UniformOutput', false);
            out = h.mean(h.decodeLatents(zs, [8 8]));
            seed = zeros(8, 8); seed(pix) = 1;
            out.backward(seed);
            J = zs{layer}.Grad(info.Rows(p), info.Cols(p), ch);
            Ap = A(:, :, 1, p);
            tc.verifyEqual(Ap(pix), J, 'AbsTol', 1e-6);
        end

        function testLayerContributions(tc)
            h = OceanTest.OceanDL.Interpret.InterpretTest.hvae();
            x = rand(RandStream('mt19937ar', 'Seed', 8), 8, 8);
            [imgs, labels] = OceanDL.Interpret.DecoderAtoms.contributions(h, x);
            tc.verifyEqual(labels, ["x", "top 1 layer(s)", "all 2 layers"]);
            tc.verifyEqual(imgs(:, :, 1, 1), x);
            tc.verifyEqual(imgs(:, :, 1, 3), double(h.reconstruct(reshape(x, 8, 8, 1, 1))), 'AbsTol', 1e-12);
            tc.verifyTrue(all(isfinite(imgs(:))));
        end
        function testSparseDictionaryRecoversKnownAtoms(tc)
            % data built from a KNOWN dictionary (24 unit atoms in R^16, 2 active per sample):
            % the sparse autoencoder must find every atom (cosine > 0.9) and reconstruct well.
            rs = RandStream('mt19937ar', 'Seed', 1);
            C = 16; Mtrue = 24; n = 20000;
            Atom = randn(rs, Mtrue, C); Atom = Atom ./ vecnorm(Atom, 2, 2);
            Code = zeros(n, Mtrue);
            for i = 1:n, j = randperm(rs, Mtrue, 2); Code(i, j) = 1 + rand(rs, 1, 2); end
            A = Code * Atom;
            D = OceanDL.Interpret.SparseDictionary(C, 'Features', 48, 'L1', 0.05, 'Seed', 0);
            D.train(A, 'Steps', 1500, 'Batch', 512, 'LR', 3e-3);
            S = D.metrics(A);
            tc.verifyEqual(nnz(max(abs(Atom * D.Wd.'), [], 2) > 0.9), Mtrue, 'every true atom recovered');
            tc.verifyGreaterThan(S.ExplainedVariance, 0.9);
            tc.verifyEqual(vecnorm(D.Wd, 2, 2), ones(48, 1), 'AbsTol', 1e-9);
            tc.verifyLessThan(mean(D.History(end-99:end)), mean(D.History(1:100)));
            % featureMaps == encode of the position vectors
            a = rand(rs, 3, 4, C, 2);
            Fm = D.featureMaps(a);
            tc.verifyEqual(squeeze(Fm(2, 3, :, 2)).', D.encode(squeeze(a(2, 3, :, 2)).'), 'AbsTol', 1e-12);
        end

        function testLatentCodesMatchEncode(tc)
            h = OceanTest.OceanDL.Interpret.InterpretTest.hvae();
            X = rand(RandStream('mt19937ar', 'Seed', 10), 8, 8, 1, 5);
            Z = OceanDL.Interpret.LatentStats.codes(h, X, [], 'Batch', 2);
            qs = h.encode(X);
            want = [reshape(OceanDL.Autograd.Tensor.raw(qs{1}.Mu), [], 5).', reshape(OceanDL.Autograd.Tensor.raw(qs{2}.Mu), [], 5).'];
            tc.verifyEqual(Z, want, 'AbsTol', 1e-12);
            tc.verifyEqual(OceanDL.Interpret.LatentStats.codes(h, X, 2), want(:, 9:end), 'AbsTol', 1e-12);
        end
    end

    methods (Static)
        function h = hvae()
            h = OceanDL.Generative.HierarchicalVAE('Widths', [8 8 8], 'LatentsPerLevel', [0 1 1], ...
                'LatentChannels', 2, 'Groups', 4, 'Seed', 3);
        end
    end
end
