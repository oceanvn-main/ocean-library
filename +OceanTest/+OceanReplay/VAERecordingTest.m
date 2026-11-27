classdef VAERecordingTest < OceanTest.BaseTestCase
    % VAERECORDINGTEST  TraceableRecordable + VAERecordable (Kingma & Welling 2019 Ch. 2):
    %   recording changes nothing; the stored numbers satisfy -ELBO = recon + sum_i KL_i,
    %   KL_i = sum of the per-unit KL; xhat = the model's reconstruction; mu = encode's
    %   posterior means (top layer); every traced activation equals trace(); the
    %   Replay Studio picks the VAE layout.

    properties
        Dir
    end

    methods (TestMethodSetup)
        function makeDir(tc)
            tc.Dir = tc.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder;
        end
    end

    methods (Test)
        function testInvarianceAndElbo(tc)
            [m, X, ids] = OceanTest.OceanReplay.VAERecordingTest.fixture();
            make = @() OceanDL.Generative.Inspect.VAERecordable(m, X, 'Ids', ids, 'Labels', [3 1 4], 'Seed', 2);
            r0 = make(); [c, ~] = r0.catalog();
            spec = OceanReplay.Core.Spec().addGroup("all", 0, [c.Name]); spec.Precision = "float64";
            f = fullfile(tc.Dir, "vae.h5");
            [ok, rep] = OceanReplay.Record.verifyInvariance(make, spec, 'File', f);
            tc.verifyTrue(ok);
            tc.verifyEmpty(OceanReplay.IO.Schema.validate(f));
            rec = OceanReplay.Core.Recording(f);
            tc.verifyEqual(rec.axis("image").Values, sort(ids));                % the axis is monotone
            L = r0.NumLatent;
            tc.verifyEqual(L, m.numLatentLayers());
            [qs] = m.encode(X);
            xh = double(m.reconstruct(X));
            for k = 1:numel(ids)
                at = struct('image', ids(k));
                g = @(n) double(rec.get(n, at, "exact"));
                kls = 0;
                for i = 1:L
                    tc.verifyEqual(g(sprintf("kl_%d", i)), sum(g(sprintf("klUnit_%d", i)), 'all'), 'AbsTol', 1e-10);
                    kls = kls + g(sprintf("kl_%d", i));
                end
                tc.verifyEqual(g("kl"), kls, 'AbsTol', 1e-10);
                tc.verifyEqual(g("negElbo"), g("recon") + g("kl"), 'AbsTol', 1e-10);
                tc.verifyEqual(g("negElbo"), rep.Without(sort(ids) == ids(k)), 'AbsTol', 1e-10);
                tc.verifyEqual(g("bitsPerDim"), g("negElbo") / (numel(X(:, :, :, 1)) * log(2)), 'AbsTol', 1e-12);
                tc.verifyEqual(g("xhat"), xh(:, :, :, k), 'AbsTol', 1e-5);
                mu1 = double(OceanDL.Autograd.Tensor.raw(qs{1}.Mu));   % top layer: depends on x only
                tc.verifyEqual(g("mu_1"), mu1(:, :, :, k), 'AbsTol', 1e-5);
                tc.verifyGreaterThanOrEqual(min(g("sigma_1"), [], 'all'), 0);
                tc.verifyEqual(g("label"), double(k == 1) * 3 + double(k == 2) * 1 + double(k == 3) * 4);
            end
        end

        function testActivationsEqualTrace(tc)
            [m, X, ids] = OceanTest.OceanReplay.VAERecordingTest.fixture();
            r = OceanDL.Interpret.TraceableRecordable(m, X, 'Ids', ids);
            [c, ~] = r.catalog();
            spec = OceanReplay.Core.Spec().addGroup("all", 0, [c.Name]); spec.Precision = "float64";
            f = fullfile(tc.Dir, "acts.h5");
            OceanReplay.Record.Recorder.run(r, spec, f, 'Overwrite', true);
            rec = OceanReplay.Core.Recording(f);
            tc.verifyNotEmpty(r.Layers);
            for k = 1:numel(ids)
                acts = OceanDL.Autograd.GradMode.noGrad(@() m.trace(X(:, :, :, k)));
                for Lname = r.Layers
                    tc.verifyEqual(double(rec.get(r.actName(Lname), struct('image', ids(k)), "exact")), ...
                        double(OceanDL.Autograd.Tensor.raw(acts(char(Lname)))), 'AbsTol', 1e-12);
                end
            end
        end

        function testSameNumbersWhateverIsRecorded(tc)
            % eps comes from a per-image substream: recording fewer channels changes no value.
            [m, X, ids] = OceanTest.OceanReplay.VAERecordingTest.fixture();
            r = OceanDL.Generative.Inspect.VAERecordable(m, X, 'Ids', ids, 'Seed', 4);
            [c, ~] = r.catalog();
            fa = fullfile(tc.Dir, "all.h5"); fb = fullfile(tc.Dir, "few.h5");
            sa = OceanReplay.Core.Spec().addGroup("all", 0, [c.Name]); sa.Precision = "float64";
            sb = OceanReplay.Core.Spec().addGroup("few", 0, ["negElbo" "z_1"]); sb.Precision = "float64";
            OceanReplay.Record.Recorder.run(r, sa, fa, 'Overwrite', true);
            OceanReplay.Record.Recorder.run(OceanDL.Generative.Inspect.VAERecordable(m, X, 'Ids', ids, 'Seed', 4), sb, fb, 'Overwrite', true);
            a = OceanReplay.Core.Recording(fa); b = OceanReplay.Core.Recording(fb);
            for id = ids
                at = struct('image', id);
                tc.verifyEqual(b.get("negElbo", at, "exact"), a.get("negElbo", at, "exact"));
                tc.verifyEqual(b.get("z_1", at, "exact"), a.get("z_1", at, "exact"));
            end
        end

        function testStudioUsesVaeLayout(tc)
            [m, X, ids] = OceanTest.OceanReplay.VAERecordingTest.fixture();
            r = OceanDL.Generative.Inspect.VAERecordable(m, X, 'Ids', ids);
            [c, ~] = r.catalog();
            f = fullfile(tc.Dir, "studio.h5");
            OceanReplay.Record.Recorder.run(r, OceanReplay.Core.Spec().addGroup("all", 0, [c.Name]), f, 'Overwrite', true);
            s = OceanViz.App.Replay.ReplayStudio(f, 'Visible', false, 'Remember', false);
            cleanup = onCleanup(@() s.close());
            tc.verifyEqual(s.Layout.name(), "VAE");
            s.stepForward(); drawnow;
            ids = sort(ids); tc.verifyEqual(s.Coords.image, ids(2));
        end
    end

    methods (Static)
        function [m, X, ids] = fixture()
            m = OceanTest.OceanDL.Interpret.InterpretTest.hvae();
            X = double(rand(RandStream('mt19937ar', 'Seed', 9), [8 8 1 3]) < 0.4);
            ids = [11 5 42];
        end
    end
end
