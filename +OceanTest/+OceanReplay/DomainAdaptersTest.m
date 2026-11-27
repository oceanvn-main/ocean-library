classdef DomainAdaptersTest < OceanTest.BaseTestCase
    % DOMAINADAPTERSTEST  The three G2 domains (Master plan Sec. 5.5):
    %   * diffusion: ReverseProcess 'OnStep' changes nothing; the recording
    %     equals the process's frames, satisfies x_i = alphaBar x_phi + sigma eps_phi,
    %     and recording (with the occlusion view) does not change x_0;
    %   * k-means: 'OnIteration' changes nothing; centroids = KMeans.History,
    %     centroids = cluster means of the assignment, sse never increases;
    %   * legacy NN checkpoint: prediction = the NNReplayEngine renderer's output.

    properties
        Dir
    end

    methods (TestMethodSetup)
        function makeDir(tc)
            tc.Dir = tc.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder;
        end
    end

    methods (Test)
        function testOnStepChangesNothing(tc)
            [pred, sch, smp] = OceanTest.OceanReplay.DomainAdaptersTest.tinyDiffusion();
            X1 = randn(RandStream('mt19937ar', 'Seed', 2), 1, 9);
            a = OceanDL.Generative.Inspect.ReverseProcess(pred, sch, smp, [1 9 1 1], 'X1', X1, ...
                'Stream', RandStream('mt19937ar', 'Seed', 3), 'Stride', 1);
            a.advance();
            seen = struct('Step', {}, 'T', {}, 'TNext', {}, 'X', {}, 'X0', {}, 'Eps', {}, 'Mu', {}, 'Var', {});
            function collect(f), seen(end+1) = f; end
            b = OceanDL.Generative.Inspect.ReverseProcess(pred, sch, smp, [1 9 1 1], 'X1', X1, ...
                'Stream', RandStream('mt19937ar', 'Seed', 3), 'Stride', 50, 'OnStep', @collect);
            b.advance();
            tc.verifyEqual(b.X, a.X);
            tc.verifyEqual(seen, a.Frames);                              % every step reported, same content
        end

        function testDiffusionRecording(tc)
            [pred, sch, smp] = OceanTest.OceanReplay.DomainAdaptersTest.tinyDiffusion();
            S = OceanReplay.Core.Sampling;
            spec = OceanReplay.Core.Spec() ...
                .addGroup("numbers", 0, ["alphaBar" "sigma" "varNext" "identityResidual"]) ...
                .addGroup("images", 1, ["x" "eps" "x0hat" "mu"], struct('step', S.every(10))) ...
                .addGroup("final", 1, "result") ...
                .addView("occlusion", "occlusion", struct('size', 1, 'stride', 1), struct('step', S.list(20)));
            spec.Precision = "float64";
            make = @() OceanDL.Generative.Inspect.DiffusionRecordable(pred, sch, smp, [1 9 1], 'Seed', 5, 'DataRange', [-3 3]);
            f = fullfile(tc.Dir, "ddpm.h5");
            [ok, rep] = OceanReplay.Record.verifyInvariance(make, spec, 'File', f);
            tc.verifyTrue(ok);
            tc.verifyEmpty(OceanReplay.IO.Schema.validate(f));
            r = OceanReplay.Core.Recording(f);
            tc.verifyEqual(r.axis("step").Values, 50:-1:1);              % book index i = t_i L, L = 50
            tc.verifyEqual(r.stored("x").Values.', [50 40 30 20 10]);
            % the recording equals the process's own frames
            rp = OceanDL.Generative.Inspect.ReverseProcess(pred, sch, smp, [1 9 1 1], ...
                'Stream', RandStream('mt19937ar', 'Seed', 5), 'Stride', 1);
            rp.advance();
            for i = [50 40 30 20 10]
                fr = rp.Frames(51 - i);
                tc.verifyEqual(r.get("x", struct('step', i), "exact"), fr.X, 'AbsTol', 1e-15);
                tc.verifyEqual(r.get("mu", struct('step', i), "exact"), fr.Mu, 'AbsTol', 1e-15);
                a = r.get("alphaBar", struct('step', i)); s = r.get("sigma", struct('step', i));
                tc.verifyEqual(r.get("x", struct('step', i)), a * r.get("x0hat", struct('step', i)) + s * r.get("eps", struct('step', i)), 'AbsTol', 1e-10);
            end
            [~, resid] = r.series("identityResidual");
            tc.verifyLessThan(max(resid), 1e-10);
            tc.verifyEqual(r.get("result"), rep.With.X, 'AbsTol', 1e-15);
            occ = r.get("occlusion", struct('step', 20), "exact");
            tc.verifyEqual(max(occ(:)), 1, 'AbsTol', 1e-12);             % normalised map
            tc.verifyEqual(string(r.Meta.layout), "diffusion");
        end

        function testKMeansRecording(tc)
            X = OceanTest.OceanReplay.DomainAdaptersTest.blobs();
            make = @() OceanStats.Unsupervised.KMeansRecordable(X, 3, 'Seed', 4);
            spec = OceanReplay.Core.Spec().addGroup("all", 0, ["data" "centroids" "assignment" "sse" "changed"]);
            spec.Precision = "float64";
            f = fullfile(tc.Dir, "km.h5");
            [ok, rep] = OceanReplay.Record.verifyInvariance(make, spec, 'File', f);
            tc.verifyTrue(ok);
            tc.verifyEmpty(OceanReplay.IO.Schema.validate(f));
            r = OceanReplay.Core.Recording(f);
            m = r.axis("iteration").Values;
            tc.verifyEqual(m, 0:numel(m) - 1);
            % same centroids as KMeans's own trajectory for the same start
            d = make(); km = OceanStats.Unsupervised.KMeans();
            km.fit(X, 3, 'InitCentroids', d.InitCentroids, 'TrackHistory', true);
            for k = 1:min(numel(km.History), numel(m))
                tc.verifyEqual(r.get("centroids", k - 1, "exact"), km.History(k).Centroids, 'AbsTol', 1e-12);
            end
            tc.verifyEqual(r.get("centroids", m(end)), rep.With.Centroids, 'AbsTol', 1e-5);
            % centroids are the cluster means of the recorded assignment (m >= 1)
            for it = m(2:end)
                L = double(r.get("assignment", it)); C = double(r.get("centroids", it));
                for k = unique(L).'
                    tc.verifyEqual(C(k, :), mean(X(L == k, :), 1), 'AbsTol', 1e-5);
                end
            end
            [~, sse] = r.series("sse");
            tc.verifyTrue(all(diff(sse) <= 1e-3 * max(sse)), 'sse increased');
            tc.verifyEqual(r.info("assignment").Categories, "cluster " + string(1:3));
            tc.verifyEqual(r.info("data").LabelsChannel, "assignment");
        end

        function testOnIterationChangesNothing(tc)
            X = OceanTest.OceanReplay.DomainAdaptersTest.blobs();
            C0 = X([1 50 100], :);
            a = OceanStats.Unsupervised.KMeans(); la = a.fit(X, 3, 'InitCentroids', C0);
            n = 0;
            function count(~), n = n + 1; end
            b = OceanStats.Unsupervised.KMeans(); lb = b.fit(X, 3, 'InitCentroids', C0, 'OnIteration', @count);
            tc.verifyEqual(lb, la); tc.verifyEqual(b.Centroids, a.Centroids);
            tc.verifyGreaterThan(n, 1);
        end

        function testNNCheckpointConverterMatchesRenderer(tc)
            main = OceanTest.OceanReplay.DomainAdaptersTest.tinyCheckpoint(tc.Dir);
            d = OceanViz.Visualizers.NNCheckpointRecordable(main);
            S = OceanReplay.Core.Sampling;
            spec = OceanReplay.Core.Spec().addGroup("curves", 0, ["cost" "rSquared"]) ...
                .addGroup("maps", 1, ["prediction" "weights"], struct('epoch', S.list([1 3]))) ...
                .addGroup("target", 0, "groundTruth");
            spec.Precision = "float64";
            f = fullfile(tc.Dir, "nn.h5");
            OceanReplay.Record.Recorder.run(d, spec, f);
            tc.verifyEmpty(OceanReplay.IO.Schema.validate(f));
            r = OceanReplay.Core.Recording(f);
            ck = load(main); w = load(strrep(main, ".mat", "_weights.mat")); ck.weightHistory = w.weightHistory;
            fig = figure('Visible', 'off'); c = onCleanup(@() close(fig)); ax = axes(fig);
            ren = OceanViz.Strategies.DecisionBoundaryRenderer();
            ren.setup(ax, ck, OceanViz.Core('Theme', 'Dark').Colors);
            for ep = [1 3]
                ren.render(ax, ck, struct(), ep);
                cdata = findobj(ax, 'Type', 'image').CData;
                tc.verifyEqual(r.get("prediction", ep, "exact"), cdata, 'AbsTol', 1e-12);
                tc.verifyEqual(r.get("weights", ep, "exact"), ck.weightHistory{ep}, 'AbsTol', 0);
            end
            [x, y] = r.series("cost");
            tc.verifyEqual(x.', 1:3); tc.verifyEqual(y.', ck.costHistory, 'AbsTol', 1e-12);
            tc.verifyEqual(r.get("groundTruth"), ck.groundTruth, 'AbsTol', 1e-12);
        end
    end

    methods (Static)
        function [pred, sch, smp] = tinyDiffusion()
            import OceanDL.Generative.Diffusion.*
            sch = Schedules.LinearVP('L', 50);
            pred = Analytic.gmm([-2 2], [0.3 0.3], [0.5 0.5], sch);       % exact denoiser of a 1-D mixture, per pixel (row data: 1 x 9 "images")
            smp = Samplers.DDPMAncestral('Steps', 50);
        end

        function X = blobs()
            s = RandStream('mt19937ar', 'Seed', 11);
            X = [randn(s, 50, 2) * 0.4 + [0 0]; randn(s, 50, 2) * 0.4 + [3 0]; randn(s, 50, 2) * 0.4 + [1.5 2.5]];
        end

        function main = tinyCheckpoint(dir)
            % A 2-3-1 tanh/sigmoid network with 3 epochs of random weights, split like the Mandelbrot run.
            s = RandStream('mt19937ar', 'Seed', 21);
            layerSizes = [2 3 1]; numEpochs = 3; DISPLAY_RES = 4;
            [a, b] = meshgrid(linspace(-1, 1, 4));
            gridNorm = [a(:) b(:)].'; a_d = linspace(-2, 1, 4); b_d = linspace(-1.5, 1.5, 4);
            costHistory = [0.9 0.5 0.2]; accHistory = [0.1 0.5 0.8];
            groundTruth = rand(s, 4, 4); activationFns = {'linear', 'tanh', 'sigmoid'};
            weightHistory = arrayfun(@(~) randn(s, 13, 1), (1:numEpochs).', 'UniformOutput', false);
            main = string(fullfile(dir, 'tiny_checkpoint.mat'));
            save(main, 'layerSizes', 'numEpochs', 'DISPLAY_RES', 'gridNorm', 'a_d', 'b_d', 'costHistory', ...
                'accHistory', 'groundTruth', 'activationFns');
            save(strrep(main, ".mat", "_weights.mat"), 'weightHistory', '-v7.3');
        end
    end
end
