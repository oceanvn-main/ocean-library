classdef ReplayStudioTest < OceanTest.BaseTestCase
    % REPLAYSTUDIOTEST  Replay Studio opens recordings of every G2 domain
    %   (headless), picks the domain layout, draws every tile, labels each tile
    %   with the ACTUAL stored coordinates, warns for views not in the texts,
    %   and the Default layout works for any recording (Master plan Sec. 5.6).

    properties
        Dir
    end

    methods (TestMethodSetup)
        function makeDir(tc)
            tc.Dir = tc.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder;
        end
    end

    methods (Test)
        function testDiffusionRecording(tc)
            f = tc.recordDiffusion();
            s = OceanViz.App.Replay.ReplayStudio(f, 'Visible', false, 'Remember', false); c = onCleanup(@() s.close());
            tc.verifyEqual(s.Layout.name(), "Diffusion");                 % from the recording's layout hint
            tc.verifyEqual(s.Coords.step, 50);                            % first point of the run
            s.setCoord("step", 33);                                       % images stored every 10
            caps = OceanTest.OceanViz.ReplayStudioTest.captions(s);
            tc.verifyTrue(any(contains(caps, "stored at step 30  ·  nearest stored to 33")), strjoin(caps, " | "));
            tc.verifyFalse(any(contains(caps, "could not draw")), strjoin(caps, " | "));
            % notation always visible; long text hidden until asked (drawer closed)
            tc.verifyEqual(string(s.Tiles(1).Notation.Text), "$x_i$");            % LaTeX notation
            notes = arrayfun(@(t) string(t.Note.Text), s.Tiles);
            tc.verifyEqual(nnz(strlength(notes) > 0), 0);                          % no note lines on the cards
            s.showDetails(true);
            info = string(s.DrawerText.Value);
            tc.verifyTrue(any(contains(info, "(Eq. 2.2.5)")));
            tc.verifyTrue(any(contains(info, "HOW IT WAS MADE")));
            s.showTileInfo(numel(s.Tiles));                               % (i) of the occlusion tile
            info = string(s.DrawerText.Value);
            tc.verifyTrue(any(contains(info, "WARNING -- NOT IN THE TEXTS")));
            tc.verifyFalse(any(contains(info, "(Eq. 2.2.5)")));            % only that tile
            % the Eq. 2.2.4 numbers in the full details come from the file at the stored step
            s.showDetails(true);
            r = OceanReplay.Core.Recording(f);
            tc.verifyTrue(any(contains(string(s.DrawerText.Value), sprintf('alphaBar_i = %.4f', r.get("alphaBar", struct('step', 33))))));
            s.showDetails(false);
            s.setLayout("Default");
            tc.verifyEqual(numel(s.Tiles), 6);                            % x eps x0hat mu result + occlusion
            tc.verifyFalse(any(contains(OceanTest.OceanViz.ReplayStudioTest.captions(s), "could not draw")));
        end

        function testKMeansRecording(tc)
            X = OceanTest.OceanReplay.DomainAdaptersTest.blobs();
            f = fullfile(tc.Dir, "km.h5");
            OceanReplay.Record.Recorder.run(OceanStats.Unsupervised.KMeansRecordable(X, 3, 'Seed', 4), ...
                OceanReplay.Core.Spec().addGroup("all", 0, ["data" "centroids" "assignment" "sse" "changed"]), f);
            s = OceanViz.App.Replay.ReplayStudio(f, 'Visible', false, 'Remember', false); c = onCleanup(@() s.close());
            tc.verifyEqual(s.Layout.name(), "Clustering");
            last = max(s.Recording.axis("iteration").Values);
            s.setCoord("iteration", last);
            caps = OceanTest.OceanViz.ReplayStudioTest.captions(s);
            tc.verifyTrue(any(contains(caps, sprintf("iteration m = %g", last))), strjoin(caps, " | "));
            s.showDetails(true);
            tc.verifyTrue(any(contains(string(s.DrawerText.Value), "Algorithm 14.1")));
            tc.verifyEqual(sort(string(s.TimelineDropdown.Items)), ["changed" "sse"]);
            tc.verifyEqual(string(s.TimelineDropdown.Value), "sse");       % the layout's choice
        end

        function testConvertedCheckpointAndToy(tc)
            main = OceanTest.OceanReplay.DomainAdaptersTest.tinyCheckpoint(tc.Dir);
            f = fullfile(tc.Dir, "nn.h5");
            S = OceanReplay.Core.Sampling;
            OceanReplay.Record.Recorder.run(OceanViz.Visualizers.NNCheckpointRecordable(main), ...
                OceanReplay.Core.Spec().addGroup("c", 0, ["cost" "rSquared" "groundTruth"]) ...
                .addGroup("m", 1, "prediction", struct('epoch', S.list([1 3]))), f);
            s = OceanViz.App.Replay.ReplayStudio(f, 'Visible', false, 'Remember', false); c = onCleanup(@() s.close());
            tc.verifyEqual(s.Layout.name(), "Default");
            s.setCoord("epoch", 2);
            caps = OceanTest.OceanViz.ReplayStudioTest.captions(s);
            tc.verifyTrue(any(contains(caps, "stored at epoch 1  ·  nearest stored to 2")) || ...
                          any(contains(caps, "stored at epoch 3  ·  nearest stored to 2")), strjoin(caps, " | "));
            % a recording with every kind (ToyRecordable) opens with the Default layout
            g = fullfile(tc.Dir, "toy.h5");
            OceanReplay.Record.Recorder.run(OceanTest.OceanReplay.ToyRecordable(), OceanTest.OceanReplay.ToyRecordable.fullSpec(), g);
            s2 = OceanViz.App.Replay.ReplayStudio(g, 'Visible', false, 'Remember', false); c2 = onCleanup(@() s2.close());
            s2.setCoord("k", 2); s2.setCoord("t", 5);
            caps = OceanTest.OceanViz.ReplayStudioTest.captions(s2);
            tc.verifyNumElements(caps, 8);                                % vec mat img seq pts tok label + view
            tc.verifyFalse(any(contains(caps, "could not draw")), strjoin(caps, " | "));
        end

        function testPlayerStepsThroughStoredPoints(tc)
            f = tc.recordDiffusion();                                     % images at steps 50, 40, 30, 20, 10
            s = OceanViz.App.Replay.ReplayStudio(f, 'Visible', false, 'Remember', false); c = onCleanup(@() s.close());
            tc.verifyEqual(s.PlayAxis, "step");
            tc.verifyEqual(s.Frames, [50 40 30 20 10]);
            tc.verifyTrue(s.stepForward());  tc.verifyEqual(s.Coords.step, 40);
            tc.verifyTrue(s.stepBack());     tc.verifyEqual(s.Coords.step, 50);
            s.setCoord("step", 33);                                       % between frames
            s.stepForward();                 tc.verifyEqual(s.Coords.step, 30);    % counting down from 33
            s.last();                        tc.verifyFalse(s.stepForward());   % at the end
            s.first();
            s.play(); tc.verifyTrue(s.Playing);
            t0 = tic;
            while s.Playing && toc(t0) < 20, pause(0.2); end               % 5 fps: the timer runs to the end
            tc.verifyFalse(s.Playing);                                    % stopped by itself at the end
            tc.verifyEqual(s.Coords.step, 10);
            tc.verifyEqual(string(s.PlayButton.Text), "▶  Play");
        end

        function testOpenAnotherRecordingAndRecentList(tc)
            % Toolbar Open... / Recent: a new recording replaces the view in the SAME window,
            % and opened files are remembered (the user's preference is saved and restored).
            had = ispref('OceanReplay', 'RecentFiles'); old = {};
            if had, old = getpref('OceanReplay', 'RecentFiles'); rmpref('OceanReplay', 'RecentFiles'); end
            restore = onCleanup(@() OceanTest.OceanViz.ReplayStudioTest.restorePref(had, old)); %#ok<NASGU>
            f1 = tc.recordDiffusion();
            X = OceanTest.OceanReplay.DomainAdaptersTest.blobs();
            f2 = fullfile(tc.Dir, "km.h5");
            OceanReplay.Record.Recorder.run(OceanStats.Unsupervised.KMeansRecordable(X, 3, 'Seed', 4), ...
                OceanReplay.Core.Spec().addGroup("all", 0, ["data" "centroids" "assignment" "sse" "changed"]), f2);
            s = OceanViz.App.Replay.ReplayStudio(f1, 'Visible', false); c = onCleanup(@() s.close());
            fig = s.Figure;
            tc.verifyEqual(s.Layout.name(), "Diffusion");
            s.loadFile(f2);
            tc.verifyTrue(s.Figure == fig);                                  % same window
            tc.verifyEqual(s.Layout.name(), "Clustering");
            tc.verifyTrue(contains(string(s.Figure.Name), "k-means") || contains(string(s.Figure.Name), "KMeans"));
            tc.verifyTrue(isfield(s.Coords, 'iteration') && ~isfield(s.Coords, 'step'));
            r = s.recentFiles();
            tc.verifyNumElements(r, 2);
            [~, n1] = fileparts(r(1)); tc.verifyEqual(string(n1), "km");    % most recent first
            tc.verifyEqual(string(s.RecentDropdown.Items{1}), "Recent recordings...");
            tc.verifyError(@() s.loadFile(fullfile(tc.Dir, "missing.h5")), 'OceanViz:Replay:noFile');
            tc.verifyEqual(s.Layout.name(), "Clustering");                  % a bad file keeps the current view
        end

        function testOnlyTheRegistryMapsKindsToDrawing(tc)
            root = fullfile(tc.LibraryRoot, '+OceanViz', '+App', '+Replay');
            files = dir(fullfile(root, '**', '*.m'));
            for i = 1:numel(files)
                if contains(files(i).folder, '+Renderers'), continue; end
                txt = fileread(fullfile(files(i).folder, files(i).name));
                tc.verifyEmpty(regexp(txt, 'switch\s+\w+\.Kind', 'once'), files(i).name);
            end
        end
    end

    methods (Static)
        function restorePref(had, old)
            if ispref('OceanReplay', 'RecentFiles'), rmpref('OceanReplay', 'RecentFiles'); end
            if had, setpref('OceanReplay', 'RecentFiles', old); end
        end

        function caps = captions(s)
            caps = arrayfun(@(t) string(t.Caption.Text), s.Tiles);
        end
    end

    methods (Access = private)
        function f = recordDiffusion(tc)
            [pred, sch, smp] = OceanTest.OceanReplay.DomainAdaptersTest.tinyDiffusion();
            S = OceanReplay.Core.Sampling;
            spec = OceanReplay.Core.Spec() ...
                .addGroup("numbers", 0, ["tLevel" "tNext" "alphaBar" "alphaBarNext" "sigma" "varNext" "identityResidual" "epsRms"]) ...
                .addGroup("images", 1, ["x" "eps" "x0hat" "mu"], struct('step', S.every(10))) ...
                .addGroup("final", 1, "result") ...
                .addView("occlusion", "occlusion", struct('size', 1, 'stride', 1), struct('step', S.list([30 20])));
            f = fullfile(tc.Dir, "ddpm.h5");
            OceanReplay.Record.Recorder.run(OceanDL.Generative.Inspect.DiffusionRecordable(pred, sch, smp, [1 9 1], ...
                'Seed', 5, 'DataRange', [-3 3]), spec, f);
        end
    end
end
