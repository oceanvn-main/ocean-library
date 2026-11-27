classdef StudioAppTest < matlab.unittest.TestCase
    % STUDIOAPPTEST  Headless (Visible false) Speech Pipeline Studio runs on real audio:
    %   FSDD with the CTC and the encoder-decoder checkpoints, a 44.1 kHz Audio Toolbox WAV
    %   (resampled + segmented), and injected microphone samples. Every stage must render
    %   without error and report the real values of the run; Workflow tabs must build from
    %   saved data; close() must clean up.

    properties
        App
        Samples
        SampleRate
        MicId
    end

    methods (TestClassSetup)
        function launchApp(tc)
            m = OceanNLP.Speech.Data.SpeechManifest.fromFSDD().canonicalFSDDSplit();
            i = find(m.Entries.Split == "test", 1, 'last');
            [tc.Samples, tc.SampleRate] = audioread(m.resolve(i));
            tc.MicId = m.Entries.Id(i);
            rec = OceanViz.App.NLP.SpeechStudio.Models.InjectedRecorder(tc.Samples, tc.SampleRate, 'Name', tc.MicId);
            tc.App = OceanViz.App.NLP.SpeechStudio.launch('Visible', false, 'Recorder', rec, 'Manifest', m);
            tc.addTeardown(@() tc.closeIfValid());
        end
    end

    methods (TestMethodSetup)
        function resetState(tc)
            % Every test starts from the same state, independent of test order.
            vm = tc.App.ViewModel;
            vm.setMode("Pipeline");
            vm.selectModel("ctc_fsdd_best");
            vm.resetSettings();
            vm.setSplit("test");
            vm.selectUtterance(vm.Utterances(1));
            vm.WavFile = "";       % no remembered WAV: switching sources restores remembered input (N2)
            vm.setCursor(NaN);
        end
    end

    methods
        function closeIfValid(tc)
            if ~isempty(tc.App) && isvalid(tc.App), tc.App.close(); end
        end

        function verifyAllStages(tc, expectFail)
            if nargin < 2, expectFail = strings(1, 0); end
            vm = tc.App.ViewModel;
            for k = 1:9
                if ~vm.StageEnabled(k), continue; end
                vm.selectStage(k);
                s = tc.App.stage(k);
                tc.verifyEqual(s.LastError, "", sprintf('Stage %d (%s) failed to draw', k, s.title()));
                tc.verifyNotEmpty(s.Rows, sprintf('Stage %d shows no values', k));
                tc.verifyNotEmpty(s.Classes, sprintf('Stage %d has no provenance', k));
                tc.verifyNotEmpty(s.shownNotation(), sprintf('Stage %d has no notation', k));
                tc.verifyEqual(char(s.Root.Visible), 'on');
            end
        end

        function v = rowValue(~, stage, quantity)
            r = stage.Rows;
            v = r(r(:, 1) == quantity, 2);
        end
    end

    methods (Test)
        function fsddCTCEveryStage(tc)
            vm = tc.App.ViewModel;
            vm.selectModel("ctc_fsdd_best");
            vm.setSplit("test");
            vm.selectUtterance(vm.Utterances(1));
            tc.verifyTrue(tc.App.run());
            tc.verifyNotEmpty(tc.App.ProgressLog);
            tc.verifyTrue(all(vm.StageEnabled));
            tc.verifyAllStages();
            ctx = vm.Context;
            % stage values equal the recorded artifacts / result
            P = ctx.artifact("Posteriors");
            tc.verifyEqual(tc.rowValue(tc.App.stage(7), "Posteriors (K x T)"), ...
                strjoin(string(size(P.Posteriors)), " x "));
            F = ctx.artifact("Frames + window");
            tc.verifyEqual(tc.rowValue(tc.App.stage(2), "Frames (L x T)"), strjoin(string(size(F.Frames)), " x "));
            tc.verifySubstring(tc.rowValue(tc.App.stage(9), "Transcript"), ctx.Result.Transcript.Text);
            tc.verifyEqual(tc.rowValue(tc.App.stage(1), "InputRate"), string(ctx.Result.Report.InputRate));
            tc.verifyTrue(all(isfinite(ctx.Result.Report.RMSdBFS)));
            % the decoded transcript equals the recogniser's own transcribe on the same audio
            tr = ctx.Recognizer.transcribe(ctx.segmentWave(), 'Decoder', "beam", 'BeamWidth', ctx.Settings.BeamWidth);
            tc.verifyEqual(ctx.segmentTranscript().Text, tr.Text);
        end

        function linkedCursorAndDrawer(tc)
            vm = tc.App.ViewModel;
            vm.selectModel("ctc_fsdd_best");
            tc.App.run();
            vm.selectStage(5);
            s = tc.App.stage(5);
            tc.verifyGreaterThanOrEqual(numel(s.Cursors), 3, 'Features stage has linked time axes');
            tc.verifyEqual(numel(s.Cursors), numel(s.TimeAxes), 'One cursor per time axis');
            for i = 1:numel(s.Cursors)
                tc.verifyTrue(s.Cursors{i}.Parent == s.TimeAxes{i}, 'Cursor lives on its time axis');
            end
            vm.setCursor(0.1);
            vals = cellfun(@(c) c.Value, s.Cursors);
            tc.verifyEqual(vals, 0.1 + s.TimeOffsets, 'AbsTol', 1e-12, ...
                'ViewModel.CursorTime propagates to every time axis of the stage');
            vm.setCursor(0.2);
            vals = cellfun(@(c) c.Value, s.Cursors);
            tc.verifyEqual(vals, 0.2 + s.TimeOffsets, 'AbsTol', 1e-12);
            tc.verifyTrue(all(cellfun(@(c) strcmp(c.Visible, 'on'), s.Cursors)));
            vm.setCursor(NaN);
            tc.verifyTrue(all(cellfun(@(c) strcmp(c.Visible, 'off'), s.Cursors)));
            tc.verifyFalse(s.drawerOpen());
            s.toggleDrawer();
            tc.verifyTrue(s.drawerOpen());
            s.toggleDrawer();
            % frame selection by click time
            vm.selectStage(2);
            f = tc.App.stage(2);
            F = vm.Context.artifact("Frames + window");
            f.onClick(F.FrameTimes(3));
            tc.verifyEqual(f.FrameIndex, 3);
            tc.verifyEqual(tc.rowValue(f, "Selected frame t"), "3");
        end

        function staleDisablesStepperUntilRerun(tc)
            vm = tc.App.ViewModel;
            vm.selectModel("ctc_fsdd_best");
            tc.App.run();
            tc.verifyTrue(all([tc.App.Stepper.Buttons.Enable] == "on"));
            vm.setSetting("Sensitivity", 0.3);
            tc.verifyTrue(all([tc.App.Stepper.Buttons.Enable] == "off"));
            tc.verifySubstring(string(tc.App.StaleLabel.Text), "Stale");
            tc.App.run();
            tc.verifyTrue(all([tc.App.Stepper.Buttons.Enable] == "on"));
            tc.verifyEqual(string(tc.App.StaleLabel.Text), "");
            tc.verifyEqual(vm.Context.Detector.Sensitivity, 0.3);
        end

        function fsddEncoderDecoderEveryStage(tc)
            vm = tc.App.ViewModel;
            tc.assumeTrue(any(vm.ModelNames == "encdec_fsdd_best"));
            vm.selectModel("encdec_fsdd_best");
            vm.setSplit("test");
            vm.selectUtterance(vm.Utterances(5));
            vm.setSetting("Decoder", "beam");
            tc.verifyTrue(tc.App.run(), tc.App.LastError);
            tc.verifyTrue(all(vm.StageEnabled));
            tc.verifyAllStages();
            p = tc.App.stage(7);
            D = vm.Context.artifact("Decoder posteriors");
            tc.verifyNotEmpty(D);
            tc.verifyEqual(tc.rowValue(p, "Posteriors (K x U)"), strjoin(string(size(D.Posteriors)), " x "));
            tc.verifyNotEmpty(tc.rowValue(p, "Cross-attention (U x T')"));
            tc.verifyTrue(any(startsWith(tc.App.stage(8).Rows(:, 1), "Decoder search steps")));
            n = tc.App.stage(8).shownNotation();
            tc.verifySubstring(n(1), "arg");
            vm.selectModel("ctc_fsdd_best");
        end

        function wavFileResampledAndSegmented(tc)
            f = fullfile(matlabroot, 'toolbox', 'audio', 'samples', 'Counting-16-44p1-mono-15secs.wav');
            tc.assumeTrue(isfile(f), 'Audio Toolbox sample not installed');
            vm = tc.App.ViewModel;
            vm.selectModel("ctc_fsdd_best");
            vm.setWavFile(string(f));
            vm.setReference("");
            tc.verifyEqual(vm.Source, "WAV");
            tc.verifyTrue(tc.App.run(), tc.App.LastError);
            r = vm.Context.Result;
            tc.verifyEqual(r.Report.InputRate, 44100);
            tc.verifyEqual(r.Conditioned.SampleRate, 8000);
            tc.verifyTrue(any(contains(r.Conditioned.History, "resample", 'IgnoreCase', true)));
            tc.verifyGreaterThan(vm.NumSegments, 1, '15 s input exceeds MaxSegment -> segmented');
            tc.verifyAllStages();
            vm.selectSegment(2);
            vm.selectStage(4);
            s = tc.App.stage(4);
            tc.verifyEqual(s.LastError, "");
            tc.verifyEqual(tc.App.SegmentDropDown.Value, 2);
            tc.verifyEqual(s.TimeOffsets(1), 0);           % log-mel axis is piece-local
            vm.selectStage(1);
            in = tc.App.stage(1);
            tc.verifyEqual(in.TimeOffsets(1), r.PieceTimes(2, 1), 'AbsTol', 1e-12);
            tc.verifyTrue(any(contains(vm.Warnings, "cannot be recognised")));
        end

        function injectedMicrophoneRun(tc)
            vm = tc.App.ViewModel;
            vm.selectModel("ctc_fsdd_best");
            vm.setSource("Microphone");
            vm.startRecording();
            for k = 1:3, vm.pollLevel(); end
            tc.verifyGreaterThan(vm.InputLevelDBFS, -120);
            tc.verifyEqual(tc.App.Sidebar.LevelGauge.Value, max(-60, min(0, vm.InputLevelDBFS)));
            vm.stopRecording();
            tc.verifyTrue(tc.App.run(), tc.App.LastError);
            r = vm.Context.Result;
            tc.verifyEqual(r.Input.Samples, tc.Samples(:, 1));
            tc.verifyEqual(r.Input.Source, "microphone");
            tc.verifyAllStages();
            % typed reference -> evaluation
            m = tc.App.ViewModel.Session.manifest();
            ref = m.Entries.Transcript(m.Entries.Id == tc.MicId);
            vm.setReference(ref);
            tc.App.run();
            tc.verifyNotEmpty(vm.Context.Result.Evaluation);
            vm.selectStage(9);
            tc.verifyNotEmpty(tc.rowValue(tc.App.stage(9), "WER"));
        end

        function workflowTabsFromSavedData(tc)
            vm = tc.App.ViewModel;
            vm.setMode("Workflow");
            tc.verifyEqual(tc.App.MainGrid.RowHeight{1}, 0);
            w = tc.App.Workflow;
            for k = 1:4
                w.selectTab(k);
                t = w.TabObjects{k};
                tc.verifyTrue(t.Built, sprintf('Workflow tab %d not built', k));
                tc.verifySubstring(string(t.SourceLabel.Text), "Source");
            end
            tc.verifyEqual(height(w.TabObjects{1}.Audit), height(vm.Session.manifest().audit()));
            h = w.TabObjects{2}.History;
            tc.verifyNotEmpty(h.StepLoss);
            tc.verifyGreaterThan(w.TabObjects{3}.Viewer.numTables(), 0);
            tc.verifyGreaterThan(w.TabObjects{4}.Viewer.numTables(), 0);
            vm.setMode("Pipeline");
            tc.verifyEqual(tc.App.MainGrid.RowHeight{2}, 0);
        end

        function pieceTimesInTranscriptTable(tc)
            f = fullfile(matlabroot, 'toolbox', 'audio', 'samples', 'Counting-16-44p1-mono-15secs.wav');
            tc.assumeTrue(isfile(f), 'Audio Toolbox sample not installed');
            vm = tc.App.ViewModel;
            vm.setWavFile(string(f));
            vm.setSetting("Segmentation", "never");
            tc.verifyTrue(tc.App.run(), tc.App.LastError);
            r = vm.Context.Result;
            tc.verifyEqual(size(r.PieceTimes, 1), 1);
            vm.selectStage(9);
            d = tc.App.stage(9).SegmentTable.Data;
            tc.verifySize(d, [1 5]);
            tc.verifyEqual([d{1, 2}, d{1, 3}], r.PieceTimes(1, :), 'AbsTol', 1e-12, ...
                'Whole-input piece spans the conditioned audio, not detector segment 1');
            tc.verifyEqual(d{1, 3}, r.Conditioned.duration(), 'AbsTol', 1e-3);
            tc.verifyEqual(string(d{1, 4}), r.Transcript.Text);
            vm.setSetting("Segmentation", "always");
            tc.App.run();
            r = vm.Context.Result;
            d = tc.App.stage(9).SegmentTable.Data;
            tc.verifyEqual(cell2mat(d(:, 2:3)), r.PieceTimes, 'AbsTol', 1e-12);
        end

        function noSpeechShowsOneClearState(tc)
            vm = tc.App.ViewModel;
            rec = vm.Recorder;
            tc.assumeTrue(isa(rec, 'OceanViz.App.NLP.SpeechStudio.Models.InjectedRecorder'));
            vm.Session.useSamples(zeros(8000, 1), 8000, "Digital silence (1 s)");
            vm.setSource("Microphone");
            tc.verifyTrue(tc.App.run(), tc.App.LastError);
            tc.verifyTrue(vm.NoSpeech);
            tc.verifyEqual(vm.StageEnabled, [true, false(1, 7), true]);
            tc.verifyEqual(string({tc.App.Stepper.Buttons(2:8).Enable}), repmat("off", 1, 7));
            vm.selectStage(1);
            tc.verifyEqual(tc.App.stage(1).LastError, "", 'Detector evidence is drawn on stage 1');
            vm.selectStage(9);
            s9 = tc.App.stage(9);
            tc.verifyEqual(s9.LastError, "");
            tc.verifySubstring(string(s9.TextLabel.Text), "No speech detected");
            vm.selectStage(4);
            tc.verifyEqual(vm.CurrentStage, 9, 'Recognition stages cannot be opened');
        end

        function runDisabledWithoutInputForSource(tc)
            vm = tc.App.ViewModel;
            tc.verifyEqual(char(tc.App.RunButton.Enable), 'on');
            vm.setSource("WAV");
            tc.verifyEqual(char(tc.App.RunButton.Enable), 'off');
            tc.verifySubstring(string(tc.App.Sidebar.InputLabel.Text), "No WAV file");
            tc.verifyFalse(tc.App.run());
            vm.setSource("FSDD");
            tc.verifyEqual(char(tc.App.RunButton.Enable), 'on');
        end

        function stepperUsesPlainNumbers(tc)
            t = string({tc.App.Stepper.Buttons.Text});
            for k = 1:numel(t)
                tc.verifyTrue(startsWith(erase(t(k), "▶ "), string(k) + "."), t(k));
            end
        end
    end

    methods (Test, TestTags = {'Close'})
        function closeCleansUpOwnInstance(tc)
            % Builds and closes its OWN app, so the shared app of the other tests is unaffected.
            app = OceanViz.App.NLP.SpeechStudio.launch('Visible', false, ...
                'Manifest', tc.App.ViewModel.Session.manifest());
            fig = app.Figure;
            vm = app.ViewModel;
            sess = vm.Session;
            app.run();
            tc.verifyEqual(sess.numCachedRuns(), 1);
            nTimers = numel(timerfindall);
            app.close();
            tc.verifyFalse(isvalid(app));
            tc.verifyFalse(isvalid(fig));
            tc.verifyFalse(isvalid(vm));
            tc.verifyEqual(sess.numCachedRuns(), 0, 'close() releases the cached runs (traces)');
            tc.verifyEmpty(sess.Context);
            tc.verifyLessThanOrEqual(numel(timerfindall), nTimers);
            tc.verifyTrue(isvalid(tc.App), 'Shared app untouched');
        end
    end
end
