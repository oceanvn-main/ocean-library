classdef StudioViewModelTest < matlab.unittest.TestCase
    % STUDIOVIEWMODELTEST  Headless tests of the Speech Pipeline Studio Model + ViewModel
    %   (no UI is created): stale / rerun logic, capability-driven controls and stages,
    %   stage navigation, injected microphone capture, run cache, report parsing.
    %   Real data: FSDD recordings and the saved checkpoints in Results/Speech/checkpoints.

    properties
        Session
    end

    methods (TestClassSetup)
        function makeSession(tc)
            tc.Session = OceanViz.App.NLP.SpeechStudio.Models.StudioSession();
            tc.assumeNotEmpty(tc.Session.models(), 'No speech checkpoints found.');
        end
    end

    methods
        function vm = newVM(tc, varargin)
            vm = OceanViz.App.NLP.SpeechStudio.ViewModels.StudioViewModel(tc.Session, varargin{:});
            tc.addTeardown(@() delete(vm));
            vm.initialize();
        end
    end

    methods (Test)
        function stagesDisabledUntilRun(tc)
            vm = tc.newVM();
            tc.verifyEqual(numel(vm.StageTitles), 9);
            tc.verifyFalse(any(vm.StageEnabled));
            tc.verifyFalse(vm.HasResult);
            tc.verifyEqual(vm.ModelName, "ctc_fsdd_best");
            tc.verifyTrue(vm.run());
            tc.verifyTrue(all(vm.StageEnabled), 'CTC + log-mel model supports every stage');
            tc.verifyTrue(vm.HasResult);
            tc.verifyFalse(vm.IsStale);
            tc.verifyFalse(vm.IsBusy);
        end

        function settingChangeMarksStaleAndRerunClears(tc)
            vm = tc.newVM();
            vm.run();
            vm.setSetting("BeamWidth", 3);
            tc.verifyTrue(vm.IsStale);
            tc.verifyFalse(any(vm.StageEnabled));
            vm.selectStage(4);
            tc.verifyEqual(vm.CurrentStage, 1, 'Stale stages cannot be selected');
            vm.run();
            tc.verifyFalse(vm.IsStale);
            tc.verifyTrue(all(vm.StageEnabled));
            tc.verifyEqual(vm.Context.Settings.BeamWidth, 3);
            tc.verifyEqual(vm.Context.Result.DecodeOptions.BeamWidth, 3);
            % same value again: not stale
            vm.setSetting("BeamWidth", 3);
            tc.verifyFalse(vm.IsStale);
            % input change: stale
            vm.selectUtterance(vm.Utterances(2));
            tc.verifyTrue(vm.IsStale);
        end

        function unknownSettingErrors(tc)
            vm = tc.newVM();
            tc.verifyError(@() vm.setSetting("NoSuchThing", 1), ...
                'OceanViz:App:NLP:SpeechStudio:StudioViewModel:UnknownSetting');
        end

        function decoderControlsFollowModel(tc)
            vm = tc.newVM();
            vm.setSetting("Decoder", "beam");
            tc.verifyTrue(vm.showControl("LMWeight"));
            tc.verifyFalse(vm.showControl("CTCRescoreWeight"));
            vm.setSetting("Decoder", "greedy");
            tc.verifyFalse(vm.showControl("LMWeight"));
            tc.verifyFalse(vm.showControl("BeamWidth"));
            names = vm.ModelNames;
            tc.assumeTrue(any(names == "encdec_fsdd_best"));
            vm.selectModel("encdec_fsdd_best");
            vm.setSetting("Decoder", "beam");
            tc.verifyTrue(any(vm.Capabilities == "attention"));
            tc.verifyFalse(vm.showControl("LMWeight"));
            tc.verifyEqual(vm.showControl("CTCRescoreWeight"), any(vm.Capabilities == "ctc-rescore"));
        end

        function navigationSkipsAndClamps(tc)
            vm = tc.newVM();
            vm.run();
            vm.selectStage(9);
            tc.verifyFalse(vm.canGoNext());
            vm.next();
            tc.verifyEqual(vm.CurrentStage, 9);
            vm.back();
            tc.verifyEqual(vm.CurrentStage, 8);
            vm.selectStage(1);
            tc.verifyFalse(vm.canGoBack());
            vm.next();
            tc.verifyEqual(vm.CurrentStage, 2);
        end

        function injectedMicrophoneSamples(tc)
            m = tc.Session.manifest();
            i = find(m.Entries.Split == "test", 1);
            [x, fs] = audioread(m.resolve(i));
            rec = OceanViz.App.NLP.SpeechStudio.Models.InjectedRecorder(x, fs, 'Name', m.Entries.Id(i));
            vm = tc.newVM('Recorder', rec);
            vm.startRecording();
            tc.verifyTrue(vm.IsRecording);
            vm.pollLevel();
            tc.verifyLessThan(vm.InputLevelDBFS, 0);
            tc.verifyGreaterThan(vm.InputLevelDBFS, -120);
            vm.stopRecording();
            tc.verifyFalse(vm.IsRecording);
            tc.verifyEqual(vm.Source, "Microphone");
            tc.verifyEqual(tc.Session.InputKind, "microphone");
            tc.verifyEqual(tc.Session.Input.Samples, x(:, 1));
            tc.verifyEqual(tc.Session.Input.Source, "microphone");
            tc.verifyTrue(vm.run());
            tc.verifyEqual(vm.Context.InputKind, "microphone");
            tc.verifyTrue(any(contains(vm.Warnings, "cannot be recognised")), ...
                'Microphone input states the model vocabulary limit');
        end

        function runCacheReturnsSameContext(tc)
            vm = tc.newVM();
            vm.run();
            c1 = vm.Context;
            vm.setSetting("BeamWidth", 2);
            vm.run();
            vm.setSetting("BeamWidth", 8);
            vm.run();
            tc.verifyTrue(vm.Context == c1, 'Same input + settings reuse the cached run');
        end

        function settingsDefaultsComeFromLibrary(tc)
            s = OceanViz.App.NLP.SpeechStudio.Models.StudioSession.defaultSettings();
            c = OceanNLP.Speech.Inference.AudioConditioner().config();
            d = OceanNLP.Speech.Inference.SpeechActivityDetector().config();
            tc.verifyEqual(s.TargetRMSdBFS, c.TargetRMSdBFS);
            tc.verifyEqual(s.MaxGainDB, c.MaxGainDB);
            tc.verifyEqual(s.Sensitivity, d.Sensitivity);
            tc.verifyEqual(s.MaxSegment, d.MaxSegment);
        end

        function reportTablesParseGeneratedReports(tc)
            f = fullfile(tc.Session.Root, 'Results', 'Speech', 'ctc_fsdd_report.md');
            tc.assumeTrue(isfile(f));
            r = OceanViz.App.NLP.SpeechStudio.Models.ReportTables.read(f);
            tc.verifySubstring(r.Generator, "Generated by");
            tc.verifyNotEmpty(r.Tables);
            t = r.Tables(1);
            tc.verifyEqual(t.Header(1), "Split");
            tc.verifyEqual(size(t.Rows, 2), numel(t.Header));
            lines = readlines(f);
            tc.verifyTrue(startsWith(strip(lines(t.Line)), "| Split"));
        end

        function reportTablesRejectHandWritten(tc)
            f = [tempname '.md'];
            writelines(["# Notes", "| a | b |", "| --- | --- |", "| 1 | 2 |"], f);
            tc.addTeardown(@() delete(f));
            tc.verifyError(@() OceanViz.App.NLP.SpeechStudio.Models.ReportTables.read(f), ...
                'OceanViz:App:NLP:SpeechStudio:ReportTables:NotGenerated');
        end

        function modelHoldsNoViewReferences(tc)
            vm = tc.newVM();
            mc = metaclass(tc.Session);
            for p = mc.PropertyList'
                if p.Dependent || p.Constant, continue; end
                if ~strcmp(p.GetAccess, 'public'), continue; end
                v = tc.Session.(p.Name);
                tc.verifyFalse(isa(v, 'OceanViz.App.NLP.SpeechStudio.ViewModels.StudioViewModel'));
                tc.verifyFalse(isa(v, 'OceanViz.App.NLP.SpeechStudio.Views.StudioView'));
            end
            tc.verifyClass(vm, 'OceanViz.App.NLP.SpeechStudio.ViewModels.StudioViewModel');
        end

        function sessionsHaveIndependentReleasableCaches(tc)
            S = 'OceanViz.App.NLP.SpeechStudio.Models.StudioSession';
            m = tc.Session.manifest();
            s1 = feval(S, 'Manifest', m);
            s2 = feval(S, 'Manifest', m);
            tc.addTeardown(@() delete([s1, s2]));
            s1.selectModel("ctc_fsdd_best");
            s1.useFSDD(m.Entries.Id(find(m.Entries.Split == "test", 1)));
            s1.run();
            tc.verifyEqual(s1.numCachedRuns(), 1);
            tc.verifyEqual(s2.numCachedRuns(), 0, 'A new session must not see another session''s runs');
            s3 = feval(S, 'Manifest', m);
            tc.addTeardown(@() delete(s3));
            tc.verifyEqual(s3.numCachedRuns(), 0);
            s1.releaseRuns();
            tc.verifyEqual(s1.numCachedRuns(), 0);
            tc.verifyEmpty(s1.Context);
        end

        function sourceChangeRequiresInputForThatSource(tc)
            vm = tc.newVM();
            tc.verifyTrue(vm.InputReady);
            vm.run();
            c1 = vm.Context;
            vm.setSource("WAV");
            tc.verifyFalse(vm.InputReady);
            tc.verifySubstring(vm.InputSummary, "No WAV file");
            tc.verifyFalse(vm.run(), 'Run must not silently reuse the FSDD input');
            tc.verifySubstring(vm.StatusText, "Cannot run");
            tc.verifyTrue(vm.Context == c1);
            vm.setSource("Microphone");
            tc.verifyFalse(vm.InputReady);
            tc.verifySubstring(vm.InputSummary, "Nothing recorded");
            vm.setSource("FSDD");
            tc.verifyTrue(vm.InputReady);
            tc.verifyTrue(vm.run());
        end

        function switchingBackRestoresRememberedInput(tc)
            % critic round 2, N2: FSDD -> load WAV -> back to FSDD left "No FSDD utterance
            % selected" although the dropdown still showed the utterance
            vm = tc.newVM();
            id = vm.UtteranceId;
            tc.assertTrue(strlength(id) > 0);
            wav = string(fullfile(matlabroot, 'toolbox', 'audio', 'samples', 'SpeechDFT-16-8-mono-5secs.wav'));
            tc.assumeTrue(isfile(wav), 'Audio Toolbox sample not present.');
            vm.setWavFile(wav);
            tc.verifyEqual(vm.Source, "WAV");
            vm.setSource("FSDD");
            tc.verifyTrue(vm.InputReady);
            tc.verifySubstring(vm.InputSummary, id);
            tc.verifyEqual(tc.Session.InputKind, "fsdd");
            vm.setSource("WAV");
            tc.verifyTrue(vm.InputReady, 'the remembered WAV file is reloaded too');
            tc.verifyEqual(tc.Session.InputKind, "wav");
        end

        function silentInputDisablesRecognitionStages(tc)
            rec = OceanViz.App.NLP.SpeechStudio.Models.InjectedRecorder(zeros(16000, 1), 16000, 'Name', "digital silence");
            vm = tc.newVM('Recorder', rec);
            vm.startRecording();
            vm.stopRecording();
            tc.verifyTrue(vm.InputReady);
            tc.verifyTrue(vm.run());
            tc.verifyTrue(vm.NoSpeech);
            tc.verifyEmpty(vm.Context.Result.SegmentTranscripts);
            tc.verifyEqual(vm.StageEnabled, [true, false(1, 7), true]);
            tc.verifySubstring(vm.StatusText, "No speech detected");
            tc.verifyTrue(any(contains(vm.Warnings, "speech", 'IgnoreCase', true)));
        end

        function randomHeldOutLeavesGlobalRngAlone(tc)
            vm = tc.newVM();
            rng(123);
            before = rng;
            id = vm.randomHeldOut();
            tc.verifyEqual(rng, before);
            tc.verifyEqual(vm.Split, "test");
            tc.verifyEqual(vm.UtteranceId, id);
            tc.verifyEqual(vm.randomHeldOut(7), vm.randomHeldOut(7), 'Seeded choice is reproducible');
        end

        function modelCardFormatsParameterCount(tc)
            vm = tc.newVM();
            tc.assumeTrue(any(vm.ModelNames == "encdec_fsdd_best"));
            vm.selectModel("encdec_fsdd_best");
            n = tc.Session.modelRow().NumParams;
            tc.verifySubstring(vm.ModelCard, sprintf("%.2f M params", n / 1e6));
            tc.verifyFalse(contains(vm.ModelCard, "e+"));
        end
    end
end
