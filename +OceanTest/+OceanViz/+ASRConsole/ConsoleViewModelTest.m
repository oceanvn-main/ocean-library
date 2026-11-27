classdef ConsoleViewModelTest < matlab.unittest.TestCase
    % CONSOLEVIEWMODELTEST  Headless tests of the ASR Console ViewModel (A10): the service is a
    %   FakeRecognitionClient, the microphone an InjectedRecorder, the speaker a spy. Checks the
    %   connect / record / transcribe / history / save flow and that every failure stays a message.

    properties
        Client
        Played = {}
        Folder
    end

    methods (TestMethodSetup)
        function setup(tc)
            tc.Client = OceanViz.App.NLP.ASRConsole.Models.FakeRecognitionClient();
            tc.Played = {};
            tc.Folder = string(tempname);
            tc.addTeardown(@() OceanTest.OceanViz.ASRConsole.ConsoleViewModelTest.removeFolder(tc.Folder));
        end
    end

    methods
        function vm = newVM(tc, x, fs)
            if nargin < 2
                [x, fs] = OceanTest.OceanViz.ASRConsole.ConsoleViewModelTest.speech(16000);
            end
            factory = @(rate) OceanViz.App.NLP.SpeechStudio.Models.InjectedRecorder(x, fs, 'Name', "tone");
            vm = OceanViz.App.NLP.ASRConsole.ViewModels.ConsoleViewModel('Client', tc.Client, ...
                'RecorderFactory', factory, 'Player', @(y, r) tc.play(y, r), 'SaveFolder', tc.Folder);
            tc.addTeardown(@() delete(vm));
        end

        function play(tc, y, r)
            tc.Played{end + 1} = {y, r};
        end
    end

    methods (Static)
        function [x, fs] = tone(seconds, fs)
            t = (0:round(seconds * fs) - 1)' / fs;
            x = 0.25 * sin(2 * pi * 220 * t);
        end

        function [x, fs] = speech(fs)
            % SPEECH  Real speech (Audio Toolbox sample: a woman reading, 3 s, recorded at 8 kHz), at rate fs.
            [x, fs0] = audioread("FemaleSpeech-16-8-mono-3secs.wav");
            x = 0.5 * x(:, 1) / max(abs(x(:, 1)));
            if fs ~= fs0, x = resample(x, fs, fs0); end
        end

        function removeFolder(f)
            if isfolder(f), rmdir(f, 's'); end
        end
    end

    methods (Test)
        function connectListsModelsAndHardware(tc)
            vm = tc.newVM();
            tc.verifyTrue(vm.connect());
            tc.verifyTrue(vm.ServiceOnline);
            t = vm.ModelTable;
            tc.verifyEqual(t.Model', ["champion", "reference", "citrinet"]);
            tc.verifyEqual(t.Use', [true true false], 'unavailable models start unticked');
            tc.verifyEqual(t.Backends{1}, ["pytorch", "onnx_fp32", "onnx_int8"]);
            tc.verifySubstring(vm.HardwareText, "NOT usable by PyTorch (fake: no kernels)");
            tc.verifySubstring(vm.HardwareText, "CPUExecutionProvider");
        end

        function offlineServiceIsAMessageNotAnError(tc)
            tc.Client.Online = false;
            vm = tc.newVM();
            tc.verifyFalse(vm.connect());
            tc.verifyFalse(vm.ServiceOnline);
            tc.verifyEqual(vm.ServiceText, "Offline");
            tc.verifySubstring(vm.StatusText, "asr serve");
        end

        function recordStopTranscribeFillsResultsAndHistory(tc)
            fs = 48000;
            x = OceanTest.OceanViz.ASRConsole.ConsoleViewModelTest.speech(fs);
            vm = tc.newVM(x, fs);
            vm.connect();
            vm.setSampleRate(48000);
            vm.setBackend("champion", "onnx_fp32");
            vm.setThreads(2);
            vm.startRecording();
            tc.verifyTrue(vm.IsRecording);
            vm.pollLevel();
            tc.verifyGreaterThan(vm.InputLevelDBFS, -60, 'the meter shows the speech level');
            vm.stopRecording();
            tc.verifyFalse(vm.IsRecording);
            tc.verifyEqual(vm.Samples, x);
            tc.verifyEqual(vm.AudioRate, 48000);
            tc.Client.Texts.champion = "HELLO WORLD";
            vm.transcribe();
            call = tc.Client.Calls(end);
            tc.verifyEqual(call.fs, 48000);
            tc.verifyEqual(call.threads, 2);
            tc.verifyEqual(string({call.choices.model}), ["champion", "reference"]);
            tc.verifyEqual(string({call.choices.backend}), ["onnx_fp32", "pytorch"]);
            r = vm.Results;
            tc.verifyEqual(r.Text', ["HELLO WORLD", "REFERENCE TEXT"]);
            tc.verifyEqual(height(vm.History), 2);
            tc.verifyFalse(vm.IsBusy);
            vm.transcribe();
            tc.verifyEqual(height(vm.History), 4, 'history accumulates');
        end

        function aFailingModelStaysInItsRow(tc)
            vm = tc.newVM();
            vm.connect();
            vm.setAudio(OceanTest.OceanViz.ASRConsole.ConsoleViewModelTest.speech(16000), 16000, "speech");
            tc.Client.FailModels = "reference";
            vm.transcribe();
            r = vm.Results;
            tc.verifyEqual(r.Error', ["", "fake failure"]);
            tc.verifyTrue(isnan(r.LatencyMs(2)));
            tc.verifySubstring(vm.StatusText, "1 failed");
            tc.verifySubstring(vm.History.Text(2), "ERROR: fake failure");
        end

        function serviceDroppingDuringTranscribeMarksOffline(tc)
            vm = tc.newVM();
            vm.connect();
            vm.setAudio(OceanTest.OceanViz.ASRConsole.ConsoleViewModelTest.speech(16000), 16000, "speech");
            tc.Client.Online = false;
            vm.transcribe();
            tc.verifyFalse(vm.ServiceOnline);
            tc.verifyFalse(vm.IsBusy);
            tc.verifySubstring(vm.StatusText, "Transcription failed");
        end

        function guardsWithoutAudioOrModels(tc)
            vm = tc.newVM();
            vm.transcribe();
            tc.verifySubstring(vm.StatusText, "Record or load audio first");
            tc.verifyEmpty(tc.Client.Calls);
            vm.connect();
            vm.setAudio(OceanTest.OceanViz.ASRConsole.ConsoleViewModelTest.speech(16000), 16000, "speech");
            vm.setUse("champion", false);
            vm.setUse("reference", false);
            vm.transcribe();
            tc.verifySubstring(vm.StatusText, "Tick at least one");
            vm.setUse("citrinet", true);
            tc.verifyFalse(vm.ModelTable.Use(3), 'an unavailable model cannot be ticked');
            tc.verifyError(@() vm.setBackend("reference", "onnx_int8"), 'OceanViz:App:NLP:ASRConsole:Backend');
        end

        function reconnectKeepsTheUsersChoices(tc)
            vm = tc.newVM();
            vm.connect();
            vm.setUse("champion", false);
            vm.setBackend("champion", "onnx_int8");
            vm.connect();
            tc.verifyFalse(vm.ModelTable.Use(1));
            tc.verifyEqual(vm.ModelTable.Backend(1), "onnx_int8");
        end

        function clippingAndQuietInputAreFlagged(tc)
            vm = tc.newVM();
            vm.setAudio(max(-1, min(1, 8 * OceanTest.OceanViz.ASRConsole.ConsoleViewModelTest.speech(16000))), 16000);
            tc.verifyTrue(vm.SpeechOk);
            tc.verifySubstring(vm.StatusText, "clips");
            vm.setAudio(1e-3 * OceanTest.OceanViz.ASRConsole.ConsoleViewModelTest.tone(1, 16000), 16000);
            tc.verifyFalse(vm.SpeechOk);
            tc.verifySubstring(vm.StatusText, "far too quiet");
        end

        function playAndSaveWriteWhatWasRecorded(tc)
            vm = tc.newVM();
            vm.connect();
            fs = 16000;
            x = OceanTest.OceanViz.ASRConsole.ConsoleViewModelTest.speech(fs);
            vm.setAudio(x, fs, "speech");
            vm.play();
            tc.verifyEqual(tc.Played{1}{1}, x);
            tc.verifyEqual(tc.Played{1}{2}, fs);
            vm.transcribe();
            wav = vm.save();
            [y, r] = audioread(wav);
            tc.verifyEqual(r, fs);
            tc.verifyEqual(y, x, 'AbsTol', 1 / 32768);
            [p, n] = fileparts(wav);
            doc = jsondecode(fileread(fullfile(p, n + ".json")));
            tc.verifyEqual(string({doc.results.Text}), ["CHAMPION TEXT", "REFERENCE TEXT"]);
            tc.verifyEqual(doc.sample_rate, fs);
        end

        function loadWavMixesToMono(tc)
            vm = tc.newVM();
            f = fullfile(tc.Folder, "stereo.wav");
            mkdir(tc.Folder);
            [x, fs] = OceanTest.OceanViz.ASRConsole.ConsoleViewModelTest.tone(0.5, 22050);
            audiowrite(f, [x, 0.5 * x], fs);
            vm.loadWav(f);
            tc.verifyEqual(vm.AudioRate, 22050);
            tc.verifyEqual(vm.Samples, 0.75 * x, 'AbsTol', 2 / 32768);
            vm.loadWav(fullfile(tc.Folder, "missing.wav"));
            tc.verifySubstring(vm.StatusText, "Could not read");
        end

        % ---- the three gates (user requirement 2026-10-20): no model runs without real speech ----------
        function noMicrophoneMeansNoRecordingAndNothingSent(tc)
            vm = OceanViz.App.NLP.ASRConsole.ViewModels.ConsoleViewModel('Client', tc.Client, ...
                'RecorderFactory', @(rate) OceanTest.OceanViz.ASRConsole.NoDeviceRecorder(), ...
                'Player', @(y, r) tc.play(y, r), 'SaveFolder', tc.Folder);
            tc.addTeardown(@() delete(vm));
            vm.connect();
            vm.startRecording();
            tc.verifyFalse(vm.IsRecording);
            tc.verifySubstring(vm.StatusText, "No microphone found");
            vm.transcribe();
            tc.verifyEmpty(tc.Client.Calls);
        end

        function pureSilenceIsRefusedBeforeTheService(tc)
            vm = tc.newVM();
            vm.connect();
            vm.setAudio(zeros(16000, 1), 16000, "muted mic");
            tc.verifyFalse(vm.SpeechOk);
            tc.verifySubstring(vm.StatusText, "pure silence");
            vm.transcribe();
            tc.verifyEmpty(tc.Client.Calls, 'gate 3: nothing reaches the service');
            tc.verifySubstring(vm.StatusText, "pure silence");
        end

        function noiseToneAndTooShortSpeechAreRefused(tc)
            vm = tc.newVM();
            vm.connect();
            rng(0);
            sp = OceanTest.OceanViz.ASRConsole.ConsoleViewModelTest.speech(16000);
            cases = {{1e-4 * randn(16000, 1), "far too quiet"}, ...
                {OceanTest.OceanViz.ASRConsole.ConsoleViewModelTest.tone(2, 16000), "No speech detected"}, ...
                {0.02 * randn(32000, 1), "No speech detected"}, ...
                {sp(1:round(0.2 * 16000)), "No speech detected"}};   % 0.2 s < 0.3 s
            for k = 1:numel(cases)
                vm.setAudio(cases{k}{1}, 16000, "case " + k);
                tc.verifyFalse(vm.SpeechOk, "case " + k);
                tc.verifySubstring(vm.StatusText, cases{k}{2}, "case " + k);
                vm.transcribe();
            end
            tc.verifyEmpty(tc.Client.Calls);
        end

        function realSpeechPassesAndIsSentOnce(tc)
            vm = tc.newVM();
            vm.connect();
            vm.setAudio(OceanTest.OceanViz.ASRConsole.ConsoleViewModelTest.speech(16000), 16000, "speech");
            tc.verifyTrue(vm.SpeechOk);
            tc.verifySubstring(vm.StatusText, "of speech. Press Transcribe");
            vm.transcribe();
            tc.verifyNumElements(tc.Client.Calls, 1);
        end
    end
end
