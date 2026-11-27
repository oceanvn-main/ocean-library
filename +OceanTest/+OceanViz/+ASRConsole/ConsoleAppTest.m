classdef ConsoleAppTest < matlab.unittest.TestCase
    % CONSOLEAPPTEST  The ASR Console window (hidden) bound to its ViewModel, with the fake
    %   service and an injected recording; plus a LIVE test against the real service that runs
    %   only when the service answers (start it with `uv run asr serve` in Production SI/ASR).

    methods (Test)
        function windowFollowsTheViewModel(tc)
            client = OceanViz.App.NLP.ASRConsole.Models.FakeRecognitionClient();
            fs = 16000;
            x = OceanTest.OceanViz.ASRConsole.ConsoleViewModelTest.speech(fs);   % real speech (gate 2 passes)
            app = OceanViz.App.NLP.ASRConsole.launch('Visible', false, 'Client', client, ...
                'RecorderFactory', @(r) OceanViz.App.NLP.SpeechStudio.Models.InjectedRecorder(x, fs), ...
                'Player', @(~, ~) [], 'SaveFolder', string(tempname));
            tc.addTeardown(@() app.close());
            vm = app.ViewModel;
            tc.verifyEqual(app.ServiceLabel.Text, 'Connected: service fake');
            tc.verifyEqual(numel(app.ModelControls), 3);
            tc.verifyEqual(app.ModelControls(3).Check.Enable, matlab.lang.OnOffSwitchState.off);
            tc.verifyEqual(app.TranscribeButton.Enable, matlab.lang.OnOffSwitchState.off, 'no audio yet');

            vm.startRecording();
            tc.verifyEqual(app.StopButton.Enable, matlab.lang.OnOffSwitchState.on);
            tc.verifyEqual(app.RecordButton.Enable, matlab.lang.OnOffSwitchState.off);
            vm.stopRecording();
            lines = findobj(app.WaveAxes, 'Type', 'line');
            tc.verifyNumElements(lines, 1);
            tc.verifyEqual(numel(lines.XData), numel(x));
            tc.verifyEqual(app.TranscribeButton.Enable, matlab.lang.OnOffSwitchState.on, 'speech: may transcribe');
            vm.setAudio(0.25 * sin(2 * pi * 220 * (0:fs - 1)' / fs), fs, "tone");
            tc.verifyEqual(app.TranscribeButton.Enable, matlab.lang.OnOffSwitchState.off, 'gate 2: a tone is not speech');
            vm.setAudio(x, fs, "speech again");
            tc.verifyNotEmpty(app.WaveAxes.Legend, 'OceanViz waveform carries a legend');

            app.ModelControls(1).Backend.Value = 'onnx_int8';
            app.ModelControls(1).Backend.ValueChangedFcn(app.ModelControls(1).Backend, []);
            tc.verifyEqual(vm.ModelTable.Backend(1), "onnx_int8");
            app.transcribe();
            tc.verifyEqual(height(app.ResultsTable.Data), 2);
            tc.verifyEqual(app.ResultsTable.Data.Backend(1), "onnx_int8");
            tc.verifyEqual(height(app.HistoryTable.Data), 2);
            tc.verifySubstring(string(app.StatusLabel.Text), "Done: 2 model(s)");
        end

        function liveServiceRecognisesSpeech(tc)
            client = OceanViz.App.NLP.ASRConsole.Models.RecognitionClient('Timeout', 300);
            try
                client.health();
            catch
                tc.assumeFail('Recognition service not running (uv run asr serve).');
            end
            wav = OceanTest.OceanViz.ASRConsole.ConsoleAppTest.librispeechSample();
            tc.assumeNotEmpty(wav, 'No LibriSpeech test-clean FLAC found in the ASR repository.');
            [x, fs] = audioread(wav);
            m = client.models();
            avail = m([m.available]);
            tc.assumeNotEmpty(avail);
            choices = struct('model', cellstr([avail.model]), 'backend', cellstr([avail.default_backend]));
            r = client.recognize(x, fs, choices, 4);
            tc.verifyEqual(r.duration_s, numel(x) / fs, 'AbsTol', 1e-6);
            for k = 1:numel(r.results)
                tc.verifyEqual(r.results(k).error, "", r.results(k).model);
                tc.verifyGreaterThan(strlength(r.results(k).text), 0, r.results(k).model);
                tc.verifyGreaterThan(r.results(k).latency_ms, 0);
            end
        end
    end

    methods (Static)
        function f = librispeechSample()
            f = "";
            root = "C:\Ocean Library\Production SI\ASR\data";
            if ~isfolder(root), return; end
            d = dir(fullfile(root, "**", "*.flac"));
            if ~isempty(d), f = string(fullfile(d(1).folder, d(1).name)); end
        end
    end
end
