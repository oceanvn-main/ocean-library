classdef ShortInputTest < OceanTest.BaseTestCase
    % SHORTINPUTTEST  Contract sec. 3: audio shorter than the encoders' receptive field
    %   gives an empty transcript, not an error -- alone and inside a batch, for both
    %   trained recognisers (critic round 1, finding 1). Uses the saved FSDD checkpoints.

    methods (Test)
        function testCtcShortWaveformAloneAndInBatch(testCase)
            testCase.checkModel("ctc_fsdd_best.mat", {'Decoder', "greedy"});
            testCase.checkModel("ctc_fsdd_best.mat", {'Decoder', "beam", 'BeamWidth', 4});
        end

        function testEncoderDecoderShortWaveformAloneAndInBatch(testCase)
            testCase.checkModel("encdec_fsdd_best.mat", {'Decoder', "greedy"});
        end
    end

    methods
        function checkModel(testCase, file, decodeArgs)
            path = fullfile(testCase.LibraryRoot, 'Results', 'Speech', 'checkpoints', file);
            testCase.assumeTrue(isfile(path), "checkpoint not present: run the experiment script first");
            ck = OceanNLP.Speech.Training.SpeechCheckpoint.load(path);
            m = ck.Model;
            fs = 8000;
            shortWave = OceanNLP.Speech.Core.Waveform(sin(2 * pi * 200 * (0:49)' / fs), fs);   % 50 samples < 1 frame
            root = OceanNLP.Speech.Data.SpeechManifest.defaultRoot();
            real = OceanNLP.Speech.Core.Waveform.read("FSDD/recordings/3_theo_10.wav", 'Root', root);
            alone = m.transcribe(shortWave, decodeArgs{:});
            testCase.verifyEqual(alone.Text, "");
            testCase.verifyEmpty(alone.TokenIds);
            both = m.transcribe([shortWave, real], decodeArgs{:});
            testCase.verifyEqual(both(1).Text, "");
            single = m.transcribe(real, decodeArgs{:});
            testCase.verifyEqual(both(2).Text, single.Text);   % padding invariance kept
        end
    end
end
