classdef FeaturePipelineTest < OceanTest.BaseTestCase
    % FEATUREPIPELINETEST  Composition equals manual chaining; config/outputDim/type policy.

    methods (Test)
        function testPipelineEqualsManualComposition(testCase)
            w = testCase.wave();
            m = OceanNLP.Speech.Features.MFCC(8000);
            d = OceanNLP.Speech.Features.DeltaFeatures();
            p = OceanNLP.Speech.Features.FeaturePipeline({m, d});
            out = p.forward(w);
            ref = d.forward(m.forward(w));
            testCase.verifyEqual(out.Values, ref.Values);
            testCase.verifyEqual(out.FrameTimes, ref.FrameTimes);
            testCase.verifyEqual(p.outputDim(), 39);
            cfg = out.Config;
            testCase.verifyEqual(cfg.Type, "FeaturePipeline");
            testCase.verifyEqual([cfg.Stages.Type], ["MFCC", "DeltaFeatures"]);
            testCase.verifyEqual(cfg.Stages(1).Config, m.config());
        end

        function testHistoryAndTypes(testCase)
            x = testCase.wave().Samples;
            w2 = OceanNLP.Speech.Core.Waveform([x, x], 8000);
            p = OceanNLP.Speech.Features.FeaturePipeline({OceanNLP.Speech.Features.LogMel(8000), ...
                OceanNLP.Speech.Features.DeltaFeatures('Order', 1)});
            out = p.forward(w2);
            testCase.verifyEqual(out.Config.InputHistory, "mono mean of 2 channels");
            testCase.verifyEqual(numel(out.FrequencyHz), 80);
            testCase.verifyError(@() OceanNLP.Speech.Features.FeaturePipeline({1}), ...
                'OceanNLP:Speech:Features:FeaturePipeline:StageType');
            testCase.verifyError(@() p.forward(out), 'OceanNLP:Speech:Features:FeaturePipeline:InputType');
            bad = OceanNLP.Speech.Features.FeaturePipeline({OceanNLP.Speech.Features.DeltaFeatures()});
            testCase.verifyError(@() bad.forward(w2), 'OceanNLP:Speech:Features:DeltaFeatures:InputType');
        end
    end

    methods
        function w = wave(testCase)
            w = OceanNLP.Speech.Core.Waveform.read("recordings/9_theo_4.wav", ...
                'Root', fullfile(testCase.LibraryRoot, 'Data', 'Datasets', 'Speech', 'FSDD'));
        end
    end
end
