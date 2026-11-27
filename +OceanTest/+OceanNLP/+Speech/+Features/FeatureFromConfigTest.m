classdef FeatureFromConfigTest < OceanTest.BaseTestCase
    % FEATUREFROMCONFIGTEST  Every feature transform's static fromConfig inverts config():
    %   same config, identical features on real audio; the factory dispatches to them
    %   (including stages inside a FeaturePipeline).

    methods (Test)
        function testEveryTransformRoundTrips(testCase)
            root = fullfile(testCase.LibraryRoot, 'Data', 'Datasets', 'Speech', 'FSDD');
            w = OceanNLP.Speech.Core.Waveform.read("recordings/5_nicolas_3.wav", 'Root', root);
            import OceanNLP.Speech.Features.*
            transforms = {LogMel(8000, 'NumMels', 32, 'Center', true, 'PadMode', "reflect", 'MelScale', "slaney", 'MelNorm', "slaney"), ...
                MFCC(8000, 'NumCoeffs', 13, 'Lifter', 22, 'NFFT', 512), ...
                EnergyContour(8000, 'ApplyWindow', true), ...
                PitchContour(8000, 'MinF0', 60, 'MaxF0', 350), ...
                FeaturePipeline({MFCC(8000), DeltaFeatures('Width', 3, 'Order', 2)})};
            for k = 1:numel(transforms)
                t = transforms{k};
                cfg = t.config();
                viaClass = feval("OceanNLP.Speech.Features." + string(cfg.Type) + ".fromConfig", cfg);
                viaFactory = OceanNLP.Speech.Models.RecognizerFactory.featuresFromConfig(cfg);
                testCase.verifyEqual(viaClass.config(), cfg, char(cfg.Type));
                testCase.verifyEqual(viaFactory.config(), cfg, char(cfg.Type));
                testCase.verifyEqual(viaFactory.forward(w).Values, t.forward(w).Values, char(cfg.Type));
            end
        end
    end
end
