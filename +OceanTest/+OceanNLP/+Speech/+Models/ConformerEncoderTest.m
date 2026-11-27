classdef ConformerEncoderTest < OceanTest.BaseTestCase
    % CONFORMERENCODERTEST  Encoder contract of Models.Encoders.ConformerEncoder: lengths
    %   (Conv1d formula), padded batch == per-utterance, parameter gradients (finite
    %   differences), config / factory round trip (runtime registration), errors.

    methods (TestClassSetup)
        function registerType(~)
            OceanNLP.Speech.Models.RecognizerFactory.register("encoder", "ConformerEncoder", ...
                @OceanNLP.Speech.Models.Encoders.ConformerEncoder.fromConfig);
        end
    end

    methods (Test)
        function testLengthsAndPaddingInvariance(testCase)
            for pos = ["rope", "abs"]
                enc = OceanNLP.Speech.Models.Encoders.ConformerEncoder(5, 'Dim', 8, 'NumHeads', 2, 'NumBlocks', 2, ...
                    'ConvKernel', 3, 'PosEnc', pos, 'Dropout', 0.2, 'Seed', 3);
                enc.eval();
                lens = [23 14 5 0];
                s = RandStream('mt19937ar', 'Seed', 1);
                X = zeros(5, 4, 23);
                for n = 1:4, X(:, n, 1:lens(n)) = randn(s, 5, 1, lens(n)); end
                [H, l2] = enc.forward(X, lens);
                expect = lens;
                for k = 1:2, expect = floor((expect + 2 - 3) / 2) + 1; end
                expect(lens == 0) = 0;
                testCase.verifyEqual(l2, expect);
                testCase.verifyEqual(enc.outputLength(lens), expect);
                testCase.verifyEqual(size(H, 1), enc.OutputDim);
                testCase.verifyTrue(all(isfinite(H.Data), 'all'));
                for n = 1:3
                    [h1, ln] = enc.forward(X(:, n, 1:lens(n)), lens(n));
                    testCase.verifyEqual(ln, l2(n));
                    testCase.verifyEqual(H.Data(:, n, 1:ln), h1.Data, 'AbsTol', 1e-11, char(pos));
                    testCase.verifyEqual(nnz(H.Data(:, n, ln+1:end)), 0);
                end
            end
        end

        function testParameterGradients(testCase)
            enc = OceanNLP.Speech.Models.Encoders.ConformerEncoder(3, 'Dim', 4, 'NumHeads', 2, 'NumBlocks', 1, ...
                'FFMult', 1, 'ConvKernel', 3, 'Seed', 2);
            X = randn(RandStream('mt19937ar', 'Seed', 2), 3, 2, 9); lens = [9 6];
            err = OceanTest.OceanNLP.Speech.Models.EncodersTest.paramGradError(enc, @(m) m.forward(X, lens), 4);
            testCase.verifyLessThan(err, 1e-6);
        end

        function testConfigRoundTrip(testCase)
            enc = OceanNLP.Speech.Models.Encoders.ConformerEncoder(6, 'Dim', 8, 'NumHeads', 4, 'NumBlocks', 2, ...
                'FFMult', 2, 'ConvKernel', 5, 'SubsampleStrides', [2 1 2], 'SubsampleKernel', 5, 'PosEnc', "abs", ...
                'Dropout', 0.1, 'AttnDropout', 0.05, 'Seed', 7);
            cfg = enc.config();
            e2 = OceanNLP.Speech.Models.RecognizerFactory.encoderFromConfig(jsondecode(jsonencode(cfg)));
            testCase.verifyClass(e2, class(enc));
            testCase.verifyEqual(e2.config(), cfg);
            X = randn(6, 2, 17); lens = [17 11];
            enc.eval(); e2.eval();
            testCase.verifyEqual(e2.forward(X, lens).Data, enc.forward(X, lens).Data);    % same seeds
        end

        function testErrors(testCase)
            E = @(varargin) OceanNLP.Speech.Models.Encoders.ConformerEncoder(varargin{:});
            testCase.verifyError(@() E(4, 'Dim', 6, 'NumHeads', 4), 'OceanNLP:Speech:Models:ConformerEncoder:HeadSplit');
            testCase.verifyError(@() E(4, 'SubsampleStrides', zeros(1, 0)), 'OceanNLP:Speech:Models:ConformerEncoder:Subsample');
            enc = E(4, 'Dim', 8, 'NumHeads', 2, 'NumBlocks', 1);
            testCase.verifyError(@() enc.forward(randn(5, 1, 6), 6), 'OceanNLP:Speech:Models:ConformerEncoder:InputDim');
            testCase.verifyError(@() enc.forward(randn(4, 2, 6), [6 7]), 'OceanNLP:Speech:Models:ConformerEncoder:Lengths');
        end
    end
end
