classdef TranscriptNormalizerTest < OceanTest.BaseTestCase
    % TRANSCRIPTNORMALIZERTEST  Punctuation set vs Unicode data; output vs jiwer pipeline.

    properties
        Fixture
    end

    methods (TestClassSetup)
        function loadFixture(testCase)
            f = fullfile(fileparts(mfilename('fullpath')), 'fixtures', 'normalizer_cases.json');
            testCase.Fixture = jsondecode(fileread(f));
        end
    end

    methods (Test)
        function testPunctuationSetEqualsUnicodeCategoryP(testCase)
            blocks = testCase.Fixture.blocks;
            codes = [];
            for k = 1:size(blocks, 1)
                codes = [codes, blocks(k, 1):blocks(k, 2)]; %#ok<AGROW>
            end
            got = codes(OceanNLP.Speech.Data.TranscriptNormalizer.isPunctuation(codes));
            testCase.verifyEqual(got(:), double(testCase.Fixture.punctuation(:)));
        end

        function testMatchesJiwerPipeline(testCase)
            n = OceanNLP.Speech.Data.TranscriptNormalizer();
            S = testCase.Fixture.strings;
            inputs = string({S.input});
            expected = string({S.jiwer});
            testCase.verifyEqual(n.forward(inputs), expected);
            testCase.verifyEqual(size(n.forward(reshape(inputs, [], 1))), [numel(inputs) 1]);
        end

        function testOptionsAndConfig(testCase)
            keep = OceanNLP.Speech.Data.TranscriptNormalizer('Lowercase', false, 'StripPunctuation', false);
            testCase.verifyEqual(keep.forward("  One,  TWO "), "One, TWO");
            cfg = keep.config();
            back = OceanNLP.Speech.Data.TranscriptNormalizer.fromConfig(cfg);
            testCase.verifyEqual(back.config(), cfg);
            % documented gap: CJK punctuation (outside the covered blocks) is kept
            n = OceanNLP.Speech.Data.TranscriptNormalizer();
            testCase.verifyEqual(n.forward("a" + char(12300) + "b"), "a" + char(12300) + "b");
            testCase.verifyError(@() n.forward(string(missing)), 'OceanNLP:Speech:Data:TranscriptNormalizer:Missing');
        end
    end
end
