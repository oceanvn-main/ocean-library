classdef ReciprocalRankFusionTest < OceanTest.BaseTestCase
    methods (Test)
        function testTopAgreementWins(testCase)
            [ids,s]=OceanNLP.Fusion.ReciprocalRankFusion('K',0).fuse({[1 2 3],[1 4 5]});
            testCase.verifyEqual(ids(1),1); testCase.verifyGreaterThan(s(1),s(2));
        end
    end
end
