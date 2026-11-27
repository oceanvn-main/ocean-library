classdef SearchQualityEvaluatorTest < OceanTest.BaseTestCase
    methods (Test)
        function testExactReferenceHasUnitRecall(testCase)
            X=randn(30,3); q=X(1:3,:); exact=OceanAlgo.Search.HNSWIndex('MaxNeighbors',29,'EfConstruction',32); exact.fit(X);
            approx=OceanAlgo.Search.HNSWIndex('MaxNeighbors',29,'EfConstruction',32); approx.fit(X);
            r=OceanAlgo.Search.SearchQualityEvaluator.evaluate(approx,exact,q,3);
            testCase.verifyEqual(r.RecallAtK,1);
        end
    end
end
