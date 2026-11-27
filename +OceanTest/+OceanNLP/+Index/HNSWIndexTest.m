classdef HNSWIndexTest < OceanTest.BaseTestCase
    methods (Test)
        function testExactPointIsReturned(testCase)
            X = [0 0; 1 0; 0 2; 10 10];
            idx = OceanNLP.Index.HNSWIndex('MaxNeighbors', 3, 'EfConstruction', 8);
            idx.fit(X);
            [ids, scores] = idx.search([10 10], 1, 'EfSearch', 8);
            testCase.verifyEqual(ids(1), 4);
            testCase.verifyEqual(scores(1), 0);
        end
        function testSearchReturnsValidUniqueIds(testCase)
            X = randn(40, 5);
            idx = OceanNLP.Index.HNSWIndex('MaxNeighbors', 8, 'EfConstruction', 16);
            idx.fit(X);
            [ids, scores] = idx.search(X(7,:), 5, 'EfSearch', 16);
            testCase.verifyTrue(all(ids > 0));
            testCase.verifyEqual(numel(unique(ids)), 5);
            testCase.verifyTrue(all(diff(scores) <= 0));
        end
    end
end
