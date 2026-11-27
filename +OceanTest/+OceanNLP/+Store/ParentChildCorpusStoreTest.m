classdef ParentChildCorpusStoreTest < OceanTest.BaseTestCase
    methods (Test)
        function testChildLookupAndParentExpansion(testCase)
            s=OceanNLP.Store.ParentChildCorpusStore(); s.fitHierarchy(["P1","P2"],["a","b","c"],[1 1 2]);
            testCase.verifyEqual(s.getChunk([1 3]),["a","c"]); [p,ids]=s.getParents([1 2 3]);
            testCase.verifyEqual(ids,[1 2]); testCase.verifyEqual(p,["P1","P2"]);
        end
    end
end
