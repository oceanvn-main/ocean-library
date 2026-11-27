classdef FunctionRerankerTest < OceanTest.BaseTestCase
    methods (Test)
        function testCandidatesAreSortedByInjectedScore(testCase)
            r=OceanNLP.Retrieval.FunctionReranker(@(q,t) double(contains(t,q)));
            r.fit(["alpha","beta"]); [ids,s]=r.rerank("alpha",[10 11],["beta","alpha"]);
            testCase.verifyEqual(ids,[11 10]); testCase.verifyEqual(s,[1 0]);
        end
    end
end
