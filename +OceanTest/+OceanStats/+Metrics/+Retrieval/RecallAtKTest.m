classdef RecallAtKTest < OceanTest.BaseTestCase
    % RECALLATKTEST  Unit tests for OceanStats.Metrics.Retrieval.RecallAtK.
    %
    % Shares the PrecisionAtKTest fixture (see that file's header for the
    % full derivation): rankedIdx = [4 1 2 5 3], relevance = [2 0 1 0 3],
    % binary-relevant set {doc1, doc3, doc5}, R = 3.

    properties
        Relevance
        RankedIdx
    end

    methods (TestClassSetup)
        function buildFixture(testCase)
            testCase.Relevance = [2 0 1 0 3];
            testCase.RankedIdx = [4 1 2 5 3];
        end
    end

    methods (Test)
        function testHandWorkedRecallAt3(testCase)
            % Top 3 of rankedIdx = [4 1 2] -> 1 relevant hit (doc1) out of
            % R = 3 total relevant docs -> 1/3.
            import OceanStats.Metrics.Retrieval.RecallAtK
            m = RecallAtK('K', 3);
            score = m.forward(testCase.RankedIdx, testCase.Relevance);
            testCase.verifyEqual(score, 1/3, 'AbsTol', 1e-12);
        end

        function testRecallReachesOneWhenKCoversAllDocs(testCase)
            import OceanStats.Metrics.Retrieval.RecallAtK
            m = RecallAtK('K', 5);
            score = m.forward(testCase.RankedIdx, testCase.Relevance);
            testCase.verifyEqual(score, 1.0, 'AbsTol', 1e-12);
        end

        function testZeroRelevantDocsAtAllReturnsNaNNotZero(testCase)
            % EDGE CASE: nnz(relevance) == 0 -> the query itself is
            % undefined -> NaN, never a fabricated 0.
            import OceanStats.Metrics.Retrieval.RecallAtK
            m = RecallAtK('K', 3);
            rankedIdx = [1 2 3];
            relevance = [0 0 0];
            score = m.forward(rankedIdx, relevance);
            testCase.verifyTrue(isnan(score));
        end

        function testKAlwaysInDenominatorPolicyMatchesPrecisionAtK(testCase)
            % Same "missing positions count as non-relevant, K unchanged"
            % handling as PrecisionAtK, but RecallAtK's denominator is
            % nnz(relevance), not K -- confirm K only affects the
            % numerator's top-K truncation here.
            import OceanStats.Metrics.Retrieval.RecallAtK
            m = RecallAtK('K', 10);   % K > numel(rankedIdx)
            rankedIdx = [1 2];
            relevance = [1 0 1];     % doc3 relevant, never retrieved; R = 2
            score = m.forward(rankedIdx, relevance);
            testCase.verifyEqual(score, 1/2, 'AbsTol', 1e-12);
        end

        function testNameReturnsRegistryKey(testCase)
            import OceanStats.Metrics.Retrieval.RecallAtK
            m = RecallAtK('K', 1);
            testCase.verifyEqual(m.name(), "recall@k");
        end
    end
end
