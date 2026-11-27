classdef ReciprocalRankTest < OceanTest.BaseTestCase
    % RECIPROCALRANKTEST  Unit tests for OceanStats.Metrics.Retrieval.ReciprocalRank.
    %
    % Shares the PrecisionAtKTest fixture: rankedIdx = [4 1 2 5 3],
    % relevance = [2 0 1 0 3]. First relevant hit is at rank 2 (doc1) ->
    % RR = 1/2.
    %
    % THE TWO DISTINCT EDGE CASES (deliberately NOT conflated -- this is
    % the easiest place in the whole metrics suite to get backwards):
    %   1. nnz(relevance) == 0 (no relevant docs defined at all) -> NaN
    %      (undefined query).
    %   2. nnz(relevance) > 0 but none of those relevant docs appear in
    %      rankedIdx -> RR = 0, a well-defined retrieval failure, NOT NaN.

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
        function testHandWorkedReciprocalRank(testCase)
            import OceanStats.Metrics.Retrieval.ReciprocalRank
            m = ReciprocalRank();
            score = m.forward(testCase.RankedIdx, testCase.Relevance);
            testCase.verifyEqual(score, 1/2, 'AbsTol', 1e-12);
        end

        function testFirstRelevantAtRankOneGivesRRofOne(testCase)
            import OceanStats.Metrics.Retrieval.ReciprocalRank
            m = ReciprocalRank();
            rankedIdx = [1 4 2 5 3];   % doc1 (relevant) is rank 1
            score = m.forward(rankedIdx, testCase.Relevance);
            testCase.verifyEqual(score, 1.0, 'AbsTol', 1e-12);
        end

        function testCase1_ZeroRelevantDocsAtAllIsUndefinedNaN(testCase)
            % Case 1: no relevant docs are defined for the query at all --
            % the query itself is undefined.
            import OceanStats.Metrics.Retrieval.ReciprocalRank
            m = ReciprocalRank();
            rankedIdx = [1 2 3];
            relevance = [0 0 0];
            score = m.forward(rankedIdx, relevance);
            testCase.verifyTrue(isnan(score));
        end

        function testCase2_RelevantDocsExistButNoneRetrievedIsWellDefinedZero(testCase)
            % Case 2: relevant docs ARE defined (doc1 is relevant), but the
            % ranking never surfaces it (rankedIdx only covers docs 2 and
            % 3) -- a genuine retrieval failure, well-defined RR = 0, and
            % MUST NOT be NaN (that would conflate it with Case 1).
            import OceanStats.Metrics.Retrieval.ReciprocalRank
            m = ReciprocalRank();
            rankedIdx = [2 3];        % doc1 never appears
            relevance = [1 0 0];      % doc1 is relevant
            score = m.forward(rankedIdx, relevance);
            testCase.verifyEqual(score, 0, 'AbsTol', 0);
            testCase.verifyFalse(isnan(score));
        end

        function testConstructorTakesNoConfig(testCase)
            import OceanStats.Metrics.Retrieval.ReciprocalRank
            testCase.verifyWarningFree(@() ReciprocalRank());
        end

        function testNameReturnsRegistryKey(testCase)
            import OceanStats.Metrics.Retrieval.ReciprocalRank
            m = ReciprocalRank();
            testCase.verifyEqual(m.name(), "rr");
        end
    end
end
