classdef AveragePrecisionTest < OceanTest.BaseTestCase
    % AVERAGEPRECISIONTEST  Unit tests for OceanStats.Metrics.Retrieval.AveragePrecision.
    %
    % Shares the PrecisionAtKTest fixture: rankedIdx = [4 1 2 5 3],
    % relevance = [2 0 1 0 3], binary-relevant set {doc1, doc3, doc5}, R = 3.
    % isRelevant sequence aligned with rankedIdx: [0 1 0 1 1].
    %
    % HAND DERIVATION (independent of the source's own running-cumsum
    % formulation -- computed here via the "average of Precision@k at each
    % relevant rank" definition directly):
    %   relevant hit at rank 2: Precision@2 = (1 relevant in top 2)/2 = 1/2
    %   relevant hit at rank 4: Precision@4 = (2 relevant in top 4)/4 = 2/4
    %   relevant hit at rank 5: Precision@5 = (3 relevant in top 5)/5 = 3/5
    %   AP = (1/2 + 2/4 + 3/5) / 3

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
        function testHandWorkedAveragePrecision(testCase)
            import OceanStats.Metrics.Retrieval.AveragePrecision
            m = AveragePrecision();
            score = m.forward(testCase.RankedIdx, testCase.Relevance);
            expected = (1/2 + 2/4 + 3/5) / 3;
            testCase.verifyEqual(score, expected, 'AbsTol', 1e-12);
        end

        function testPerfectRankingGivesApOfOne(testCase)
            % All relevant docs ranked first, in any order -> AP = 1.
            import OceanStats.Metrics.Retrieval.AveragePrecision
            m = AveragePrecision();
            rankedIdx = [1 3 5 2 4];   % relevant docs (1,3,5) all lead
            relevance = [2 0 1 0 3];
            score = m.forward(rankedIdx, relevance);
            testCase.verifyEqual(score, 1.0, 'AbsTol', 1e-12);
        end

        function testZeroRelevantDocsAtAllReturnsNaNNotZero(testCase)
            import OceanStats.Metrics.Retrieval.AveragePrecision
            m = AveragePrecision();
            rankedIdx = [1 2 3];
            relevance = [0 0 0];
            score = m.forward(rankedIdx, relevance);
            testCase.verifyTrue(isnan(score));
        end

        function testConstructorTakesNoConfig(testCase)
            % Unlike PrecisionAtK/RecallAtK/NDCGAtK, AveragePrecision has no
            % K (considers the whole ranked list) -- the zero-arg
            % constructor must succeed.
            import OceanStats.Metrics.Retrieval.AveragePrecision
            testCase.verifyWarningFree(@() AveragePrecision());
        end

        function testNameReturnsRegistryKey(testCase)
            import OceanStats.Metrics.Retrieval.AveragePrecision
            m = AveragePrecision();
            testCase.verifyEqual(m.name(), "ap");
        end
    end
end
