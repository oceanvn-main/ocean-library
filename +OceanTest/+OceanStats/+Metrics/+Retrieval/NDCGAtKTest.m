classdef NDCGAtKTest < OceanTest.BaseTestCase
    % NDCGATKTEST  Unit tests for OceanStats.Metrics.Retrieval.NDCGAtK.
    %
    % Shares the PrecisionAtKTest fixture: rankedIdx = [4 1 2 5 3],
    % relevance = [2 0 1 0 3] (GRADED, not binarized -- NDCG is the one
    % metric in this family that uses the raw grade).
    %
    % HAND DERIVATION (K = 3, written as an INDEPENDENT explicit expression
    % in this test -- not by calling the source's private dcgAtK_ helper):
    %   achievedGrades (aligned with rankedIdx, first 3 ranks) = [0 2 0]
    %     (doc4=0, doc1=2, doc2=0)
    %   DCG@3 = (2^0-1)/log2(2) + (2^2-1)/log2(3) + (2^0-1)/log2(4)
    %         =         0       +      3/log2(3)  +        0
    %   idealGrades = sort(relevance,'descend') = [3 2 1 0 0]; top 3 = [3 2 1]
    %   IDCG@3 = (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4)
    %          =        7        +      3/log2(3)  +      1/2
    %   NDCG@3 = DCG@3 / IDCG@3

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
        function testHandWorkedNDCGAt3(testCase)
            import OceanStats.Metrics.Retrieval.NDCGAtK
            m = NDCGAtK('K', 3);
            score = m.forward(testCase.RankedIdx, testCase.Relevance);

            expectedDcg  = (2^0 - 1) / log2(2) + (2^2 - 1) / log2(3) + (2^0 - 1) / log2(4);
            expectedIdcg = (2^3 - 1) / log2(2) + (2^2 - 1) / log2(3) + (2^1 - 1) / log2(4);
            expectedNdcg = expectedDcg / expectedIdcg;

            testCase.verifyEqual(score, expectedNdcg, 'AbsTol', 1e-12);
        end

        function testGradedNotBinarized(testCase)
            % Unlike PrecisionAtK/RecallAtK/AveragePrecision/ReciprocalRank,
            % NDCG must distinguish grade 3 from grade 1 -- two rankings
            % that put the SAME set of relevant docs in the top K but in
            % different grade order must score differently.
            import OceanStats.Metrics.Retrieval.NDCGAtK
            m = NDCGAtK('K', 2);

            relevance = [3 0 1];   % doc1 grade 3, doc3 grade 1
            scoreBestFirst  = m.forward([1 3 2], relevance);   % grade 3 then grade 1
            scoreWorstFirst = m.forward([3 1 2], relevance);   % grade 1 then grade 3

            testCase.verifyEqual(scoreBestFirst, 1.0, 'AbsTol', 1e-12);   % matches the ideal order
            testCase.verifyLessThan(scoreWorstFirst, scoreBestFirst);
        end

        function testZeroRelevantDocsAtAllReturnsZeroNotNaN(testCase)
            % EDGE CASE (deliberately DIFFERENT convention from
            % RecallAtK/AveragePrecision/ReciprocalRank's NaN choice, not
            % an inconsistency -- see class THEORY note): IDCG@K == 0
            % (every grade is 0) -> NDCG@K is defined to be 0, matching
            % trec_eval/pytrec_eval/scikit-learn's ndcg_score convention.
            import OceanStats.Metrics.Retrieval.NDCGAtK
            m = NDCGAtK('K', 3);
            rankedIdx = [1 2 3];
            relevance = [0 0 0];
            score = m.forward(rankedIdx, relevance);
            testCase.verifyEqual(score, 0, 'AbsTol', 0);
            testCase.verifyFalse(isnan(score));
        end

        function testKMustBePositiveIntegerAtConstruction(testCase)
            import OceanStats.Metrics.Retrieval.NDCGAtK
            testCase.verifyError(@() NDCGAtK('K', 0), 'MATLAB:validators:mustBePositive');
            testCase.verifyError(@() NDCGAtK('K', 1.5), 'MATLAB:validators:mustBeInteger');
        end

        function testNameReturnsRegistryKey(testCase)
            import OceanStats.Metrics.Retrieval.NDCGAtK
            m = NDCGAtK('K', 1);
            testCase.verifyEqual(m.name(), "ndcg@k");
        end
    end
end
