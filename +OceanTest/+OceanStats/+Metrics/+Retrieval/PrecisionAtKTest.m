classdef PrecisionAtKTest < OceanTest.BaseTestCase
    % PRECISIONATKTEST  Unit tests for OceanStats.Metrics.Retrieval.PrecisionAtK.
    %
    % SHARED HAND-WORKED EXAMPLE (also used by RecallAtKTest/AveragePrecisionTest/
    % ReciprocalRankTest/NDCGAtKTest for a consistent cross-metric fixture):
    %   5 documents. relevance grades (indexed by ORIGINAL doc id):
    %     doc1=2 (relevant), doc2=0, doc3=1 (relevant), doc4=0, doc5=3 (relevant)
    %   ranking under test: rankedIdx = [4 1 2 5 3]
    %     rank1 = doc4 (grade 0, NOT relevant)
    %     rank2 = doc1 (grade 2, relevant)      <- first relevant hit
    %     rank3 = doc2 (grade 0, NOT relevant)
    %     rank4 = doc5 (grade 3, relevant)
    %     rank5 = doc3 (grade 1, relevant)
    %   Binary-relevant set (nonzero grade): {doc1, doc3, doc5}, R = 3.
    %   isRelevant sequence aligned with rankedIdx: [0 1 0 1 1]

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
        function testHandWorkedPrecisionAt3(testCase)
            % Top 3 of rankedIdx = [4 1 2] -> relevant hits: doc1 only -> 1/3.
            import OceanStats.Metrics.Retrieval.PrecisionAtK
            m = PrecisionAtK('K', 3);
            score = m.forward(testCase.RankedIdx, testCase.Relevance);
            testCase.verifyEqual(score, 1/3, 'AbsTol', 1e-12);
        end

        function testKAlwaysInDenominatorWhenRankedListShorterThanK(testCase)
            % POLICY: if numel(rankedIdx) < K, missing positions count as
            % non-relevant and the denominator STAYS K (not rescaled to
            % numel(rankedIdx)). Ranked list has only 2 entries; K = 5.
            import OceanStats.Metrics.Retrieval.PrecisionAtK
            m = PrecisionAtK('K', 5);

            rankedIdx = [1 2];              % only 2 documents returned
            relevance = [1 0 1];            % doc3 is relevant but never retrieved

            score = m.forward(rankedIdx, relevance);
            % topK = rankedIdx(1:min(5,2)) = [1 2]; numRelevant = 1 (doc1)
            % score = 1 / K = 1/5, NOT 1/2 (which a rescaled-denominator
            % implementation would incorrectly produce).
            testCase.verifyEqual(score, 1/5, 'AbsTol', 1e-12);
        end

        function testPrecisionAtKWithKLargerThanNoveralAllDocs(testCase)
            % K = 10 > N = 5 (full corpus), using the shared fixture: all 5
            % positions are scored, 3 are relevant, but the denominator is
            % still K = 10, not 5.
            import OceanStats.Metrics.Retrieval.PrecisionAtK
            m = PrecisionAtK('K', 10);
            score = m.forward(testCase.RankedIdx, testCase.Relevance);
            testCase.verifyEqual(score, 3/10, 'AbsTol', 1e-12);
        end

        function testPerfectPrecisionWhenAllTopKRelevant(testCase)
            import OceanStats.Metrics.Retrieval.PrecisionAtK
            m = PrecisionAtK('K', 2);
            rankedIdx = [1 3 2];
            relevance = [1 0 1];   % docs 1,3 relevant, doc2 not
            score = m.forward(rankedIdx, relevance);
            testCase.verifyEqual(score, 1.0, 'AbsTol', 1e-12);
        end

        function testKMustBePositiveIntegerAtConstruction(testCase)
            import OceanStats.Metrics.Retrieval.PrecisionAtK
            testCase.verifyError(@() PrecisionAtK('K', 0), 'MATLAB:validators:mustBePositive');
            testCase.verifyError(@() PrecisionAtK('K', 2.5), 'MATLAB:validators:mustBeInteger');
        end

        function testKIsRequiredNoDefault(testCase)
            % K has no default in the arguments block; omitting it errors
            % (MATLAB reports the missing required field as "nonExistentField",
            % verified empirically rather than assumed).
            import OceanStats.Metrics.Retrieval.PrecisionAtK
            testCase.verifyError(@() PrecisionAtK(), 'MATLAB:nonExistentField');
        end

        function testNameReturnsRegistryKey(testCase)
            import OceanStats.Metrics.Retrieval.PrecisionAtK
            m = PrecisionAtK('K', 1);
            testCase.verifyEqual(m.name(), "precision@k");
        end
    end
end
