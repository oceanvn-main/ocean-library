classdef BM25WeightingTest < OceanTest.BaseTestCase
    % BM25WEIGHTINGTEST  Unit tests for OceanStats.Data.Weighting.BM25Weighting.
    %
    % Includes a REGRESSION test (testSingleDocumentRowDoesNotCrash) for the
    % bug fixed during Phase 3: forward() on a single-row (1-document) count
    % matrix crashed because find() on a (1 x V) row-vector matrix returns
    % i/j/v as ROW vectors (a documented MATLAB special case; every other
    % matrix shape returns columns). Without forcing i=i(:); j=j(:); v=v(:);
    % the elementwise .* between obj.IDF(j)' (a column) and satTf_v (which
    % stayed a row) implicitly broadcast to a dense (nnz x nnz) matrix and
    % sparse(i,j,w_v,...) failed/produced garbage. This test must never be
    % weakened or removed so the bug cannot silently regress.
    %
    % The 3-document hand-worked example was independently computed from
    % the documented BM25 formula (idf_t = log((N-df_t+0.5)/(df_t+0.5)+1),
    % w(d,t) = idf_t * tf / (tf + K1*(1-B+B*|d|/avgdl))) directly in
    % MATLAB using raw arithmetic — not by calling BM25Weighting itself.

    properties
        Counts     % (3 x 3) hand-picked count matrix: vocab = {cat, dog, bird}
        ExpectedW  % hand-derived full BM25 weight matrix, K1=1.2, B=0.75 defaults
    end

    methods (TestClassSetup)
        function buildHandWorkedExample(testCase)
            testCase.Counts = [1 1 0; ...   % Doc1: cat=1, dog=1
                                0 2 1; ...   % Doc2: dog=2, bird=1
                                0 0 2];      % Doc3: bird=2

            testCase.ExpectedW = [1.04170831009521,  0.499176268302368, 0; ...
                                   0, 0.598186437221845, 0.420817202929321; ...
                                   0, 0, 0.673307524686914] / 2.2;
        end
    end

    methods (Test)
        function testHandWorkedThreeDocumentExample(testCase)
            import OceanStats.Data.Weighting.BM25Weighting
            bm25 = BM25Weighting();  % K1=1.2, B=0.75 defaults (Elasticsearch/Lucene)
            bm25.fit(testCase.Counts);
            W = bm25.forward(testCase.Counts);

            testCase.verifyEqual(full(W), testCase.ExpectedW, 'AbsTol', 1e-9);
            testCase.verifyEqual(full(W(1, 1)), 1.041708 / 2.2, 'AbsTol', 1e-6); % w(Doc1,cat)
            testCase.verifyEqual(full(W(3, 3)), 0.673308 / 2.2, 'AbsTol', 1e-6); % w(Doc3,bird)
        end

        function testSingleDocumentRowDoesNotCrash(testCase)
            % REGRESSION for the Phase-3 1-row bugfix — see class help.
            import OceanStats.Data.Weighting.BM25Weighting
            counts1 = [2 1 0 3];

            bm25 = BM25Weighting();
            bm25.fit(counts1);
            W1 = bm25.forward(counts1);

            expected = [0.395562849621199, 0.287682072451781, 0, 0.452071828138513] / 2.2;
            testCase.verifyEqual(full(W1), expected, 'AbsTol', 1e-9);
            testCase.verifyTrue(issparse(W1));
            testCase.verifySize(W1, [1 4]);
        end

        function testMultiRowSparsityPreserved(testCase)
            % Sparsity is a correctness requirement (Master Plan §3.2): the
            % triplet-based nonzero-only transform must never densify.
            import OceanStats.Data.Weighting.BM25Weighting
            rng(11);
            sparseCounts = round(sprand(30, 150, 0.04) * 5);

            bm25 = BM25Weighting();
            bm25.fit(sparseCounts);
            W = bm25.forward(sparseCounts);

            testCase.verifyTrue(issparse(W));
            testCase.verifyEqual(nnz(W), nnz(sparseCounts));
        end

        function testNotFittedErrors(testCase)
            import OceanStats.Data.Weighting.BM25Weighting
            bm25 = BM25Weighting();
            testCase.verifyError(@() bm25.forward(testCase.Counts), ...
                'OceanStats:Data:Weighting:BM25Weighting:NotFitted');
        end

        function testVocabMismatchErrors(testCase)
            import OceanStats.Data.Weighting.BM25Weighting
            bm25 = BM25Weighting();
            bm25.fit(testCase.Counts);
            testCase.verifyError(@() bm25.forward([1 2 3 4]), ...
                'OceanStats:Data:Weighting:BM25Weighting:VocabMismatch');
        end

        function testStateDictLoadStateDictRoundTrip(testCase)
            import OceanStats.Data.Weighting.BM25Weighting
            b1 = BM25Weighting('K1', 1.5, 'B', 0.6);
            b1.fit(testCase.Counts);

            s = b1.stateDict();
            b2 = BM25Weighting();
            b2.loadStateDict(s);

            testCase.verifyEqual(full(b2.forward(testCase.Counts)), ...
                full(b1.forward(testCase.Counts)), 'AbsTol', 1e-12);
            testCase.verifyEqual(b2.K1, 1.5);
            testCase.verifyEqual(b2.B, 0.6);
        end
    end
end
