classdef DenseRankerTest < OceanTest.BaseTestCase
    % DENSERANKERTEST  Unit tests for OceanNLP.Retrieval.DenseRanker.
    %
    % SCOPE, DELIBERATELY LIMITED TO MECHANICS (documented, not an
    % oversight): this test file verifies shapes, sort correctness, error
    % paths, and IsSymbolic discrimination ONLY. It does NOT assert
    % semantic-quality/topical-clustering behavior (e.g. "a cat query
    % ranks the cat document first") against this synthetic, UNTRAINED
    % encoder -- random/hand-picked weights produce no learned semantic
    % structure, so asserting topical clustering here would assert
    % something not actually true, the same honesty discipline the master
    % plan's Execution Log already applied to this exact class (§7, Phase
    % 2: "honestly flagged that random/untrained weights produce no
    % semantic clustering -- correctly scoped as expected, not a defect").
    % Real semantic-quality validation is deferred to Phase 5's external
    % validation against real bert-tiny weights (see the same precedent as
    % the earlier NLP build's RegexTokenizer dead-code documentation, rather
    % than faking a test that would pass for the wrong reason).

    properties
        VocabFile
        Weights
        Corpus
        % NOTE: Encoder is intentionally NOT built here as a shared
        % TestClassSetup property. BertEncoder is a handle class, and
        % DenseRanker.fit() calls obj.Encoder.fit(corpus) internally,
        % which mutates the SHARED encoder's isFitted() state -- a
        % TestClassSetup-level Encoder would leak fitted state across test
        % methods (methods run in the same class instance share one
        % object). A FRESH BertEncoder is built per test method instead
        % (see freshEncoder(), called explicitly where an unfitted-state
        % guarantee matters), same isolation discipline the
        % matlab-write-test skill's fixtures guidance recommends for
        % mutable shared state.
    end

    methods (TestClassSetup)
        function buildFixture(testCase)
            tokens = ["[PAD]"; "[UNK]"; "[CLS]"; "[SEP]"; "cat"; "dog"; "money"];
            vocabPath = [tempname() '.txt'];
            fid = fopen(vocabPath, 'w');
            testCase.assertNotEqual(fid, -1, 'Failed to create temp vocab file.');
            for i = 1:numel(tokens)
                fprintf(fid, '%s\n', tokens(i));
            end
            fclose(fid);
            testCase.addTeardown(@() delete(vocabPath));
            testCase.VocabFile = vocabPath;

            W = struct();
            W.TokEmbedding.WordEmbeddings      = [0 0; 0 0; 1 1; 2 2; 3 0; 0 3; 1 -1];
            W.TokEmbedding.PositionEmbeddings  = [10 0; 0 10; 10 10; 5 5; 1 1; 2 2];
            W.TokEmbedding.TokenTypeEmbeddings = [100 0; 0 100];
            W.TokEmbedding.LayerNormGamma = [1; 1];
            W.TokEmbedding.LayerNormBeta  = [0; 0];
            L.WQ = eye(2); L.bQ = [0;0]; L.WK = eye(2); L.bK = [0;0];
            L.WV = eye(2); L.bV = [0;0]; L.WO = eye(2); L.bO = [0;0];
            L.NumHeads = 1;
            L.W1 = eye(2); L.b1 = [0;0]; L.W2 = eye(2); L.b2 = [0;0];
            L.LN1Gamma = [1;1]; L.LN1Beta = [0;0];
            L.LN2Gamma = [1;1]; L.LN2Beta = [0;0];
            W.Layers = L;

            testCase.Weights = W;
            testCase.Corpus = ["cat cat", "dog dog", "money money"];
        end
    end

    methods (Access = private)
        function enc = freshEncoder(testCase)
            import OceanNLP.Embedding.BertEncoder
            enc = BertEncoder('Weights', testCase.Weights, 'VocabFile', testCase.VocabFile);
        end
    end

    methods (Test)
        function testEncoderIsRequiredNoDefault(testCase)
            import OceanNLP.Retrieval.DenseRanker
            testCase.verifyError(@() DenseRanker(), 'MATLAB:nonExistentField');
        end

        function testRankBeforeFitErrors(testCase)
            import OceanNLP.Retrieval.DenseRanker
            ranker = DenseRanker('Encoder', testCase.freshEncoder());
            testCase.verifyError(@() ranker.rank("cat"), ...
                'OceanNLP:Retrieval:DenseRanker:NotFitted');
        end

        function testReturnShapeIsRowVectorsMatchingCorpusSize(testCase)
            import OceanNLP.Retrieval.DenseRanker
            ranker = DenseRanker('Encoder', testCase.freshEncoder());
            ranker.fit(testCase.Corpus);

            [rankedIdx, scores] = ranker.rank("cat");

            N = numel(testCase.Corpus);
            testCase.verifySize(rankedIdx, [1 N]);
            testCase.verifySize(scores, [1 N]);
            testCase.verifyEqual(sort(rankedIdx), 1:N);   % a permutation of 1:N
            testCase.verifyTrue(issorted(scores, 'descend'));
        end

        function testIsFittedDelegatesToEncoder(testCase)
            import OceanNLP.Retrieval.DenseRanker
            ranker = DenseRanker('Encoder', testCase.freshEncoder());
            testCase.verifyFalse(ranker.isFitted());
            ranker.fit(testCase.Corpus);
            testCase.verifyTrue(ranker.isFitted());
        end

        function testNameIncludesEncoderName(testCase)
            import OceanNLP.Retrieval.DenseRanker
            enc = testCase.freshEncoder();
            ranker = DenseRanker('Encoder', enc);
            testCase.verifyEqual(ranker.name(), "dense-" + enc.name() + "-cosine");
        end

        function testCorpusMatrixIsDenseNotSparse(testCase)
            import OceanNLP.Retrieval.DenseRanker
            ranker = DenseRanker('Encoder', testCase.freshEncoder());
            ranker.fit(testCase.Corpus);
            testCase.verifyFalse(issparse(ranker.CorpusMatrix));
        end

        function testIsSymbolicFalseAndComposedPartsAreHonest(testCase)
            import OceanNLP.Retrieval.DenseRanker
            ranker = DenseRanker('Encoder', testCase.freshEncoder());
            meta = ranker.getMetadata();
            testCase.verifyFalse(meta.IsSymbolic);
            testCase.verifyEmpty(meta.SymbolicLatex);
        end

        function testGetPipelineSpecComposesFromIncludesEncoderClassDynamically(testCase)
            import OceanNLP.Retrieval.DenseRanker
            ranker = DenseRanker('Encoder', testCase.freshEncoder());
            spec = ranker.getPipelineSpec();
            testCase.verifyEqual(spec.Kind, "VIRTUAL");
            testCase.verifyTrue(any(strcmp(spec.ComposesFrom, 'OceanNLP.Embedding.BertEncoder')));
            testCase.verifyTrue(any(strcmp(spec.ComposesFrom, 'OceanStats.Similarity.CosineSimilarity')));
        end
    end
end
