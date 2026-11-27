classdef BertEncoderTest < OceanTest.BaseTestCase
    % BERTENCODERTEST  Unit tests for OceanNLP.Embedding.BertEncoder.
    %
    % END-TO-END HAND RECOMPUTE (d=2, 1 transformer layer, numHeads=1,
    % d_ff=2, VocabSize=6): a synthetic vocab.txt/Weights struct is built
    % in TestClassSetup. The "expected" embedding is computed by
    % INDEPENDENTLY assembling a SEPARATE WordPieceTokenizer +
    % TokenEmbedding + MultiHeadAttention + 2x LayerNormLayer +
    % TransformerEncoderBlock object graph from the SAME weight values
    % (never by reading BertEncoder's own internal/private state) and
    % running tokenize -> wrap [CLS]/[SEP] -> encode -> embed -> encoder
    % block -> mean-pool by hand in this test -- exactly the "composing
    % already-verified parts, re-verified for the wiring itself" discipline
    % the master plan requires (each of those sub-parts already has its
    % own dedicated hand-worked test file).

    properties
        VocabFile
        Weights
    end

    methods (TestClassSetup)
        function buildFixture(testCase)
            tokens = ["[PAD]"; "[UNK]"; "[CLS]"; "[SEP]"; "cat"; "dog"];
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
            W.TokEmbedding.WordEmbeddings      = [0 0; 0 0; 1 1; 2 2; 3 0; 0 3];   % (6 x 2), id0..id5
            W.TokEmbedding.PositionEmbeddings  = [10 0; 0 10; 10 10; 5 5; 1 1];    % (5 x 2), pos0..pos4
            W.TokEmbedding.TokenTypeEmbeddings = [100 0; 0 100];                   % (2 x 2), seg0/seg1
            W.TokEmbedding.LayerNormGamma = [1.1; 0.9];   % non-trivial, exercises scale/shift (see BertEncoder.m WEIGHTS SCHEMA note)
            W.TokEmbedding.LayerNormBeta  = [0.05; -0.05];

            L.WQ = [1 0.5; 0.2 1];       L.bQ = [0.1; -0.1];
            L.WK = [0.9 -0.3; 0.4 0.8];  L.bK = [0; 0.2];
            L.WV = eye(2);               L.bV = [0; 0];
            L.WO = [0.7 0.3; 0.1 0.9];   L.bO = [0.05; -0.05];
            L.NumHeads = 1;
            L.W1 = [0.5 0.5; -0.2 0.3];  L.b1 = [0; 0.1];
            L.W2 = [0.6 -0.4; 0.2 0.9];  L.b2 = [0; 0];
            L.LN1Gamma = [1.2; 0.8]; L.LN1Beta = [0.1; -0.1];
            L.LN2Gamma = [1.0; 1.0]; L.LN2Beta = [0.0; 0.0];
            W.Layers = L;

            testCase.Weights = W;
        end
    end

    methods (Test)
        function testForwardBeforeFitErrors(testCase)
            import OceanNLP.Embedding.BertEncoder
            enc = BertEncoder('Weights', testCase.Weights, 'VocabFile', testCase.VocabFile);
            testCase.verifyFalse(enc.isFitted());
            testCase.verifyError(@() enc.forward("cat dog"), ...
                'OceanNLP:Embedding:BertEncoder:NotFitted');
        end

        function testFitIsRequiredNoOpThatFlipsIsFitted(testCase)
            import OceanNLP.Embedding.BertEncoder
            enc = BertEncoder('Weights', testCase.Weights, 'VocabFile', testCase.VocabFile);
            testCase.verifyWarningFree(@() enc.fit(["cat dog", "a totally different corpus"]));
            testCase.verifyTrue(enc.isFitted());
        end

        function testAssembledDimensionsReadFromWeights(testCase)
            import OceanNLP.Embedding.BertEncoder
            enc = BertEncoder('Weights', testCase.Weights, 'VocabFile', testCase.VocabFile);
            testCase.verifyEqual(enc.HiddenDim, 2);
            testCase.verifyEqual(enc.NumLayers, 1);
            testCase.verifyEqual(enc.embeddingDim(), 2);
        end

        function testNameIncludesModelName(testCase)
            import OceanNLP.Embedding.BertEncoder
            enc = BertEncoder('Weights', testCase.Weights, 'VocabFile', testCase.VocabFile, ...
                'ModelName', "test-bert");
            testCase.verifyEqual(enc.name(), "bert:test-bert");
        end

        function testEndToEndHandRecomputeMatchesForwardExactly(testCase)
            import OceanNLP.Embedding.BertEncoder
            import OceanNLP.Embedding.Tokenize.WordPieceTokenizer
            import OceanNLP.Embedding.Modules.TokenEmbedding
            import OceanNLP.Embedding.Modules.MultiHeadAttention
            import OceanNLP.Embedding.Modules.TransformerEncoderBlock
            import OceanDL.Norm.LayerNormLayer

            enc = BertEncoder('Weights', testCase.Weights, 'VocabFile', testCase.VocabFile, ...
                'ModelName', "test-bert");
            enc.fit("cat dog");
            E = enc.forward("cat dog");

            % --- Independent parallel assembly (separate objects, same weights) ---
            W = testCase.Weights;
            L = W.Layers(1);

            tok2 = WordPieceTokenizer('VocabFile', testCase.VocabFile);
            subTokens = tok2.forward("cat dog");                    % ["cat" "dog"], whole-word matches
            wrapped   = ["[CLS]", subTokens, "[SEP]"];               % (1 x 4)
            ids       = tok2.encode(wrapped);                        % 0-indexed
            segIds    = zeros(1, numel(ids));

            tokEmbed2 = TokenEmbedding(W.TokEmbedding.WordEmbeddings, ...
                W.TokEmbedding.PositionEmbeddings, W.TokEmbedding.TokenTypeEmbeddings);
            X0raw = tokEmbed2.forward(ids, segIds);                    % (d x T), BEFORE embedding LayerNorm
            % BertEncoder applies an embedding-level LayerNorm (Epsilon=1e-12,
            % matching real BERT) right after TokEmbed, before the first
            % block -- see BertEncoder.m ASSEMBLY note. Every LayerNormLayer
            % built here uses the SAME Epsilon=1e-12 BertEncoder's own
            % constructor hardcodes, so this independent recompute matches
            % forward() at tight tolerance, not just "close."
            embLN2 = LayerNormLayer(2, 'Epsilon', 1e-12);
            embLN2.Gamma = W.TokEmbedding.LayerNormGamma; embLN2.Beta = W.TokEmbedding.LayerNormBeta;
            X0 = embLN2.normalize(X0raw, 'train');                     % (d x T)

            mha2 = MultiHeadAttention(L.WQ, L.bQ, L.WK, L.bK, L.WV, L.bV, L.WO, L.bO, L.NumHeads);
            ln1_2 = LayerNormLayer(2, 'Epsilon', 1e-12); ln1_2.Gamma = L.LN1Gamma; ln1_2.Beta = L.LN1Beta;
            ln2_2 = LayerNormLayer(2, 'Epsilon', 1e-12); ln2_2.Gamma = L.LN2Gamma; ln2_2.Beta = L.LN2Beta;
            block2 = TransformerEncoderBlock(mha2, ln1_2, ln2_2, L.W1, L.b1, L.W2, L.b2);

            X1 = block2.forward(X0);                                  % 1 layer, (d x T)
            expectedEmbedding = mean(X1, 2)';                          % mean-pool over T -> (1 x d)

            testCase.verifyEqual(E, expectedEmbedding, 'AbsTol', 1e-9);
        end

        function testMultipleDocumentsStackIntoNxD(testCase)
            % Uses its OWN local d=4 fixture, not the shared class-level
            % d=2 one (testCase.Weights) -- a 2-dimensional LayerNorm output
            % has only ONE bit of information left after normalization
            % (mean-centering two values forces them to be negatives of each
            % other; unit-variance normalization then fixes the magnitude),
            % so at d=2 most/all distinct documents collapse onto the same
            % of only two possible directions post-LayerNorm -- confirmed
            % empirically (2026-09-18): with the shared d=2 fixture, "cat
            % dog" vs "cat cat" AND "dog dog" vs "cat cat" both collapsed to
            % an identical [1,-1] output. This is a real mathematical
            % property of low-dimensional LayerNorm, not a BertEncoder bug
            % (already proven correct at d=128 against real trained
            % bert-tiny weights -- see master plan §5/§7). This test's job
            % is purely mechanical (shape + row-distinctness), so a wider,
            % simple identity-based local fixture is used instead of
            % enlarging the shared fixture every other test in this file
            % depends on.
            import OceanNLP.Embedding.BertEncoder

            tokens = ["[PAD]"; "[UNK]"; "[CLS]"; "[SEP]"; "cat"; "dog"];
            vocabPath = [tempname() '.txt'];
            fid = fopen(vocabPath, 'w');
            testCase.assertNotEqual(fid, -1, 'Failed to create temp vocab file.');
            for i = 1:numel(tokens)
                fprintf(fid, '%s\n', tokens(i));
            end
            fclose(fid);
            testCase.addTeardown(@() delete(vocabPath));

            d = 4;
            W = struct();
            W.TokEmbedding.WordEmbeddings = [zeros(2,d); eye(4)];   % PAD,UNK,CLS,SEP,cat,dog -> one-hot for cat/dog
            W.TokEmbedding.PositionEmbeddings = 0.1 * eye(5, d);
            W.TokEmbedding.TokenTypeEmbeddings = zeros(2, d);
            W.TokEmbedding.LayerNormGamma = ones(d, 1);
            W.TokEmbedding.LayerNormBeta = zeros(d, 1);
            L.WQ = eye(d); L.bQ = zeros(d,1);
            L.WK = eye(d); L.bK = zeros(d,1);
            L.WV = eye(d); L.bV = zeros(d,1);
            L.WO = eye(d); L.bO = zeros(d,1);
            L.NumHeads = 2;
            L.W1 = eye(d); L.b1 = zeros(d,1);
            L.W2 = eye(d); L.b2 = zeros(d,1);
            L.LN1Gamma = ones(d,1); L.LN1Beta = zeros(d,1);
            L.LN2Gamma = ones(d,1); L.LN2Beta = zeros(d,1);
            W.Layers = L;

            enc = BertEncoder('Weights', W, 'VocabFile', vocabPath);
            enc.fit(["cat dog", "cat cat"]);
            E = enc.forward(["cat dog", "cat cat"]);
            testCase.verifySize(E, [2 d]);
            testCase.verifyNotEqual(E(1,:), E(2,:));   % different documents -> different embeddings
        end

        function testMissingSpecialTokensErrorsAtConstruction(testCase)
            % A vocab.txt without literal "[CLS]"/"[SEP]" entries must fail
            % FAST at construction, not silently substitute [UNK].
            import OceanNLP.Embedding.BertEncoder
            tokens = ["[PAD]"; "[UNK]"; "cat"; "dog"];   % no [CLS]/[SEP]
            badVocabPath = [tempname() '.txt'];
            fid = fopen(badVocabPath, 'w');
            testCase.assertNotEqual(fid, -1, 'Failed to create temp vocab file.');
            for i = 1:numel(tokens)
                fprintf(fid, '%s\n', tokens(i));
            end
            fclose(fid);
            testCase.addTeardown(@() delete(badVocabPath));

            W = testCase.Weights;
            W.TokEmbedding.WordEmbeddings = W.TokEmbedding.WordEmbeddings(1:4, :);   % match 4-token vocab

            testCase.verifyError(@() BertEncoder('Weights', W, 'VocabFile', badVocabPath), ...
                'OceanNLP:Embedding:BertBackbone:MissingSpecialTokens');
        end
    end
end
