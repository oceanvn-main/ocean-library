classdef NeuralEncodersTest < OceanTest.BaseTestCase
    % NEURALENCODERSTEST  BertBackbone, key masking, LinearHead, PoolerHead,
    % CrossEncoder, ColBERTEncoder, BertEncoder NormalizeOutput.
    % Expected values are recomputed in this file from the definitions.

    properties
        VocabFile
        W
    end

    methods (TestMethodSetup)
        function makeFixture(testCase)
            F = OceanTest.OceanNLP.Embedding.TinyBertFixture;
            testCase.VocabFile = F.writeVocab();
            testCase.W = F.weights(1, 2);
            testCase.addTeardown(@() delete(testCase.VocabFile));
        end
    end

    methods (Test)
        % ---------------- heads ----------------
        function testLinearHeadHandWorked(testCase)
            h = OceanNLP.Embedding.Modules.LinearHead([1 2; 3 4; 5 6], [1; 0; -1]);
            testCase.verifyEqual(h.forward([1 0; 1 1]), [4 3; 7 4; 10 5], 'AbsTol', 1e-12);
            g = OceanNLP.Embedding.Modules.LinearHead([1 2; 3 4]);
            testCase.verifyFalse(g.HasBias);
            testCase.verifyEqual(g.forward([1; 1]), [3; 7], 'AbsTol', 1e-12);
        end

        function testLinearHeadErrors(testCase)
            testCase.verifyError(@() OceanNLP.Embedding.Modules.LinearHead(eye(2), [1; 2; 3]), ...
                'OceanNLP:Embedding:Modules:LinearHead:BiasSizeMismatch');
            h = OceanNLP.Embedding.Modules.LinearHead(eye(2));
            testCase.verifyError(@() h.forward([1; 2; 3]), 'OceanNLP:Embedding:Modules:LinearHead:InDimMismatch');
        end

        function testPoolerReadsOnlyClsColumn(testCase)
            p = OceanNLP.Embedding.Modules.PoolerHead(eye(2), [0; 0.5]);
            X = [1 100 -50; 2 100 -50];                 % only column 1 may matter
            testCase.verifyEqual(p.forward(X), [tanh(1); tanh(2.5)], 'AbsTol', 1e-12);
            testCase.verifyError(@() OceanNLP.Embedding.Modules.PoolerHead(ones(2, 3), [0; 0]), ...
                'OceanNLP:Embedding:Modules:PoolerHead:ShapeMismatch');
        end

        function testHeadsAreSymbolicWithLatex(testCase)
            testCase.verifyTrue(OceanNLP.Embedding.Modules.LinearHead(eye(2)).getMetadata().IsSymbolic);
            m = OceanNLP.Embedding.Modules.PoolerHead(eye(2), [0; 0]).getMetadata();
            testCase.verifyTrue(m.IsSymbolic);
            testCase.verifyTrue(contains(m.SymbolicLatex, "\tanh"));
        end

        % ---------------- attention key mask ----------------
        function testAttentionKeyMaskIgnoresMaskedKeys(testCase)
            bb = OceanNLP.Embedding.BertBackbone('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            mha = bb.Blocks{1}.Attention;
            rng(5);
            X = randn(4, 3);
            X2 = [X, randn(4, 1)];
            X3 = [X, 100 * randn(4, 1)];               % different content in the masked slot
            mask = [true true true false];
            Y2 = mha.forward(X2, mask);
            Y3 = mha.forward(X3, mask);
            testCase.verifyEqual(Y2(:, 1:3), Y3(:, 1:3), 'AbsTol', 1e-10);   % real columns unaffected
            testCase.verifyEqual(mha.forward(X), mha.forward(X, true(1, 3)), 'AbsTol', 1e-14);
            testCase.verifyError(@() mha.forward(X, [true true]), ...
                'OceanNLP:Embedding:Modules:MultiHeadAttention:MaskLengthMismatch');
        end

        % ---------------- backbone ----------------
        function testBackbonePairShapesAndSegments(testCase)
            bb = OceanNLP.Embedding.BertBackbone('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            [X, ids, seg] = bb.encodePair("cat", "dog , a");
            testCase.verifyEqual(size(X), [4 numel(ids)]);
            testCase.verifyEqual(ids, [2 7 3 8 9 11 3]);
            testCase.verifyEqual(seg, [0 0 0 1 1 1 1]);
        end

        function testEncoderDelegatesAndMeanPoolsBackbone(testCase)
            enc = OceanNLP.Embedding.BertEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            enc.fit("x");
            E = enc.forward("cat , dog");
            X = OceanNLP.Embedding.BertBackbone('Weights', testCase.W, 'VocabFile', testCase.VocabFile).encodeText("cat , dog");
            testCase.verifyEqual(E, mean(X, 2)', 'AbsTol', 1e-12);
        end

        function testNormalizeOutputGivesUnitRows(testCase)
            enc = OceanNLP.Embedding.BertEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile, 'NormalizeOutput', true);
            enc.fit("x");
            E = enc.forward(["cat dog", "a cat"]);
            testCase.verifyEqual(sqrt(sum(E.^2, 2)), [1; 1], 'AbsTol', 1e-12);
            raw = OceanNLP.Embedding.BertEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            raw.fit("x");
            R = raw.forward(["cat dog", "a cat"]);
            testCase.verifyEqual(E, R ./ sqrt(sum(R.^2, 2)), 'AbsTol', 1e-12);
        end

        % ---------------- cross-encoder ----------------
        function testCrossEncoderMatchesManualHeadComputation(testCase)
            ce = OceanNLP.Embedding.CrossEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            bb = OceanNLP.Embedding.BertBackbone('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            X = bb.encodePair("cat", "dog a");
            p = tanh(testCase.W.Pooler.W * X(:, 1) + testCase.W.Pooler.b);
            expected = testCase.W.Classifier.W * p + testCase.W.Classifier.b;
            testCase.verifyEqual(ce.score("cat", "dog a"), expected, 'AbsTol', 1e-12);
        end

        function testCrossEncoderSigmoidAndScoreDocs(testCase)
            raw = OceanNLP.Embedding.CrossEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            sig = OceanNLP.Embedding.CrossEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile, 'Activation', "sigmoid");
            docs = ["dog a", "cat cat", "a"];
            r = raw.scoreDocs("cat", docs);
            testCase.verifyEqual(size(r), [1 3]);
            testCase.verifyEqual(sig.scoreDocs("cat", docs), 1 ./ (1 + exp(-r)), 'AbsTol', 1e-12);
        end

        function testCrossEncoderIsOrderSensitiveJointModel(testCase)
            ce = OceanNLP.Embedding.CrossEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            testCase.verifyNotEqual(ce.score("cat", "dog"), ce.score("dog", "cat"));
        end

        function testCrossEncoderMissingHeadsErrors(testCase)
            W2 = rmfield(testCase.W, 'Classifier');
            testCase.verifyError(@() OceanNLP.Embedding.CrossEncoder('Weights', W2, 'VocabFile', testCase.VocabFile), ...
                'OceanNLP:Embedding:CrossEncoder:MissingHead');
            W3 = testCase.W; W3.Classifier.W = ones(2, 4);
            testCase.verifyError(@() OceanNLP.Embedding.CrossEncoder('Weights', W3, 'VocabFile', testCase.VocabFile), ...
                'OceanNLP:Embedding:CrossEncoder:NotSingleLabel');
        end

        function testCrossEncoderMetadataPointersResolve(testCase)
            ce = OceanNLP.Embedding.CrossEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            m = ce.getMetadata();
            testCase.verifyFalse(m.IsSymbolic);
            testCase.verifyTrue(contains(m.ComposedPartReference.PoolerSymbolicLatex, "\tanh"));
            testCase.verifyEqual(ce.getPipelineSpec().Kind, "VIRTUAL");
        end

        % ---------------- ColBERT ----------------
        function testColbertQueryPaddedToFixedLengthWithUnitRows(testCase)
            enc = OceanNLP.Embedding.ColBERTEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile, 'QueryMaxLength', 8);
            Q = enc.encodeQuery("cat dog");                    % [CLS][Q] cat dog [SEP] + 3 [MASK]
            testCase.verifyEqual(size(Q), [8 3]);
            testCase.verifyEqual(sqrt(sum(Q.^2, 2)), ones(8, 1), 'AbsTol', 1e-12);
        end

        function testColbertQueryEqualsIndependentRecompute(testCase)
            % ids follow the documented layout [CLS][Q] cat dog [SEP] [MASK]x3, with the
            % [MASK] keys masked (attend_to_mask_tokens=false). Expected values are built
            % here from the verified backbone + a hand projection/normalization, so a
            % wrong marker, marker position or mask would fail this test.
            enc = OceanNLP.Embedding.ColBERTEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile, 'QueryMaxLength', 8);
            bb = OceanNLP.Embedding.BertBackbone('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            ids = [2 5 7 8 3 4 4 4];                                  % 5 = [unused0] = [Q]
            X = bb.hiddenStates(ids, zeros(1, 8), logical([1 1 1 1 1 0 0 0]));
            Y = (testCase.W.Projection.W * X)';
            expected = Y ./ sqrt(sum(Y.^2, 2));
            testCase.verifyEqual(enc.encodeQuery("cat dog"), expected, 'AbsTol', 1e-12);
            % the [D] marker (6) must give a DIFFERENT result: markers are not interchangeable
            Xd = bb.hiddenStates([2 6 7 8 3 4 4 4], zeros(1, 8), logical([1 1 1 1 1 0 0 0]));
            Yd = (testCase.W.Projection.W * Xd)';
            testCase.verifyGreaterThan(max(abs(enc.encodeQuery("cat dog") - Yd ./ sqrt(sum(Yd.^2, 2))), [], 'all'), 1e-6);
        end

        function testColbertDocumentEqualsIndependentRecompute(testCase)
            enc = OceanNLP.Embedding.ColBERTEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            bb = OceanNLP.Embedding.BertBackbone('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            ids = [2 6 7 9 8 10 3];                                   % [CLS][D] cat , dog . [SEP]; 9=",", 10="."
            X = bb.hiddenStates(ids, zeros(1, 7));
            Y = (testCase.W.Projection.W * X)';
            Y = Y ./ sqrt(sum(Y.^2, 2));
            testCase.verifyEqual(enc.encodeDocument("cat , dog ."), Y([1 2 3 5 7], :), 'AbsTol', 1e-12);
        end

        function testColbertQueryWithoutExpansionIsUnpadded(testCase)
            enc = OceanNLP.Embedding.ColBERTEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile, ...
                'QueryMaxLength', 8, 'QueryExpansion', false);
            testCase.verifyEqual(size(enc.encodeQuery("cat dog")), [5 3]);
        end

        function testColbertQueryTruncatesToMaxLength(testCase)
            enc = OceanNLP.Embedding.ColBERTEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile, 'QueryMaxLength', 5);
            testCase.verifyEqual(size(enc.encodeQuery("cat cat cat cat cat cat")), [5 3]);
        end

        function testColbertDocumentDropsPunctuationRows(testCase)
            enc = OceanNLP.Embedding.ColBERTEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            % ids: [CLS][D] cat , dog . [SEP] = 7 rows, minus "," and "." = 5
            testCase.verifyEqual(size(enc.encodeDocument("cat , dog .")), [5 3]);
            keep = OceanNLP.Embedding.ColBERTEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile, 'SkipPunctuation', false);
            testCase.verifyEqual(size(keep.encodeDocument("cat , dog .")), [7 3]);
        end

        function testColbertDocumentTruncation(testCase)
            enc = OceanNLP.Embedding.ColBERTEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile, 'DocMaxLength', 6);
            testCase.verifyEqual(size(enc.encodeDocument("cat cat cat cat cat")), [6 3]);
        end

        function testMaskPaddingDoesNotAffectRealTokensWhenNotAttended(testCase)
            a = OceanNLP.Embedding.ColBERTEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile, 'QueryMaxLength', 6);
            b = OceanNLP.Embedding.ColBERTEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile, 'QueryMaxLength', 9);
            Qa = a.encodeQuery("cat dog"); Qb = b.encodeQuery("cat dog");
            testCase.verifyEqual(Qa(1:5, :), Qb(1:5, :), 'AbsTol', 1e-10);   % [CLS][Q] cat dog [SEP] identical
            c = OceanNLP.Embedding.ColBERTEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile, ...
                'QueryMaxLength', 6, 'AttendToMaskTokens', true);
            d = OceanNLP.Embedding.ColBERTEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile, ...
                'QueryMaxLength', 9, 'AttendToMaskTokens', true);
            Qc = c.encodeQuery("cat dog"); Qd = d.encodeQuery("cat dog");
            testCase.verifyGreaterThan(max(abs(Qc(1:5, :) - Qd(1:5, :)), [], 'all'), 1e-6);   % attending to pads changes them
        end

        function testEmptyTextEdgeCases(testCase)
            ce = OceanNLP.Embedding.CrossEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            testCase.verifyTrue(isfinite(ce.score("", "cat")));
            testCase.verifyTrue(isfinite(ce.score("cat", "")));
            testCase.verifyTrue(isfinite(ce.score("", "")));
            cb = OceanNLP.Embedding.ColBERTEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile, 'QueryMaxLength', 8);
            testCase.verifyEqual(size(cb.encodeQuery("")), [8 3]);            % [CLS][Q][SEP] + 5 [MASK]
            testCase.verifyEqual(size(cb.encodeDocument("")), [3 3]);         % [CLS][D][SEP]
            enc = OceanNLP.Embedding.BertEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            enc.fit("x");
            E = enc.forward("");
            testCase.verifyEqual(size(E), [1 4]);
            testCase.verifyTrue(all(isfinite(E)));
        end

        function testAllPunctuationDocumentKeepsOnlySpecialRows(testCase)
            enc = OceanNLP.Embedding.ColBERTEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            testCase.verifyEqual(size(enc.encodeDocument(", . ,")), [3 3]);    % only [CLS][D][SEP] survive
        end

        function testColbertMissingProjectionAndMetadata(testCase)
            W2 = rmfield(testCase.W, 'Projection');
            testCase.verifyError(@() OceanNLP.Embedding.ColBERTEncoder('Weights', W2, 'VocabFile', testCase.VocabFile), ...
                'OceanNLP:Embedding:ColBERTEncoder:MissingProjection');
            enc = OceanNLP.Embedding.ColBERTEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            m = enc.getMetadata();
            testCase.verifyFalse(m.IsSymbolic);
            testCase.verifyGreaterThan(strlength(m.ComposedPartReference.ProjectionSymbolicLatex), 0);
            testCase.verifyEqual(enc.EmbeddingDim, 3);
        end
    end
end
