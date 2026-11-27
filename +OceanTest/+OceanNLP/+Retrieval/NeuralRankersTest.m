classdef NeuralRankersTest < OceanTest.BaseTestCase
    % NEURALRANKERSTEST  BiEncoderRanker, CrossEncoderReranker, ColBERTRanker
    % contracts (full permutation, sorted scores, -Inf tail, composition).

    properties
        VocabFile
        W
        Corpus = ["cat dog", "a cat cat", "dog , dog .", "a a a", "cat , dog a"]
    end

    methods (TestMethodSetup)
        function makeFixture(testCase)
            F = OceanTest.OceanNLP.Embedding.TinyBertFixture;
            testCase.VocabFile = F.writeVocab();
            testCase.W = F.weights(3, 1);
            testCase.addTeardown(@() delete(testCase.VocabFile));
        end
    end

    methods (Test)
        % ---------------- bi-encoder ----------------
        function testBiEncoderScoresMatchManualDotProducts(testCase)
            enc = OceanNLP.Embedding.BertEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile, 'NormalizeOutput', true);
            r = OceanNLP.Retrieval.BiEncoderRanker('DocEncoder', enc);
            r.fit(testCase.Corpus);
            [idx, s] = r.rank("cat");
            D = enc.forward(testCase.Corpus);
            q = enc.forward("cat");
            manual = (D * q')';
            [expS, expIdx] = sort(manual, 'descend');
            testCase.verifyEqual(idx, expIdx);
            testCase.verifyEqual(s, expS, 'AbsTol', 1e-12);
            testCase.verifyEqual(sort(idx), 1:numel(testCase.Corpus));
        end

        function testBiEncoderDotOnNormalizedEqualsCosineDenseRanker(testCase)
            enc = OceanNLP.Embedding.BertEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile, 'NormalizeOutput', true);
            bi = OceanNLP.Retrieval.BiEncoderRanker('DocEncoder', enc);
            bi.fit(testCase.Corpus);
            dense = OceanNLP.Retrieval.DenseRanker('Encoder', enc);
            dense.fit(testCase.Corpus);
            [i1, s1] = bi.rank("dog a");
            [i2, s2] = dense.rank("dog a");
            testCase.verifyEqual(i1, i2);
            testCase.verifyEqual(s1, s2, 'AbsTol', 1e-10);
        end

        function testBiEncoderSeparateQueryEncoderIsUsed(testCase)
            docEnc = OceanNLP.Embedding.BertEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile, 'NormalizeOutput', true);
            W2 = OceanTest.OceanNLP.Embedding.TinyBertFixture.weights(9, 1);
            qEnc = OceanNLP.Embedding.BertEncoder('Weights', W2, 'VocabFile', testCase.VocabFile, 'NormalizeOutput', true);
            r = OceanNLP.Retrieval.BiEncoderRanker('DocEncoder', docEnc, 'QueryEncoder', qEnc);
            r.fit(testCase.Corpus);
            [~, s] = r.rank("cat");
            q = qEnc.forward("cat");
            D = docEnc.forward(testCase.Corpus);
            testCase.verifyEqual(s, sort((D * q')', 'descend'), 'AbsTol', 1e-12);
        end

        function testBiEncoderNotFittedErrorsAndMetadata(testCase)
            enc = OceanNLP.Embedding.BertEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            r = OceanNLP.Retrieval.BiEncoderRanker('DocEncoder', enc);
            testCase.verifyFalse(r.isFitted());
            testCase.verifyError(@() r.rank("cat"), 'OceanNLP:Retrieval:BiEncoderRanker:NotFitted');
            m = r.getMetadata();
            testCase.verifyFalse(m.IsSymbolic);
            testCase.verifyTrue(contains(m.ComposedPartReference.SimilaritySymbolicLatex, "\cdot"));
        end

        % ---------------- cross-encoder reranker ----------------
        function testRerankerFullPermutationWithInfTail(testCase)
            ce = OceanNLP.Embedding.CrossEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            first = OceanNLP.Retrieval.BM25Ranker();
            rr = OceanNLP.Retrieval.CrossEncoderReranker('FirstStage', first, 'CrossEncoder', ce, 'K', 3);
            rr.fit(testCase.Corpus);
            [idx, s] = rr.rank("cat");
            N = numel(testCase.Corpus);
            testCase.verifyEqual(sort(idx), 1:N);
            testCase.verifyEqual(s(4:end), [-Inf -Inf]);
            testCase.verifyTrue(issorted(s, 'descend'));
            % tail keeps first-stage order
            firstIdx = first.rank("cat");
            testCase.verifyEqual(idx(4:end), firstIdx(4:end));
            % head is the first-stage top-3 re-ordered by cross-encoder score
            testCase.verifyEqual(sort(idx(1:3)), sort(firstIdx(1:3)));
            manual = ce.scoreDocs("cat", testCase.Corpus(idx(1:3)));
            testCase.verifyEqual(s(1:3), manual, 'AbsTol', 1e-12);
        end

        function testRerankerKLargerThanCorpusRescoresAll(testCase)
            ce = OceanNLP.Embedding.CrossEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            rr = OceanNLP.Retrieval.CrossEncoderReranker('FirstStage', OceanNLP.Retrieval.BM25Ranker(), 'CrossEncoder', ce, 'K', 100);
            rr.fit(testCase.Corpus);
            [idx, s] = rr.rank("dog");
            testCase.verifyEqual(sort(idx), 1:numel(testCase.Corpus));
            testCase.verifyFalse(any(isinf(s)));
        end

        function testRerankerKIsRequiredAndNotFittedErrors(testCase)
            ce = OceanNLP.Embedding.CrossEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            testCase.verifyError(@() OceanNLP.Retrieval.CrossEncoderReranker( ...
                'FirstStage', OceanNLP.Retrieval.BM25Ranker(), 'CrossEncoder', ce), 'OceanNLP:Retrieval:CrossEncoderReranker:MissingOption');
            rr = OceanNLP.Retrieval.CrossEncoderReranker('FirstStage', OceanNLP.Retrieval.BM25Ranker(), 'CrossEncoder', ce, 'K', 2);
            testCase.verifyError(@() rr.rank("cat"), 'OceanNLP:Retrieval:CrossEncoderReranker:NotFitted');
        end

        function testRerankerWorksInsideRetrievalEvaluator(testCase)
            ce = OceanNLP.Embedding.CrossEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            rr = OceanNLP.Retrieval.CrossEncoderReranker('FirstStage', OceanNLP.Retrieval.BM25Ranker(), 'CrossEncoder', ce, 'K', 3);
            rr.fit(testCase.Corpus);
            ev = OceanNLP.Retrieval.RetrievalEvaluator(rr, {OceanStats.Metrics.Retrieval.AveragePrecision()});
            rep = ev.evaluate(["cat", "dog"], {[1 1 0 0 1], [1 0 1 0 1]});
            testCase.verifyEqual(size(rep, 1), 3);          % 2 queries + Mean row
            ap = rep{:, "ap"};
            testCase.verifyTrue(all(ap >= 0 & ap <= 1));
            testCase.verifyEqual(ap(3), mean(ap(1:2)), 'AbsTol', 1e-12);
            % independent AP for query 1 from the reranker's own ranking
            ranked = rr.rank("cat");
            relQ1 = [1 1 0 0 1];
            hit = relQ1(ranked);
            expectedAP = sum((cumsum(hit) ./ (1:5)) .* hit) / sum(hit);
            testCase.verifyEqual(ap(1), expectedAP, 'AbsTol', 1e-12);
        end

        function testRerankerKOneAndSingleDocAndNoTermMatch(testCase)
            ce = OceanNLP.Embedding.CrossEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            r1 = OceanNLP.Retrieval.CrossEncoderReranker('FirstStage', OceanNLP.Retrieval.BM25Ranker(), 'CrossEncoder', ce, 'K', 1);
            r1.fit(testCase.Corpus);
            [idx, s] = r1.rank("cat");
            testCase.verifyEqual(sort(idx), 1:5);
            testCase.verifyEqual(s(2:end), -Inf(1, 4));
            testCase.verifyTrue(isfinite(s(1)));
            first = OceanNLP.Retrieval.BM25Ranker(); first.fit(testCase.Corpus);
            firstIdx = first.rank("cat");
            testCase.verifyEqual(idx(1), firstIdx(1));                   % K=1 cannot reorder anything
            % no query term in any document: first stage is all-zero, reranker must still return a permutation
            [idx2, ~] = r1.rank("zzzunseen");
            testCase.verifyEqual(sort(idx2), 1:5);
            % single-document corpus
            r2 = OceanNLP.Retrieval.CrossEncoderReranker('FirstStage', OceanNLP.Retrieval.BM25Ranker(), 'CrossEncoder', ce, 'K', 5);
            r2.fit("cat dog");
            [i3, s3] = r2.rank("cat");
            testCase.verifyEqual(i3, 1);
            testCase.verifyTrue(isfinite(s3));
        end

        function testBiEncoderAndColbertSingleDocCorpus(testCase)
            enc = OceanNLP.Embedding.BertEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile, 'NormalizeOutput', true);
            bi = OceanNLP.Retrieval.BiEncoderRanker('DocEncoder', enc);
            bi.fit("cat dog");
            testCase.verifyEqual(bi.rank("cat"), 1);
            cb = OceanNLP.Retrieval.ColBERTRanker('Encoder', OceanNLP.Embedding.ColBERTEncoder( ...
                'Weights', testCase.W, 'VocabFile', testCase.VocabFile, 'QueryMaxLength', 8));
            cb.fit("cat dog");
            testCase.verifyEqual(cb.rank("cat"), 1);
        end

        % ---------------- ColBERT ranker ----------------
        function testColbertRankerMatchesManualMaxSim(testCase)
            enc = OceanNLP.Embedding.ColBERTEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile, 'QueryMaxLength', 8);
            r = OceanNLP.Retrieval.ColBERTRanker('Encoder', enc);
            r.fit(testCase.Corpus);
            [idx, s] = r.rank("cat dog");
            Q = enc.encodeQuery("cat dog");
            manual = zeros(1, numel(testCase.Corpus));
            for j = 1:numel(manual)
                D = single(enc.encodeDocument(testCase.Corpus(j)));
                manual(j) = double(sum(max(single(Q) * D', [], 2)));
            end
            [expS, expIdx] = sort(manual, 'descend');
            testCase.verifyEqual(idx, expIdx);
            testCase.verifyEqual(s, expS, 'AbsTol', 1e-5);
            testCase.verifyEqual(sort(idx), 1:numel(testCase.Corpus));
        end

        function testColbertDocMatricesAreSingle(testCase)
            enc = OceanNLP.Embedding.ColBERTEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            r = OceanNLP.Retrieval.ColBERTRanker('Encoder', enc);
            r.fit(testCase.Corpus);
            testCase.verifyClass(r.DocMatrices{1}, 'single');
            testCase.verifyEqual(numel(r.DocMatrices), numel(testCase.Corpus));
        end

        function testColbertNotFittedAndMetadata(testCase)
            enc = OceanNLP.Embedding.ColBERTEncoder('Weights', testCase.W, 'VocabFile', testCase.VocabFile);
            r = OceanNLP.Retrieval.ColBERTRanker('Encoder', enc);
            testCase.verifyError(@() r.rank("cat"), 'OceanNLP:Retrieval:ColBERTRanker:NotFitted');
            m = r.getMetadata();
            testCase.verifyFalse(m.IsSymbolic);
            testCase.verifyTrue(contains(m.ComposedPartReference.MaxSimSymbolicLatex, "\max"));
        end
    end
end
