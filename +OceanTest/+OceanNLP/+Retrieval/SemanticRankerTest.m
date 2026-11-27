classdef SemanticRankerTest < OceanTest.BaseTestCase
    % SEMANTICRANKERTEST  Unit tests for OceanNLP.Retrieval.SemanticRanker.

    properties
        Corpus   % topic-clustered: docs 1-2 about cats, docs 3-5 about finance
    end

    methods (TestClassSetup)
        function buildCorpus(testCase)
            testCase.Corpus = [ ...
                "cats are small furry feline pets that purr and meow", ...
                "kittens are baby cats that like to play with yarn", ...
                "the stock market rallied today on strong earnings reports", ...
                "investors watched bond yields rise amid inflation concerns", ...
                "quarterly earnings for tech companies beat analyst expectations"];
        end
    end

    methods (Test)
        function testPlantedRelevanceRetrievesObviousDocumentFirst(testCase)
            import OceanNLP.Retrieval.SemanticRanker
            sem = SemanticRanker();
            sem.fit(testCase.Corpus);

            [rankedIdx, ~] = sem.rank("cats and kittens are cute feline pets");

            testCase.verifyEqual(rankedIdx(1), 1);
        end

        function testReturnShapeIsRowVectorsMatchingCorpusSize(testCase)
            import OceanNLP.Retrieval.SemanticRanker
            sem = SemanticRanker();
            sem.fit(testCase.Corpus);

            [rankedIdx, scores] = sem.rank("earnings report");

            N = numel(testCase.Corpus);
            testCase.verifySize(rankedIdx, [1 N]);
            testCase.verifySize(scores, [1 N]);
            testCase.verifyEqual(sort(rankedIdx), 1:N);
            testCase.verifyTrue(issorted(scores, 'descend'));
        end

        function testOOVQueryTermsDoNotCrash(testCase)
            import OceanNLP.Retrieval.SemanticRanker
            sem = SemanticRanker();
            sem.fit(testCase.Corpus);

            [rankedIdx, scores] = sem.rank("zzzznonsenseword qqqqfoo");

            N = numel(testCase.Corpus);
            testCase.verifySize(rankedIdx, [1 N]);
            testCase.verifyEqual(scores, zeros(1, N), 'AbsTol', 1e-9);
        end

        function testRankBeforeFitErrors(testCase)
            import OceanNLP.Retrieval.SemanticRanker
            sem = SemanticRanker();
            testCase.verifyError(@() sem.rank("hi"), ...
                'OceanNLP:Retrieval:SemanticRanker:NotFitted');
        end

        function testIdenticalReturnShapeToBM25Ranker(testCase)
            % [rankedIdx, scores] must be the exact same shape/contract as
            % BM25Ranker so either may be substituted behind IRanker.
            import OceanNLP.Retrieval.SemanticRanker
            import OceanNLP.Retrieval.BM25Ranker

            sem = SemanticRanker();
            sem.fit(testCase.Corpus);
            bm25 = BM25Ranker();
            bm25.fit(testCase.Corpus);

            [idxSem, scoresSem] = sem.rank("cats and kittens");
            [idxBm25, scoresBm25] = bm25.rank("cats and kittens");

            testCase.verifyEqual(size(idxSem), size(idxBm25));
            testCase.verifyEqual(size(scoresSem), size(scoresBm25));
            testCase.verifyClass(idxSem, class(idxBm25));
            testCase.verifyClass(scoresSem, class(scoresBm25));
        end
    end
end
