classdef RetrievalEvaluatorTest < OceanTest.BaseTestCase
    % RETRIEVALEVALUATORTEST  Unit tests for OceanNLP.Retrieval.RetrievalEvaluator.
    %
    % Corpus mirrors BM25RankerTest's fixture (topic-clustered: docs 1-2
    % cats, docs 3-5 finance).

    properties
        Corpus
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
        function testWiringMatchesHandComposedRankerPlusMetricForOneQuery(testCase)
            % "Hand-computed expectation" here means: call ranker.rank()
            % and metric.forward() DIRECTLY, outside the evaluator (both
            % already independently covered by their own test files), and
            % confirm the evaluator's report reproduces EXACTLY that value
            % -- this isolates and verifies the evaluator's own orchestration
            % wiring, not the ranking or metric math themselves.
            import OceanNLP.Retrieval.BM25Ranker
            import OceanNLP.Retrieval.RetrievalEvaluator
            import OceanStats.Metrics.Retrieval.PrecisionAtK

            bm25 = BM25Ranker();
            bm25.fit(testCase.Corpus);

            query = "cats and kittens are cute feline pets";
            relevance = [1 1 0 0 0];   % docs 1,2 (the cat cluster) are relevant

            [rankedIdxDirect, ~] = bm25.rank(query);
            metric = PrecisionAtK('K', 2);
            expectedScore = metric.forward(rankedIdxDirect, relevance);

            evaluator = RetrievalEvaluator(bm25, {metric});
            report = evaluator.evaluate(query, {relevance});

            testCase.verifyEqual(report{"Query1", "precision@k_K2"}, expectedScore, 'AbsTol', 1e-12);
            testCase.verifyEqual(report{"Mean", "precision@k_K2"}, expectedScore, 'AbsTol', 1e-12);
        end

        function testNaNExclusionWithDisclosure(testCase)
            % Query 2 has ZERO relevant docs defined at all -> AveragePrecision
            % is undefined -> NaN for that query. The Mean row must EXCLUDE
            % it (mean over the one defined query only), the exclusion must
            % be disclosed via the `excluded` output, AND via a warning.
            import OceanNLP.Retrieval.BM25Ranker
            import OceanNLP.Retrieval.RetrievalEvaluator
            import OceanStats.Metrics.Retrieval.AveragePrecision

            bm25 = BM25Ranker();
            bm25.fit(testCase.Corpus);

            queries = ["cats and kittens are cute feline pets", "stock market earnings"];
            relevance1 = [1 1 0 0 0];   % well-defined
            relevance2 = [0 0 0 0 0];   % zero relevant docs at all -> undefined -> NaN

            metric = AveragePrecision();
            [rankedIdx1, ~] = bm25.rank(queries(1));
            expectedScore1 = metric.forward(rankedIdx1, relevance1);

            evaluator = RetrievalEvaluator(bm25, {metric});

            testCase.verifyWarning(@() evaluator.evaluate(queries, {relevance1, relevance2}), ...
                'OceanNLP:Retrieval:RetrievalEvaluator:ExcludedQueries');

            [report, excluded] = evaluator.evaluate(queries, {relevance1, relevance2});

            testCase.verifyEqual(report{"Query1", "ap"}, expectedScore1, 'AbsTol', 1e-12);
            testCase.verifyTrue(isnan(report{"Query2", "ap"}));
            % Mean excludes the NaN query -> equals query 1's own score exactly.
            testCase.verifyEqual(report{"Mean", "ap"}, expectedScore1, 'AbsTol', 1e-12);
            testCase.verifyEqual(excluded{"ExcludedQueries", "ap"}, 1);
        end

        function testWorksUniformlyAcrossBM25StrategiesNoSpecialCasing(testCase)
            import OceanNLP.Retrieval.BM25Ranker
            import OceanNLP.Retrieval.RetrievalEvaluator
            import OceanStats.Metrics.Retrieval.ReciprocalRank

            query = "earnings report";
            relevance = [0 0 1 0 1];   % docs 3,5 (finance cluster) relevant

            bm25 = BM25Ranker();
            bm25.fit(testCase.Corpus);
            iir = BM25Ranker('Strategy', "matrix");   % InvertedIndexRanker was folded into BM25Ranker('Strategy',...)
            iir.fit(testCase.Corpus);

            metric = ReciprocalRank();
            evalBM25 = RetrievalEvaluator(bm25, {metric});
            evalIIR  = RetrievalEvaluator(iir, {metric});

            reportBM25 = evalBM25.evaluate(query, {relevance});
            reportIIR  = evalIIR.evaluate(query, {relevance});

            % Per §3.1's equivalence requirement, BM25Ranker's two
            % strategies (inverted_index, matrix) must agree here too -- a second-angle
            % re-confirmation of the same equivalence, now through the
            % evaluator.
            testCase.verifyEqual(reportBM25{"Query1", "rr"}, reportIIR{"Query1", "rr"}, 'AbsTol', 1e-10);
        end

        function testWorksWithSemanticRankerConstructionAndEvaluationSucceed(testCase)
            % Verifies (does not assume) that SemanticRanker construction
            % and evaluation both work through RetrievalEvaluator -- this
            % was the site of a real, previously-fixed blocker (an
            % out-of-band NormScheme refactor broke TfidfVectorizer/
            % SemanticRanker construction; see master plan §7 Execution
            % Log, Phase 3). Re-confirmed here rather than assumed fixed.
            import OceanNLP.Retrieval.SemanticRanker
            import OceanNLP.Retrieval.RetrievalEvaluator
            import OceanStats.Metrics.Retrieval.PrecisionAtK

            sem = SemanticRanker();
            testCase.verifyWarningFree(@() sem.fit(testCase.Corpus));

            metric = PrecisionAtK('K', 2);
            evaluator = RetrievalEvaluator(sem, {metric});
            relevance = [1 1 0 0 0];
            report = evaluator.evaluate("cats and kittens", {relevance});
            testCase.verifyTrue(istable(report));
            testCase.verifyFalse(isnan(report{"Query1", "precision@k_K2"}));
        end

        function testConstructorRequiresAlreadyFittedRanker(testCase)
            import OceanNLP.Retrieval.BM25Ranker
            import OceanNLP.Retrieval.RetrievalEvaluator
            import OceanStats.Metrics.Retrieval.ReciprocalRank

            unfittedBm25 = BM25Ranker();
            testCase.verifyError(@() RetrievalEvaluator(unfittedBm25, {ReciprocalRank()}), ...
                'OceanNLP:Retrieval:RetrievalEvaluator:NotFitted');
        end

        function testColumnNamesDisambiguateSameMetricDifferentK(testCase)
            import OceanNLP.Retrieval.BM25Ranker
            import OceanNLP.Retrieval.RetrievalEvaluator
            import OceanStats.Metrics.Retrieval.PrecisionAtK

            bm25 = BM25Ranker();
            bm25.fit(testCase.Corpus);

            metrics = {PrecisionAtK('K', 1), PrecisionAtK('K', 3)};
            evaluator = RetrievalEvaluator(bm25, metrics);
            relevance = [1 1 0 0 0];
            report = evaluator.evaluate("cats and kittens", {relevance});

            testCase.verifyTrue(ismember("precision@k_K1", string(report.Properties.VariableNames)));
            testCase.verifyTrue(ismember("precision@k_K3", string(report.Properties.VariableNames)));
        end
    end
end
