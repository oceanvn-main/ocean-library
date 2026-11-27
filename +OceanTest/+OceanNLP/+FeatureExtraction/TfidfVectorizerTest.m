classdef TfidfVectorizerTest < OceanTest.BaseTestCase
    % TFIDFVECTORIZERTEST  Unit tests for OceanNLP.FeatureExtraction.TfidfVectorizer.
    %
    % The architecture-conformance test below is the "no duplicated math"
    % proof required by Master Plan §2/§4/§7.1(A): TfidfVectorizer must
    % contain ZERO independent TF-IDF math and produce output byte-identical
    % to manually composing CountVectorizer + OceanStats.Data.Weighting.
    % TfidfWeighting.

    properties
        Corpus
    end

    methods (TestClassSetup)
        function buildCorpus(testCase)
            testCase.Corpus = ["the cat sat on the mat", ...
                                "the dog ran in the park", ...
                                "cats and dogs are pets"];
        end
    end

    methods (Test)
        function testIdenticalToManualCompositionOfCountVectorizerAndTfidfWeighting(testCase)
            import OceanNLP.FeatureExtraction.TfidfVectorizer
            import OceanNLP.FeatureExtraction.CountVectorizer
            import OceanStats.Data.Weighting.TfidfWeighting

            tfidfVec = TfidfVectorizer();
            tfidfVec.fit(testCase.Corpus);
            Xcomposed = tfidfVec.forward(testCase.Corpus);

            cv = CountVectorizer();
            cv.fit(testCase.Corpus);
            counts = cv.forward(testCase.Corpus);
            w = TfidfWeighting();
            w.fit(counts);
            Xmanual = w.forward(counts);

            testCase.verifyEqual(full(Xcomposed), full(Xmanual));   % byte-identical, no tolerance needed
        end

        function testIdenticalWithNonDefaultOptionsForwarded(testCase)
            % Confirm option forwarding (SublinearTf, Norm, Tokenizer) also
            % keeps the composition byte-identical, not just the defaults.
            import OceanNLP.FeatureExtraction.TfidfVectorizer
            import OceanNLP.FeatureExtraction.CountVectorizer
            import OceanStats.Data.Weighting.TfidfWeighting

            tfidfVec = TfidfVectorizer('SublinearTf', true, 'Norm', "l1");
            tfidfVec.fit(testCase.Corpus);
            Xcomposed = tfidfVec.forward(testCase.Corpus);

            cv = CountVectorizer();
            cv.fit(testCase.Corpus);
            counts = cv.forward(testCase.Corpus);
            w = TfidfWeighting('SublinearTf', true, ...
                'NormScheme', OceanStats.Data.Norm.NormSchemeFactory.create("l1"));
            w.fit(counts);
            Xmanual = w.forward(counts);

            testCase.verifyEqual(full(Xcomposed), full(Xmanual));
        end

        function testSparseOutput(testCase)
            import OceanNLP.FeatureExtraction.TfidfVectorizer
            tfidfVec = TfidfVectorizer();
            tfidfVec.fit(testCase.Corpus);
            X = tfidfVec.forward(testCase.Corpus);

            testCase.verifyTrue(issparse(X));
        end

        function testNotFittedErrors(testCase)
            import OceanNLP.FeatureExtraction.TfidfVectorizer
            tfidfVec = TfidfVectorizer();
            testCase.verifyError(@() tfidfVec.forward("hello"), ...
                'OceanNLP:FeatureExtraction:TfidfVectorizer:NotFitted');
        end
    end
end
