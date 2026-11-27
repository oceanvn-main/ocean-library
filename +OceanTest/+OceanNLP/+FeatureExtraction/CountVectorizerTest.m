classdef CountVectorizerTest < OceanTest.BaseTestCase
    % COUNTVECTORIZERTEST  Unit tests for OceanNLP.FeatureExtraction.CountVectorizer
    % and the OceanNLP.FeatureExtraction.IVectorizer abstract contract.

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
        function testSparseOutputOnRealisticCorpus(testCase)
            import OceanNLP.FeatureExtraction.CountVectorizer
            cv = CountVectorizer();
            cv.fit(testCase.Corpus);
            X = cv.forward(testCase.Corpus);

            testCase.verifyTrue(issparse(X));
            testCase.verifyEqual(size(X, 1), numel(testCase.Corpus));
            testCase.verifyEqual(size(X, 2), cv.Vocabulary.VocabSize);
            testCase.verifyGreaterThan(nnz(X), 0);
        end

        function testOOVHandlingAtForwardTime(testCase)
            % A brand-new document at forward() time containing tokens
            % never seen during fit() must collapse those tokens into the
            % reserved OOV column, never error and never drop them.
            import OceanNLP.FeatureExtraction.CountVectorizer
            cv = CountVectorizer();
            cv.fit(testCase.Corpus);

            Xnew = cv.forward("a brand new zzzunseen word");
            oovCol = cv.Vocabulary.OOVId;

            testCase.verifyEqual(full(Xnew(1, oovCol)), 5);   % 5 unseen tokens
            testCase.verifyEqual(nnz(Xnew), 1);                % only the OOV column is nonzero
        end

        function testNotFittedErrors(testCase)
            import OceanNLP.FeatureExtraction.CountVectorizer
            cv = CountVectorizer();
            testCase.verifyError(@() cv.forward("hello"), ...
                'OceanNLP:FeatureExtraction:CountVectorizer:NotFitted');
        end

        function testAbstractInterfaceCannotBeInstantiated(testCase)
            testCase.verifyError(@() OceanNLP.FeatureExtraction.IVectorizer(), ...
                'MATLAB:class:abstract');
        end
    end
end
