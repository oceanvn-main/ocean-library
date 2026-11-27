classdef SimilarityFactoryTest < OceanTest.BaseTestCase
    % SIMILARITYFACTORYTEST  Unit tests for OceanStats.Similarity.SimilarityFactory.

    methods (Test)
        function testCreateCosineReturnsCosineSimilarity(testCase)
            import OceanStats.Similarity.SimilarityFactory
            sim = SimilarityFactory.create('cosine');

            testCase.verifyClass(sim, 'OceanStats.Similarity.CosineSimilarity');
            testCase.verifyEqual(sim.name(), "cosine");
        end

        function testCaseAndWhitespaceInsensitiveMatching(testCase)
            import OceanStats.Similarity.SimilarityFactory
            sim = SimilarityFactory.create('  CoSiNe  ');

            testCase.verifyClass(sim, 'OceanStats.Similarity.CosineSimilarity');
        end

        function testUnknownNameErrors(testCase)
            import OceanStats.Similarity.SimilarityFactory
            testCase.verifyError(@() SimilarityFactory.create('bogus'), ...
                'OceanStats:Similarity:SimilarityFactory:Unknown');
        end

        function testValidNames(testCase)
            import OceanStats.Similarity.SimilarityFactory
            names = SimilarityFactory.validNames();

            testCase.verifyClass(names, 'cell');
            testCase.verifyTrue(any(strcmp(names, 'cosine')));
        end
    end
end
