classdef DotProductSimilarityTest < OceanTest.BaseTestCase
    % DOTPRODUCTSIMILARITYTEST  Hand-worked expectations, written in this file.

    methods (Test)
        function testForwardRowwiseHandComputed(testCase)
            sim = OceanStats.Similarity.DotProductSimilarity();
            A = [1 2 3; 0 1 0];
            B = [4 5 6; 7 8 9];
            % row1: 1*4+2*5+3*6 = 32 ; row2: 0*7+1*8+0*9 = 8
            testCase.verifyEqual(sim.forward(A, B), [32; 8], 'AbsTol', 1e-12);
        end

        function testPairwiseHandComputed(testCase)
            sim = OceanStats.Similarity.DotProductSimilarity();
            A = [1 0; 0 2];
            B = [3 4; 5 6; 7 8];
            % A(1,:).B' = [3 5 7] ; A(2,:).B' = [8 12 16]
            testCase.verifyEqual(sim.pairwise(A, B), [3 5 7; 8 12 16], 'AbsTol', 1e-12);
        end

        function testEqualsCosineOnUnitVectors(testCase)
            A = [3 4; 1 0] ./ [5; 1];
            B = [0 1; 1 1] ./ [1; sqrt(2)];
            dotS = OceanStats.Similarity.DotProductSimilarity().pairwise(A, B);
            cosS = OceanStats.Similarity.CosineSimilarity().pairwise(A, B);
            testCase.verifyEqual(dotS, cosS, 'AbsTol', 1e-12);
        end

        function testNotNormalized(testCase)
            sim = OceanStats.Similarity.DotProductSimilarity();
            testCase.verifyEqual(sim.forward([2 0], [3 0]), 6, 'AbsTol', 1e-12);   % cosine would be 1
        end

        function testShapeAndDimErrors(testCase)
            sim = OceanStats.Similarity.DotProductSimilarity();
            testCase.verifyError(@() sim.forward([1 2], [1 2; 3 4]), ...
                'OceanStats:Similarity:DotProductSimilarity:ShapeMismatch');
            testCase.verifyError(@() sim.pairwise([1 2], [1 2 3]), ...
                'OceanStats:Similarity:DotProductSimilarity:DimMismatch');
        end

        function testThreePillarsAndFactory(testCase)
            sim = OceanStats.Similarity.SimilarityFactory.create('dot');
            testCase.verifyClass(sim, 'OceanStats.Similarity.DotProductSimilarity');
            testCase.verifyEqual(sim.name(), "dot");
            meta = sim.getMetadata();
            testCase.verifyTrue(meta.IsSymbolic);
            testCase.verifyGreaterThan(strlength(meta.SymbolicLatex), 0);
            testCase.verifyEqual(sim.getPipelineSpec().Kind, "NONVIRTUAL");
            testCase.verifyTrue(any(strcmp(OceanStats.Similarity.SimilarityFactory.validNames(), 'dot')));
        end
    end
end
