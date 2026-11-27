classdef CosineSimilarityTest < OceanTest.BaseTestCase
    % COSINESIMILARITYTEST  Unit tests for OceanStats.Similarity.CosineSimilarity
    % and the OceanStats.Similarity.ISimilarity abstract contract.

    methods (Test)
        function test45DegreeHandComputed(testCase)
            % A = [1 0], B = [1 1]/sqrt(2) is the textbook 45-degree example:
            % cos(45deg) = sqrt(2)/2 = 0.70710678118654752...
            import OceanStats.Similarity.CosineSimilarity
            sim = CosineSimilarity();

            A = [1 0];
            B = [1 1] / sqrt(2);

            s = sim.forward(A, B);

            testCase.verifyEqual(s, cosd(45), 'AbsTol', 1e-10);
        end

        function testFunctionalComputeMatchesInstanceForward(testCase)
            % The static compute() core must agree with the instance wrapper.
            import OceanStats.Similarity.CosineSimilarity
            sim = CosineSimilarity();

            A = [3 4; 1 0];
            B = [4 3; 0 1];

            sInstance = sim.forward(A, B);
            sFunctional = CosineSimilarity.compute(A, B, sim.Eps);

            testCase.verifyEqual(sInstance, sFunctional, 'AbsTol', 1e-12);
        end

        function testShapeMismatchErrors(testCase)
            import OceanStats.Similarity.CosineSimilarity
            sim = CosineSimilarity();

            testCase.verifyError(@() sim.forward([1 2], [1 2 3]), ...
                'OceanStats:Similarity:CosineSimilarity:ShapeMismatch');
        end

        function testZeroVectorGuardIsFiniteNotNaNOrInf(testCase)
            % An all-zero row must not produce NaN/Inf from a 0/0 division:
            % the Eps guard on the denominator must catch it.
            import OceanStats.Similarity.CosineSimilarity
            sim = CosineSimilarity();

            A = [0 0];
            B = [1 1];

            s = sim.forward(A, B);

            testCase.verifyFalse(isnan(s));
            testCase.verifyFalse(isinf(s));
            testCase.verifyEqual(s, 0, 'AbsTol', 1e-6);
        end

        function testPairwiseSparseInputMatchesDenseInput(testCase)
            % pairwise() on sparse A/B must produce numerically identical
            % results to the same computation on the dense equivalents —
            % this is the primary real-world call site (cosine on sparse
            % TF-IDF rows), so correctness under sparsity must be verified
            % empirically, not assumed.
            import OceanStats.Similarity.CosineSimilarity
            sim = CosineSimilarity();

            A = sparse([1 0 2; 0 3 0]);
            B = sparse([2 0 0; 1 1 1; 0 0 4]);

            Psparse = sim.pairwise(A, B);
            Pdense  = sim.pairwise(full(A), full(B));

            testCase.verifyEqual(full(Psparse), Pdense, 'AbsTol', 1e-12);
            testCase.verifySize(Psparse, [size(A, 1), size(B, 1)]);
        end

        function testPairwiseDifferentRowCountsSameColumns(testCase)
            import OceanStats.Similarity.CosineSimilarity
            sim = CosineSimilarity();

            Query  = [1 0 0];
            Corpus = [1 0 0; 0 1 0; 1 1 0];

            P = sim.pairwise(Query, Corpus);

            testCase.verifySize(P, [1 3]);
            testCase.verifyEqual(P(1), 1, 'AbsTol', 1e-12);
            testCase.verifyEqual(P(2), 0, 'AbsTol', 1e-12);
        end

        function testAbstractInterfaceCannotBeInstantiated(testCase)
            testCase.verifyError(@() OceanStats.Similarity.ISimilarity(), ...
                'MATLAB:class:abstract');
        end
    end
end
