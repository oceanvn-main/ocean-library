classdef LateInteractionTest < OceanTest.BaseTestCase
    % LATEINTERACTIONTEST  MaxSim s(q,d) = sum_i max_j q_i . d_j, hand-worked.

    methods (Test)
        function testHandWorkedMaxSim(testCase)
            Q = [1 0; 0 1];                 % two query tokens
            D = [0.6 0.8; 1 0; 0 0.5];      % three doc tokens
            % q1=[1 0]: dots = [0.6 1 0]   -> max 1
            % q2=[0 1]: dots = [0.8 0 0.5] -> max 0.8
            testCase.verifyEqual(OceanStats.Similarity.LateInteraction.maxSim(Q, D), 1.8, 'AbsTol', 1e-12);
        end

        function testNotSymmetric(testCase)
            Q = [1 0];
            D = [1 0; 0 1];
            f = @OceanStats.Similarity.LateInteraction.maxSim;
            testCase.verifyEqual(f(Q, D), 1, 'AbsTol', 1e-12);   % 1 query token
            testCase.verifyEqual(f(D, Q), 1, 'AbsTol', 1e-12);   % q1 -> 1, q2 -> 0, sum 1 (equal here)
            Q2 = [1 0; 1 0];
            testCase.verifyEqual(f(Q2, D), 2, 'AbsTol', 1e-12);  % both query tokens hit
            testCase.verifyEqual(f(D, Q2), 1, 'AbsTol', 1e-12);  % asymmetric
        end

        function testNegativeSimilarityIsKept(testCase)
            % max over j of all-negative dots stays negative (no clamp at 0).
            testCase.verifyEqual(OceanStats.Similarity.LateInteraction.maxSim([1 0], [-1 0; -0.5 0]), -0.5, 'AbsTol', 1e-12);
        end

        function testSinglePrecisionDocsAreHandled(testCase)
            Q = [1 0; 0 1];
            D = single([0.6 0.8; 1 0; 0 0.5]);
            s = OceanStats.Similarity.LateInteraction.maxSim(Q, D);
            testCase.verifyClass(s, 'double');
            testCase.verifyEqual(s, 1.8, 'AbsTol', 1e-6);
        end

        function testEmptyScoresZeroAndDimMismatchErrors(testCase)
            f = @OceanStats.Similarity.LateInteraction.maxSim;
            testCase.verifyEqual(f(zeros(0,2), [1 0]), 0);
            testCase.verifyError(@() f([1 0 0], [1 0]), 'OceanStats:Similarity:LateInteraction:DimMismatch');
        end

        function testWrapperScoresCellOfDocs(testCase)
            li = OceanStats.Similarity.LateInteraction();
            Q = [1 0; 0 1];
            docs = {[0.6 0.8; 1 0; 0 0.5], [1 0], [0 0]};
            % doc2: q1->1, q2->0 = 1 ; doc3: 0
            testCase.verifyEqual(li.score(Q, docs), [1.8 1 0], 'AbsTol', 1e-12);
        end

        function testThreePillars(testCase)
            li = OceanStats.Similarity.LateInteraction();
            meta = li.getMetadata();
            testCase.verifyTrue(meta.IsSymbolic);
            testCase.verifyTrue(contains(meta.SymbolicLatex, "\max"));
            testCase.verifyEqual(li.getPipelineSpec().Kind, "NONVIRTUAL");
        end
    end
end
