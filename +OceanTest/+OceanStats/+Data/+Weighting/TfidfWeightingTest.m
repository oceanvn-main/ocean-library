classdef TfidfWeightingTest < OceanTest.BaseTestCase
    % TFIDFWEIGHTINGTEST  Unit tests for OceanStats.Data.Weighting.TfidfWeighting
    % and the OceanStats.Data.Weighting.IWeighting abstract contract.
    %
    % The hand-worked 3-document example below was independently computed
    % from the documented smooth-idf formula (idf_t = log((1+N)/(1+df_t))+1,
    % tf .* idf, row-L2-normalize) directly in MATLAB using raw arithmetic —
    % not by calling TfidfWeighting itself — to provide a genuine external
    % ground truth for the regression check.

    properties
        Counts       % (3 x 4) hand-picked count matrix
        ExpectedW    % pre-normalization tf .* idf, hand-derived
        ExpectedWn   % row-L2-normalized, hand-derived (default Norm="l2")
    end

    methods (TestClassSetup)
        function buildHandWorkedExample(testCase)
            % Vocabulary of 4 terms across 3 documents:
            %   term1: df=2 (doc1,doc3)   term2: df=2 (doc2,doc3)
            %   term3: df=1 (doc1)        term4: df=2 (doc2,doc3)
            testCase.Counts = [3 0 1 0; ...
                                0 2 0 1; ...
                                1 1 0 2];

            testCase.ExpectedW = [3.86304621735534, 0, 1.69314718055995, 0; ...
                                   0, 2.57536414490356, 0, 1.28768207245178; ...
                                   1.28768207245178, 1.28768207245178, 0, 2.57536414490356];

            testCase.ExpectedWn = [0.915890331969465, 0, 0.40142857372746, 0; ...
                                    0, 0.894427190999916, 0, 0.447213595499958; ...
                                    0.408248290463863, 0.408248290463863, 0, 0.816496580927726];
        end
    end

    methods (Test)
        function testHandWorkedExampleNoNorm(testCase)
            import OceanStats.Data.Weighting.TfidfWeighting
            w = TfidfWeighting.create('NormScheme', "none");
            w.fit(testCase.Counts);
            W = w.forward(testCase.Counts);

            testCase.verifyEqual(full(W), testCase.ExpectedW, 'AbsTol', 1e-9);
        end

        function testHandWorkedExampleDefaultL2Norm(testCase)
            import OceanStats.Data.Weighting.TfidfWeighting
            w = TfidfWeighting();  % Norm="l2" by default
            w.fit(testCase.Counts);
            W = w.forward(testCase.Counts);

            testCase.verifyEqual(full(W), testCase.ExpectedWn, 'AbsTol', 1e-9);
            % Rows of an L2-normalized non-empty matrix have unit norm.
            rowNorms = sqrt(sum(full(W).^2, 2));
            testCase.verifyEqual(rowNorms, ones(3, 1), 'AbsTol', 1e-9);
        end

        function testNotFittedErrors(testCase)
            import OceanStats.Data.Weighting.TfidfWeighting
            w = TfidfWeighting();
            testCase.verifyError(@() w.forward(testCase.Counts), ...
                'OceanStats:Data:Weighting:TfidfWeighting:NotFitted');
        end

        function testVocabMismatchErrors(testCase)
            import OceanStats.Data.Weighting.TfidfWeighting
            w = TfidfWeighting();
            w.fit(testCase.Counts);
            testCase.verifyError(@() w.forward([1 2 3]), ...
                'OceanStats:Data:Weighting:TfidfWeighting:VocabMismatch');
        end

        function testStateDictLoadStateDictRoundTrip(testCase)
            import OceanStats.Data.Weighting.TfidfWeighting
            w1 = TfidfWeighting.create('SublinearTf', true, 'NormScheme', "l1");
            w1.fit(testCase.Counts);

            s = w1.stateDict();
            w2 = TfidfWeighting.create();
            w2.loadStateDict(s);

            testCase.verifyEqual(full(w2.forward(testCase.Counts)), ...
                full(w1.forward(testCase.Counts)), 'AbsTol', 1e-12);
            testCase.verifyEqual(w2.SublinearTf, true);
            testCase.verifyEqual(w2.NormScheme.name(), "l1");
        end

        function testSparsityPreservedThroughForward(testCase)
            % Sparsity is a correctness requirement (Master Plan §3.2):
            % a >90%-sparse input must not silently densify anywhere in
            % the forward() pipeline (tf-scaling, sublinear spfun step,
            % and row-L2-normalization division).
            import OceanStats.Data.Weighting.TfidfWeighting
            rng(7);
            sparseCounts = round(sprand(50, 200, 0.03) * 5);

            for sublinear = [false true]
                w = TfidfWeighting('SublinearTf', sublinear);
                w.fit(sparseCounts);
                W = w.forward(sparseCounts);
                testCase.verifyTrue(issparse(W));
            end
        end

        function testAbstractInterfaceCannotBeInstantiated(testCase)
            testCase.verifyError(@() OceanStats.Data.Weighting.IWeighting(), ...
                'MATLAB:class:abstract');
        end
    end
end
