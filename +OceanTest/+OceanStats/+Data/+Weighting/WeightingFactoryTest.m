classdef WeightingFactoryTest < OceanTest.BaseTestCase
    % WEIGHTINGFACTORYTEST  Unit tests for OceanStats.Data.Weighting.WeightingFactory.

    methods (Test)
        function testCreateTfidfReturnsTfidfWeighting(testCase)
            import OceanStats.Data.Weighting.WeightingFactory
            w = WeightingFactory.create('tfidf');

            testCase.verifyClass(w, 'OceanStats.Data.Weighting.TfidfWeighting');
        end

        function testCreateBm25ReturnsBm25Weighting(testCase)
            import OceanStats.Data.Weighting.WeightingFactory
            w = WeightingFactory.create('bm25');

            testCase.verifyClass(w, 'OceanStats.Data.Weighting.BM25Weighting');
        end

        function testCaseAndWhitespaceInsensitiveMatching(testCase)
            import OceanStats.Data.Weighting.WeightingFactory
            w = WeightingFactory.create('  Bm25  ');

            testCase.verifyClass(w, 'OceanStats.Data.Weighting.BM25Weighting');
        end

        function testUnknownNameErrors(testCase)
            import OceanStats.Data.Weighting.WeightingFactory
            testCase.verifyError(@() WeightingFactory.create('bogus'), ...
                'OceanStats:Data:Weighting:WeightingFactory:Unknown');
        end

        function testValidNames(testCase)
            import OceanStats.Data.Weighting.WeightingFactory
            names = WeightingFactory.validNames();

            testCase.verifyClass(names, 'cell');
            testCase.verifyTrue(any(strcmp(names, 'tfidf')));
            testCase.verifyTrue(any(strcmp(names, 'bm25')));
        end

        function testCreatedWeightingIsUsable(testCase)
            % Factory-created instances must be immediately fit/forward-able,
            % same as directly-constructed ones.
            import OceanStats.Data.Weighting.WeightingFactory
            counts = [2 1 0; 0 1 1; 1 0 3];

            w = WeightingFactory.create('tfidf');
            w.fit(counts);
            W = w.forward(counts);
            testCase.verifyTrue(w.isFitted());
            testCase.verifyEqual(size(W), size(counts));
        end
    end
end
