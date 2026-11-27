classdef RetrievalMetricFactoryTest < OceanTest.BaseTestCase
    % RETRIEVALMETRICFACTORYTEST  Unit tests for
    % OceanStats.Metrics.Retrieval.RetrievalMetricFactory and the
    % OceanStats.Metrics.Retrieval.IRetrievalMetric abstract contract.

    methods (Test)
        function testValidNamesListsAllSixMetrics(testCase)
            % Five per-query metrics plus 'map' (MeanAveragePrecision, the
            % multi-query extension of 'ap').
            import OceanStats.Metrics.Retrieval.RetrievalMetricFactory
            names = RetrievalMetricFactory.validNames();
            testCase.verifyEqual(sort(string(names)), ...
                sort(["ap", "map", "ndcg@k", "precision@k", "rr", "recall@k"]));
        end

        function testCreateMapReturnsMeanAveragePrecision(testCase)
            import OceanStats.Metrics.Retrieval.RetrievalMetricFactory
            m = RetrievalMetricFactory.create('map');
            testCase.verifyClass(m, 'OceanStats.Metrics.Retrieval.MeanAveragePrecision');
            testCase.verifyEqual(m.name(), "map");
        end

        function testCreateEachNameReturnsCorrectConcreteClass(testCase)
            import OceanStats.Metrics.Retrieval.RetrievalMetricFactory
            testCase.verifyClass(RetrievalMetricFactory.create('precision@k', 'K', 5), ...
                'OceanStats.Metrics.Retrieval.PrecisionAtK');
            testCase.verifyClass(RetrievalMetricFactory.create('recall@k', 'K', 5), ...
                'OceanStats.Metrics.Retrieval.RecallAtK');
            testCase.verifyClass(RetrievalMetricFactory.create('ap'), ...
                'OceanStats.Metrics.Retrieval.AveragePrecision');
            testCase.verifyClass(RetrievalMetricFactory.create('rr'), ...
                'OceanStats.Metrics.Retrieval.ReciprocalRank');
            testCase.verifyClass(RetrievalMetricFactory.create('ndcg@k', 'K', 5), ...
                'OceanStats.Metrics.Retrieval.NDCGAtK');
        end

        function testCreateIsCaseInsensitive(testCase)
            import OceanStats.Metrics.Retrieval.RetrievalMetricFactory
            m = RetrievalMetricFactory.create('PRECISION@K', 'K', 3);
            testCase.verifyClass(m, 'OceanStats.Metrics.Retrieval.PrecisionAtK');
        end

        function testKPassthroughReachesTheMetric(testCase)
            % The factory's name-value passthrough must forward 'K'
            % unmodified -- verify the constructed metric actually carries
            % the requested K, not just that construction succeeded.
            import OceanStats.Metrics.Retrieval.RetrievalMetricFactory
            m = RetrievalMetricFactory.create('ndcg@k', 'K', 7);
            testCase.verifyEqual(m.K, 7);
        end

        function testUnknownNameErrors(testCase)
            import OceanStats.Metrics.Retrieval.RetrievalMetricFactory
            testCase.verifyError(@() RetrievalMetricFactory.create('bm25@k'), ...
                'OceanStats:Metrics:Retrieval:RetrievalMetricFactory:UnknownMetric');
        end

        function testMissingRequiredKSurfacesTheMetricsOwnConstructorError(testCase)
            % Passthrough is unmodified -- a metric requiring K that
            % doesn't get one surfaces THAT metric's own arguments-block
            % error, not a factory-level error.
            import OceanStats.Metrics.Retrieval.RetrievalMetricFactory
            testCase.verifyError(@() RetrievalMetricFactory.create('precision@k'), ...
                'MATLAB:nonExistentField');
        end

        function testAbstractInterfaceCannotBeInstantiated(testCase)
            testCase.verifyError(@() OceanStats.Metrics.Retrieval.IRetrievalMetric(), ...
                'MATLAB:class:abstract');
        end
    end
end
