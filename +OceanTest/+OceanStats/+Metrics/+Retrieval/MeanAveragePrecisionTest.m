classdef MeanAveragePrecisionTest < OceanTest.BaseTestCase
    % MEANAVERAGEPRECISIONTEST  Unit tests for
    % OceanStats.Metrics.Retrieval.MeanAveragePrecision (IIR eq. 11.16:
    % MAP = (1/|Q|) * sum_q AP(q)).
    %
    % Expected values are written as explicit arithmetic IN THIS FILE from
    % the AP definition (mean of Precision@k over relevant-hit ranks) —
    % never by calling MeanAveragePrecision/AveragePrecision to produce their
    % own expectation.
    %
    %   Query A: rankedIdx = [3 7 1 9 2 5 8 4 6], relevance over 10 docs =
    %            [0 3 2 0 1 0 0 2 0 0]  -> relevant docs {2,3,5,8}, R = 4.
    %            Hits at ranks 1 (doc 3), 5 (doc 2), 6 (doc 5), 7 (doc 8):
    %            AP_A = (1/1 + 2/5 + 3/6 + 4/7) / 4
    %   Query B: rankedIdx = [1 2 3], relevance = [0 1 1] -> R = 2.
    %            Hits at ranks 2 (doc 2), 3 (doc 3):
    %            AP_B = (1/2 + 2/3) / 2
    %   Query C: relevance = [0 0 0] -> no relevant docs, AP undefined (NaN).

    properties (Constant)
        RankedA = [3 7 1 9 2 5 8 4 6]
        RelA    = [0 3 2 0 1 0 0 2 0 0]
        RankedB = [1 2 3]
        RelB    = [0 1 1]
        RankedC = [1 2 3]
        RelC    = [0 0 0]
    end

    methods (Test)
        function testSingleQueryMapEqualsAp(testCase)
            % The text's own remark: "The MAP for the single query (hence = AP)".
            import OceanStats.Metrics.Retrieval.MeanAveragePrecision
            m = MeanAveragePrecision();
            apB = (1/2 + 2/3) / 2;

            mapScore = m.aggregate({testCase.RankedB}, {testCase.RelB});

            testCase.verifyEqual(mapScore, apB, 'AbsTol', 1e-12);
        end

        function testTwoQueryHandWorkedMap(testCase)
            import OceanStats.Metrics.Retrieval.MeanAveragePrecision
            m = MeanAveragePrecision();
            apA = (1/1 + 2/5 + 3/6 + 4/7) / 4;
            apB = (1/2 + 2/3) / 2;
            expectedMap = (apA + apB) / 2;

            [mapScore, perQueryAP, numExcluded] = m.aggregate( ...
                {testCase.RankedA, testCase.RankedB}, {testCase.RelA, testCase.RelB});

            testCase.verifyEqual(mapScore, expectedMap, 'AbsTol', 1e-12);
            testCase.verifyEqual(perQueryAP, [apA apB], 'AbsTol', 1e-12);
            testCase.verifyEqual(numExcluded, 0);
        end

        function testUndefinedQueryExcludedAndCounted(testCase)
            % A query with zero relevant docs has AP = NaN. It must be
            % EXCLUDED from the mean (not averaged in as 0, not poisoning the
            % result to NaN) and the exclusion must be REPORTED.
            import OceanStats.Metrics.Retrieval.MeanAveragePrecision
            m = MeanAveragePrecision();
            apA = (1/1 + 2/5 + 3/6 + 4/7) / 4;
            apB = (1/2 + 2/3) / 2;

            [mapScore, perQueryAP, numExcluded] = m.aggregate( ...
                {testCase.RankedA, testCase.RankedB, testCase.RankedC}, ...
                {testCase.RelA,    testCase.RelB,    testCase.RelC});

            testCase.verifyEqual(mapScore, (apA + apB) / 2, 'AbsTol', 1e-12);   % NOT (apA+apB+0)/3
            testCase.verifyTrue(isnan(perQueryAP(3)));
            testCase.verifyEqual(numExcluded, 1);
        end

        function testAllQueriesUndefinedGivesNaNNotZero(testCase)
            import OceanStats.Metrics.Retrieval.MeanAveragePrecision
            m = MeanAveragePrecision();
            [mapScore, ~, numExcluded] = m.aggregate( ...
                {testCase.RankedC, testCase.RankedC}, {testCase.RelC, testCase.RelC});
            testCase.verifyTrue(isnan(mapScore));
            testCase.verifyEqual(numExcluded, 2);
        end

        function testEmptyQuerySetGivesNaN(testCase)
            import OceanStats.Metrics.Retrieval.MeanAveragePrecision
            m = MeanAveragePrecision();
            [mapScore, perQueryAP, numExcluded] = m.aggregate({}, {});
            testCase.verifyTrue(isnan(mapScore));
            testCase.verifyEmpty(perQueryAP);
            testCase.verifyEqual(numExcluded, 0);
        end

        function testQueryCountMismatchErrors(testCase)
            import OceanStats.Metrics.Retrieval.MeanAveragePrecision
            m = MeanAveragePrecision();
            testCase.verifyError( ...
                @() m.aggregate({testCase.RankedA, testCase.RankedB}, {testCase.RelA}), ...
                'OceanStats:Metrics:Retrieval:MeanAveragePrecision:QueryCountMismatch');
        end

        function testIsAnExtensionOfAveragePrecisionWithUnchangedForward(testCase)
            % Liskov check: still an AveragePrecision / IRetrievalMetric, and
            % the inherited per-query forward() is byte-identical to AP's.
            import OceanStats.Metrics.Retrieval.MeanAveragePrecision
            import OceanStats.Metrics.Retrieval.AveragePrecision
            m = MeanAveragePrecision();

            testCase.verifyTrue(isa(m, 'OceanStats.Metrics.Retrieval.AveragePrecision'));
            testCase.verifyTrue(isa(m, 'OceanStats.Metrics.Retrieval.IRetrievalMetric'));
            testCase.verifyEqual( ...
                m.forward(testCase.RankedA, testCase.RelA), ...
                AveragePrecision().forward(testCase.RankedA, testCase.RelA));
            testCase.verifyEqual(m.name(), "map");
        end

        function testEvaluatorMeanRowEqualsAggregate(testCase)
            % Two routes to MAP must agree: RetrievalEvaluator's Mean row for
            % this metric's column vs. aggregate() on the same rankings.
            import OceanStats.Metrics.Retrieval.MeanAveragePrecision
            corpus = ["the cat sat on the mat", "the dog ran in the park", ...
                      "cats and dogs are pets", "rockets launch into space", ...
                      "the moon orbits earth"];
            queries = ["cat", "space"];
            relevance = {[1 0 1 0 0], [0 0 0 1 1]};

            ranker = OceanNLP.Retrieval.BM25Ranker();
            ranker.fit(corpus);
            m = MeanAveragePrecision();
            ev = OceanNLP.Retrieval.RetrievalEvaluator(ranker, {m});
            report = ev.evaluate(queries, relevance);

            ranked = cell(1, numel(queries));
            for q = 1:numel(queries)
                ranked{q} = ranker.rank(queries(q));
            end
            [mapDirect, perQueryAP] = m.aggregate(ranked, relevance);

            col = report.("map");
            testCase.verifyEqual(col(end), mapDirect, 'AbsTol', 1e-12);          % Mean row
            testCase.verifyEqual(col(1:numel(queries))', perQueryAP, 'AbsTol', 1e-12);
        end
    end
end
