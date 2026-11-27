classdef TestBinancePipeline < OceanTest.BaseTestCase
    % TESTBINANCEPIPELINE  Unit tests for Binance Data Ingestion & Preprocessing Pipeline.

    methods (Test)
        function testDataReaderConstruction(testCase)
            reader = OceanPipeline.Ingestion.BinanceDataReader(...
                'Symbol', 'BTCUSDT', 'MarketType', 'spot', 'Interval', '1h', 'Limit', 100);

            testCase.verifyEqual(reader.Symbol, "BTCUSDT");
            testCase.verifyEqual(reader.MarketType, "spot");
            testCase.verifyEqual(reader.Interval, "1h");
            testCase.verifyEqual(reader.Limit, 100);
            testCase.verifyEqual(reader.getSource(), "BinanceREST:spot:BTCUSDT:1h");
        end

        function testTSAProcessorRegularizationAndFeatures(testCase)
            % Construct synthetic sample timetable
            N = 100;
            t0 = datetime('2026-01-01 00:00:00', 'TimeZone', 'UTC');
            dt = t0 + hours(0:N-1)';
            prices = 50000 + 1000 * sin((1:N)' / 10);
            vol = 50 + 10 * cos((1:N)' / 5);

            TT = timetable(dt, prices, prices*1.01, prices*0.99, prices, vol, vol*50000, ...
                repmat(100, N, 1), vol*0.5, vol*25000, ...
                'VariableNames', {'Open', 'High', 'Low', 'Close', 'Volume', ...
                                  'QuoteAssetVolume', 'NumberOfTrades', ...
                                  'TakerBuyBaseVolume', 'TakerBuyQuoteVolume'});
            TT.Properties.DimensionNames{1} = 'Time';

            processor = OceanPipeline.Transforms.BinanceTSAProcessor('TimeStep', '1h', 'VolWindow', 12);

            % Regularize
            TT_reg = processor.regularizeTimetable(TT, '1h');
            testCase.verifyEqual(height(TT_reg), N);

            % Feature Engineering
            TT_feat = processor.engineerFeatures(TT_reg, 'VolWindow', 12);
            testCase.verifyTrue(ismember('LogReturn', TT_feat.Properties.VariableNames));
            testCase.verifyTrue(ismember('RealizedVol', TT_feat.Properties.VariableNames));
            testCase.verifyTrue(ismember('VolumeZScore', TT_feat.Properties.VariableNames));
            testCase.verifyTrue(ismember('TakerBuyRatio', TT_feat.Properties.VariableNames));
            testCase.verifyTrue(ismember('VWAP', TT_feat.Properties.VariableNames));
            testCase.verifyEqual(height(TT_feat), N - 1);
        end

        function testEDAComputation(testCase)
            N = 120;
            t0 = datetime('2026-01-01 00:00:00', 'TimeZone', 'UTC');
            dt = t0 + hours(0:N-1)';
            r = 0.01 * randn(N, 1);
            p = 60000 * exp(cumsum(r));

            TT = timetable(dt, p, p, p, p, repmat(100, N, 1), ...
                'VariableNames', {'Open', 'High', 'Low', 'Close', 'Volume'});
            TT.Properties.DimensionNames{1} = 'Time';

            processor = OceanPipeline.Transforms.BinanceTSAProcessor('MaxLag', 20);
            TT_feat = processor.engineerFeatures(TT, 'VolWindow', 10);

            eda = processor.runEDA(TT_feat, 'LogReturn', 'MaxLag', 20);
            testCase.verifyEqual(eda.VarName, "LogReturn");
            testCase.verifyEqual(length(eda.ACF), 21);
            testCase.verifyEqual(length(eda.PACF), 20);
            testCase.verifyTrue(isfield(eda, 'JarqueBeraStat'));
            testCase.verifyTrue(isfield(eda, 'ADF_IsStationary'));
        end

        function testTemporalTrainValTestSplit(testCase)
            N = 200;
            t0 = datetime('2026-01-01 00:00:00', 'TimeZone', 'UTC');
            dt = t0 + hours(0:N-1)';
            r = 0.01 * randn(N, 1);
            p = 60000 * exp(cumsum(r));

            TT = timetable(dt, p, p, p, p, repmat(100, N, 1), ...
                'VariableNames', {'Open', 'High', 'Low', 'Close', 'Volume'});
            TT.Properties.DimensionNames{1} = 'Time';

            processor = OceanPipeline.Transforms.BinanceTSAProcessor(...
                'TrainRatio', 0.70, 'ValRatio', 0.15);
            TT_feat = processor.engineerFeatures(TT, 'VolWindow', 10);

            splits = processor.splitTimeSeries(TT_feat, 'TrainRatio', 0.70, 'ValRatio', 0.15);

            nTotal = height(TT_feat);
            testCase.verifyEqual(height(splits.TT_Train) + height(splits.TT_Val) + height(splits.TT_Test), nTotal);

            % Chronological monotonicity check
            testCase.verifyTrue(splits.TT_Train.Time(end) < splits.TT_Val.Time(1));
            testCase.verifyTrue(splits.TT_Val.Time(end) < splits.TT_Test.Time(1));
        end
    end
end
