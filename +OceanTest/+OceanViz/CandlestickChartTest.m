classdef CandlestickChartTest < OceanTest.BaseTestCase
    % CANDLESTICKCHARTTEST Unit tests for OceanViz.Charts.CandlestickChart

    methods (Test)
        function testTimetableRendering(testCase)
            % Test rendering with a synthetic timetable
            t = (datetime('2026-09-01') : hours(1) : datetime('2026-09-05'))';
            N = length(t);
            openP  = 100 + cumsum(randn(N, 1));
            highP  = openP + abs(randn(N, 1));
            lowP   = openP - abs(randn(N, 1));
            closeP = openP + randn(N, 1);
            vol    = randi([100, 1000], N, 1);

            TT = timetable(t, openP, highP, lowP, closeP, vol, ...
                'VariableNames', {'Open', 'High', 'Low', 'Close', 'Volume'});
            TT.Properties.DimensionNames{1} = 'Time';

            fig = figure('Visible', 'off');
            cleanupObj = onCleanup(@() close(fig));
            ax = axes(fig);

            layout = OceanViz.Charts.CandlestickChart.plot(ax, TT, 'Title', 'Test Timetable');
            testCase.verifyNotEmpty(layout);
        end

        function testStructRenderingWithoutVolume(testCase)
            % Test rendering with a struct without Volume
            N = 50;
            data.Open  = 100 + randn(N, 1);
            data.High  = data.Open + 2;
            data.Low   = data.Open - 2;
            data.Close = data.Open + randn(N, 1);
            data.Title = "Test Struct No Vol";

            fig = figure('Visible', 'off');
            cleanupObj = onCleanup(@() close(fig));
            ax = axes(fig);

            chart = OceanViz.Charts.CandlestickChart('Theme', 'Light', 'ShowVolume', false);
            layout = chart.draw(ax, data);
            testCase.verifyNotEmpty(layout);
        end

        function testOceanVizCoreIntegration(testCase)
            % Test rendering via OceanViz.Core with 'Candlestick' strategy
            N = 30;
            data.Open  = 50000 + randn(N, 1)*100;
            data.High  = data.Open + 500;
            data.Low   = data.Open - 500;
            data.Close = data.Open + randn(N, 1)*100;
            data.Volume = randi([50, 500], N, 1);

            viz = OceanViz.Core('Theme', 'Dark');
            fig = viz.createFigure('Test Core Integration');
            cleanupObj = onCleanup(@() close(fig));
            ax = axes(fig);

            viz.plot(ax, 'Candlestick', data);
            testCase.verifyTrue(isvalid(fig));
        end
    end
end
