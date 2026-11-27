classdef TestBinanceHistoryAndSelectiveFilter < OceanTest.BaseTestCase
    methods (Test)
        function validatorAcceptsRegularWeeklyOhlc(testCase)
            time=(datetime(2020,1,6,'TimeZone','UTC')+calweeks(0:20))';
            closePrice=100+(0:20)';
            data=timetable(time,closePrice,closePrice+2,closePrice-2,closePrice, ...
                'VariableNames',{'Open','High','Low','Close'});
            report=OceanPipeline.Ingestion.MarketDataValidator.validate( ...
                data,'ExpectedStep',days(7));
            testCase.verifyTrue(report.IsValid);
            testCase.verifyEqual(report.MissingIntervals,0);
        end

        function validatorDetectsGap(testCase)
            time=(datetime(2020,1,6,'TimeZone','UTC')+calweeks([0:5 7:10]))';
            closePrice=100+(0:numel(time)-1)';
            data=timetable(time,closePrice,closePrice+2,closePrice-2,closePrice, ...
                'VariableNames',{'Open','High','Low','Close'});
            report=OceanPipeline.Ingestion.MarketDataValidator.validate( ...
                data,'ExpectedStep',days(7));
            testCase.verifyFalse(report.IsValid);
            testCase.verifyEqual(report.MissingIntervals,1);
        end

        function filterRecoversFourYearComponent(testCase)
            samplesPerYear=52;
            n=16*samplesPerYear;
            time=(datetime(2000,1,3)+calweeks(0:n-1))';
            index=(0:n-1)';
            signal=0.003*index+0.30*sin(2*pi*index/(4*samplesPerYear))+ ...
                0.12*sin(2*pi*index/samplesPerYear)+ ...
                0.04*cos(2*pi*index/(samplesPerYear/2));
            pipeline=OceanPipeline.Transforms.SelectiveFilterPipeline( ...
                'SamplesPerYear',samplesPerYear);
            result=pipeline.run(time,signal);
            testCase.verifyLessThan(abs(result.Audit.EstimatedPeriodYears-4),0.2);
            annualFrequency=1;
            [~,i1]=min(abs(result.Layers(1).Frequency-annualFrequency));
            [~,i2]=min(abs(result.Layers(2).Frequency-annualFrequency));
            testCase.verifyLessThan(result.Layers(2).Periodogram(i2), ...
                0.02*result.Layers(1).Periodogram(i1));
        end
    end
end
