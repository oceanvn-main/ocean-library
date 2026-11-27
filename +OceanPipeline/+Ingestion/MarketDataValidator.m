classdef MarketDataValidator
    % MARKETDATAVALIDATOR Audits regular OHLCV timetable integrity.

    methods (Static)
        function report = validate(data,options)
            arguments
                data timetable
                options.ExpectedStep (1,1) duration {mustBePositive}
            end
            required = ["Open","High","Low","Close"];
            names = string(data.Properties.VariableNames);
            missingNames = required(~ismember(required,names));
            issues = strings(0,1);
            if ~isempty(missingNames)
                issues(end+1,1) = "Missing variables: "+strjoin(missingNames,', ');
                report = struct('IsValid',false,'Issues',issues, ...
                    'DuplicateCount',NaN,'MissingIntervals',NaN,'IrregularCount',NaN);
                return
            end

            rowTimes = data.Properties.RowTimes;
            duplicateCount = height(data)-numel(unique(rowTimes));
            if duplicateCount>0, issues(end+1,1)="Duplicate timestamps detected."; end
            nonfinite = any(~isfinite(data{:,cellstr(required)}),'all');
            if nonfinite, issues(end+1,1)="Nonfinite OHLC values detected."; end
            nonpositive = any(data{:,cellstr(required)}<=0,'all');
            if nonpositive, issues(end+1,1)="Nonpositive OHLC values detected."; end
            invalidOhlc = any(data.Low>min(data.Open,data.Close) | ...
                data.High<max(data.Open,data.Close) | data.Low>data.High);
            if invalidOhlc, issues(end+1,1)="OHLC ordering constraints violated."; end

            delta = diff(rowTimes);
            ratio = seconds(delta)/seconds(options.ExpectedStep);
            missingIntervals = sum(max(round(ratio)-1,0));
            irregularCount = sum(abs(ratio-round(ratio))>0.05);
            if missingIntervals>0
                issues(end+1,1)=sprintf('%d expected intervals are missing.',missingIntervals);
            end
            if irregularCount>0
                issues(end+1,1)=sprintf('%d timestamp spacings are irregular.',irregularCount);
            end
            report = struct('IsValid',isempty(issues),'Issues',issues, ...
                'DuplicateCount',duplicateCount,'MissingIntervals',missingIntervals, ...
                'IrregularCount',irregularCount);
        end
    end
end
