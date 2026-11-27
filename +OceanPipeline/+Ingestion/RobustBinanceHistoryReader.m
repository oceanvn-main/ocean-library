classdef RobustBinanceHistoryReader < handle
    % ROBUSTBINANCEHISTORYREADER Reliable, cached Binance OHLCV range reader.
    %
    % Composes BinanceDataReader and adds pagination, bounded retries,
    % closed-candle selection, cache reuse, and an auditable coverage report.

    properties
        Symbol (1,1) string = "BTCUSDT"
        MarketType (1,1) string = "spot"
        Interval (1,1) string = "1w"
        MaxPerRequest (1,1) double {mustBeInteger,mustBeBetween(MaxPerRequest,1,1000)} = 1000
        Timeout (1,1) double {mustBePositive} = 20
        MaxRetries (1,1) double {mustBeInteger,mustBeNonnegative} = 3
        BackoffSeconds (1,1) double {mustBeNonnegative} = 0.75
        CacheDirectory (1,1) string = ""
        ClosedCandlesOnly (1,1) logical = true
        Verbose (1,1) logical = true
        LastReport struct = struct()
    end

    properties (Access=private)
        Reader_
    end

    methods
        function obj = RobustBinanceHistoryReader(options)
            arguments
                options.Symbol (1,1) string = "BTCUSDT"
                options.MarketType (1,1) string = "spot"
                options.Interval (1,1) string = "1w"
                options.MaxPerRequest (1,1) double = 1000
                options.Timeout (1,1) double = 20
                options.MaxRetries (1,1) double = 3
                options.BackoffSeconds (1,1) double = 0.75
                options.CacheDirectory (1,1) string = ""
                options.ClosedCandlesOnly (1,1) logical = true
                options.Verbose (1,1) logical = true
            end
            obj.Symbol = upper(options.Symbol);
            obj.MarketType = lower(options.MarketType);
            obj.Interval = lower(options.Interval);
            obj.MaxPerRequest = options.MaxPerRequest;
            obj.Timeout = options.Timeout;
            obj.MaxRetries = options.MaxRetries;
            obj.BackoffSeconds = options.BackoffSeconds;
            obj.CacheDirectory = options.CacheDirectory;
            obj.ClosedCandlesOnly = options.ClosedCandlesOnly;
            obj.Verbose = options.Verbose;
            obj.Reader_ = OceanPipeline.Ingestion.BinanceDataReader( ...
                'Symbol',obj.Symbol,'MarketType',obj.MarketType, ...
                'Interval',obj.Interval,'Limit',obj.MaxPerRequest, ...
                'Timeout',obj.Timeout);
        end

        function result = readRange(obj,startTime,endTime,options)
            arguments
                obj
                startTime (1,1) datetime
                endTime (1,1) datetime
                options.UseCache (1,1) logical = true
                options.Refresh (1,1) logical = false
            end
            [startTime,endTime] = obj.asUtc_(startTime,endTime);
            if endTime <= startTime
                error('OceanPipeline:Binance:InvalidRange', ...
                    'EndTime must be later than StartTime.');
            end

            cachePath = obj.cachePath_();
            cachedData = timetable();
            cacheMeta = struct('RequestedStart',NaT,'RequestedEnd',NaT);
            cacheHit = false;
            if options.UseCache && ~options.Refresh && cachePath ~= "" && isfile(cachePath)
                stored = load(cachePath,'Data','CacheMeta');
                if isfield(stored,'Data') && isfield(stored,'CacheMeta')
                    cachedData = stored.Data;
                    cacheMeta = stored.CacheMeta;
                    coversStart = cacheMeta.RequestedStart <= startTime;
                    noNewClosedCandle = ~isempty(cachedData) && ...
                        cachedData.Time(end)+2*obj.intervalDuration_() > endTime;
                    coversEnd = cacheMeta.RequestedEnd >= endTime || noNewClosedCandle;
                    cacheHit = coversStart && coversEnd;
                end
            end

            pageFailures = strings(0,1);
            fetchedPages = 0;
            if cacheHit
                allData = cachedData;
                if obj.Verbose
                    fprintf('[Binance cache] Complete range hit: %s\n',cachePath);
                end
            else
                fetchStart = startTime;
                if ~isempty(cachedData) && ~options.Refresh && ...
                        ~isnat(cacheMeta.RequestedEnd) && cacheMeta.RequestedEnd > startTime
                    fetchStart = min(max(cachedData.Time(end)+obj.intervalDuration_(), ...
                        startTime),endTime);
                end
                if fetchStart < endTime
                    [freshData,fetchedPages,pageFailures] = obj.fetchPages_(fetchStart,endTime);
                else
                    freshData = timetable();
                end
                allData = obj.merge_(cachedData,freshData);
                cacheMeta.RequestedStart = obj.minDate_(cacheMeta.RequestedStart,startTime);
                cacheMeta.RequestedEnd = obj.maxDate_(cacheMeta.RequestedEnd,endTime);
                cacheMeta.UpdatedAt = datetime('now','TimeZone','UTC');
                if options.UseCache && cachePath ~= ""
                    cacheFolder = fileparts(cachePath);
                    if ~exist(cacheFolder,'dir'), mkdir(cacheFolder); end
                    Data = allData;
                    CacheMeta = cacheMeta;
                    save(cachePath,'Data','CacheMeta');
                end
            end

            data = allData(timerange(startTime,endTime,'closedleft'),:);
            if obj.ClosedCandlesOnly && ~isempty(data)
                cutoff = min(endTime,datetime('now','TimeZone','UTC'));
                isClosed = data.Time + obj.intervalDuration_() <= cutoff;
                data = data(isClosed,:);
            end
            if isempty(data)
                error('OceanPipeline:Binance:NoData', ...
                    'Binance returned no closed candles for %s in the requested range.',obj.Symbol);
            end

            quality = OceanPipeline.Ingestion.MarketDataValidator.validate( ...
                data,'ExpectedStep',obj.intervalDuration_());
            report = obj.makeReport_(startTime,endTime,data,quality, ...
                cacheHit,fetchedPages,pageFailures);
            obj.LastReport = report;
            if ~isempty(pageFailures)
                error('OceanPipeline:Binance:IncompleteFetch', ...
                    'Historical fetch was incomplete: %s',strjoin(pageFailures,' | '));
            end
            if ~quality.IsValid
                error('OceanPipeline:Binance:InvalidData', ...
                    'Fetched market data failed validation: %s',strjoin(quality.Issues,' | '));
            end
            result = struct('Data',data,'Coverage',report,'Quality',quality, ...
                'Source',obj.getSource());
        end

        function source = getSource(obj)
            source = sprintf('BinanceREST:%s:%s:%s', ...
                obj.MarketType,obj.Symbol,obj.Interval);
        end
    end

    methods (Access=private)
        function [data,pageCount,failures] = fetchPages_(obj,startTime,endTime)
            stepMs = milliseconds(obj.intervalDuration_());
            total = ceil(milliseconds(endTime-startTime)/stepMs);
            pageCount = ceil(total/obj.MaxPerRequest);
            pages = cell(pageCount,1);
            failures = strings(0,1);
            startMs = posixtime(startTime)*1000;
            endMs = posixtime(endTime)*1000;
            for pageIndex = 1:pageCount
                pageStart = startMs+(pageIndex-1)*obj.MaxPerRequest*stepMs;
                pageEnd = min(pageStart+obj.MaxPerRequest*stepMs,endMs);
                success = false;
                lastMessage = "";
                for attempt = 1:obj.MaxRetries+1
                    try
                        pages{pageIndex} = obj.Reader_.fetchKlinesTimetable( ...
                            'StartTime',pageStart,'EndTime',pageEnd-1, ...
                            'Limit',obj.MaxPerRequest);
                        success = true;
                        break
                    catch ME
                        lastMessage = string(ME.message);
                        if attempt <= obj.MaxRetries
                            pause(obj.BackoffSeconds*2^(attempt-1));
                        end
                    end
                end
                if ~success
                    failures(end+1,1) = sprintf('page %d: %s',pageIndex,lastMessage); %#ok<AGROW>
                elseif obj.Verbose
                    fprintf('[Binance] Page %d/%d: %d candles.\n', ...
                        pageIndex,pageCount,height(pages{pageIndex}));
                end
            end
            pages = pages(~cellfun(@isempty,pages));
            if isempty(pages), data = timetable(); else, data = vertcat(pages{:}); end
            data = obj.merge_(timetable(),data);
        end

        function report = makeReport_(obj,startTime,endTime,data,quality,cacheHit,pageCount,failures)
            actualYears = days(data.Time(end)-data.Time(1))/365.2425;
            report = struct( ...
                'RequestedStart',startTime,'RequestedEnd',endTime, ...
                'ActualStart',data.Time(1),'ActualEnd',data.Time(end), ...
                'RequestedYears',days(endTime-startTime)/365.2425, ...
                'ActualYears',actualYears,'ObservationCount',height(data), ...
                'FourYearCycles',actualYears/4,'StartTruncated',data.Time(1)>startTime+obj.intervalDuration_(), ...
                'EndTruncated',data.Time(end)+obj.intervalDuration_()<endTime-obj.intervalDuration_(), ...
                'CacheHit',cacheHit,'FetchedPages',pageCount, ...
                'PageFailures',failures,'MissingIntervals',quality.MissingIntervals);
        end

        function path = cachePath_(obj)
            if obj.CacheDirectory == "", path = ""; return; end
            filename = sprintf('binance_%s_%s_%s.mat', ...
                lower(obj.MarketType),lower(obj.Symbol),lower(obj.Interval));
            path = fullfile(obj.CacheDirectory,filename);
        end

        function durationValue = intervalDuration_(obj)
            names = ["1m","3m","5m","15m","30m","1h","2h","4h", ...
                "6h","8h","12h","1d","3d","1w"];
            secondsValue = [60 180 300 900 1800 3600 7200 14400 21600 ...
                28800 43200 86400 259200 604800];
            index = find(names==obj.Interval,1);
            if isempty(index)
                error('OceanPipeline:Binance:UnsupportedInterval', ...
                    'Unsupported Binance interval: %s',obj.Interval);
            end
            durationValue = seconds(secondsValue(index));
        end
    end

    methods (Static,Access=private)
        function merged = merge_(left,right)
            if isempty(left), merged=right; elseif isempty(right), merged=left; else, merged=[left;right]; end
            if isempty(merged), return; end
            merged = sortrows(merged,'Time');
            [~,keep] = unique(merged.Time,'last');
            merged = merged(sort(keep),:);
        end

        function [startTime,endTime] = asUtc_(startTime,endTime)
            startTime.TimeZone='UTC';
            endTime.TimeZone='UTC';
        end

        function value = minDate_(left,right)
            if isnat(left), value=right; else, value=min(left,right); end
        end

        function value = maxDate_(left,right)
            if isnat(left), value=right; else, value=max(left,right); end
        end
    end
end
