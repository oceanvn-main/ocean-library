classdef BinanceFetcher < handle
    % BINANCEFETCHER  Windowed Binance data acquisition with auto-pagination.
    %
    %   Fetches a user-specified historical window ('1d', '3d', '1w', '1m')
    %   of candlestick (OHLCV) data from the Binance REST API and returns a
    %   single merged timetable. Automatically splits the window into pages
    %   of at most MaxPerRequest candles to respect the Binance 1000-candle
    %   per-request limit.
    %
    %   Supported Window Strings
    %     '1d'  — last calendar day     (24 × 1-hour candles, or 288 × 5-minute, etc.)
    %     '3d'  — last 3 calendar days
    %     '1w'  — last 7 calendar days
    %     '1m'  — last 30 calendar days
    %     Or any valid MATLAB duration string: '48h', '12h', etc.
    %
    %   Supported Interval Strings (Binance klines)
    %     '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h',
    %     '12h', '1d', '3d', '1w'
    %
    %   Syntax
    %     fetcher = OceanPipeline.Ingestion.BinanceFetcher( ...
    %                   'Symbol', 'BTCUSDT', ...
    %                   'Window', '1w', ...
    %                   'Interval', '1h', ...
    %                   'MarketType', 'spot');
    %     TT = fetcher.fetch();
    %
    %   Output timetable columns (inherited from BinanceDataReader)
    %     Time, Open, High, Low, Close, Volume,
    %     QuoteAssetVolume, NumberOfTrades,
    %     TakerBuyBaseVolume, TakerBuyQuoteVolume
    %
    %   See also: OceanPipeline.Ingestion.BinanceDataReader, timetable

    properties
        Symbol        (1,1) string = "BTCUSDT"
        Window        (1,1) string = "1d"    % '1d','3d','1w','1m' or duration string
        Interval      (1,1) string = "1h"    % Binance kline interval string
        MarketType    (1,1) string = "spot"  % 'spot' or 'futures'
        MaxPerRequest (1,1) double = 1000    % Binance hard limit per REST call
        Timeout       (1,1) double = 15      % HTTP timeout in seconds
        Verbose       (1,1) logical = true   % Print progress
    end

    properties (Access = private)
        Reader  % BinanceDataReader instance
    end

    % -----------------------------------------------------------------------
    methods
        function obj = BinanceFetcher(options)
            % BINANCEFETCHER  Constructor with Name-Value arguments.
            arguments
                options.Symbol        (1,1) string  = "BTCUSDT"
                options.Window        (1,1) string  = "1d"
                options.Interval      (1,1) string  = "1h"
                options.MarketType    (1,1) string  = "spot"
                options.MaxPerRequest (1,1) double  = 1000
                options.Timeout       (1,1) double  = 15
                options.Verbose       (1,1) logical = true
            end

            obj.Symbol        = upper(options.Symbol);
            obj.Window        = lower(options.Window);
            obj.Interval      = lower(options.Interval);
            obj.MarketType    = lower(options.MarketType);
            obj.MaxPerRequest = options.MaxPerRequest;
            obj.Timeout       = options.Timeout;
            obj.Verbose       = options.Verbose;

            obj.Reader = OceanPipeline.Ingestion.BinanceDataReader( ...
                'Symbol',     obj.Symbol, ...
                'MarketType', obj.MarketType, ...
                'Interval',   obj.Interval, ...
                'Limit',      obj.MaxPerRequest, ...
                'Timeout',    obj.Timeout);
        end

        % -------------------------------------------------------------------
        function TT = fetch(obj)
            % FETCH  Fetch candles for the configured window.
            %
            %   TT = fetch() fetches all candles in the time window that
            %   ends at 'now' (UTC). Pages are concatenated and duplicates
            %   removed. The result is sorted by ascending time.

            [tStart, tEnd] = obj.computeWindow();
            TT = obj.fetchRange(tStart, tEnd);
        end

        % -------------------------------------------------------------------
        function TT = fetchRange(obj, tStart, tEnd)
            % FETCHRANGE  Fetch candles between two datetime values (UTC).
            %
            %   tStart — MATLAB datetime (UTC) start of range (inclusive)
            %   tEnd   — MATLAB datetime (UTC) end   of range (exclusive)

            arguments
                obj
                tStart (1,1) datetime
                tEnd   (1,1) datetime
            end

            % Convert to POSIX milliseconds (as required by Binance REST)
            startMs = posixtime(tStart) * 1000;
            endMs   = posixtime(tEnd)   * 1000;

            % Interval duration in milliseconds
            intervalDur = obj.intervalToMilliseconds(obj.Interval);

            % Expected total candles — used to plan pages
            totalCandles = ceil((endMs - startMs) / intervalDur);

            if obj.Verbose
                fprintf("[BinanceFetcher] Window: %s  -->  %s  UTC\n", ...
                    char(tStart), char(tEnd));
                fprintf("[BinanceFetcher] Interval: %s  |  Expected candles: %d\n", ...
                    obj.Interval, totalCandles);
            end

            % -------- Auto-pagination ----------------------------------------
            pages = {};
            pageStart = startMs;
            pageNum   = 0;

            while pageStart < endMs
                pageEnd  = min(pageStart + obj.MaxPerRequest * intervalDur, endMs);
                pageNum  = pageNum + 1;

                if obj.Verbose
                    tS = datetime(pageStart / 1000, 'ConvertFrom', 'posixtime', 'TimeZone', 'UTC');
                    tE = datetime(pageEnd   / 1000, 'ConvertFrom', 'posixtime', 'TimeZone', 'UTC');
                    fprintf("  [Page %d] %s  -->  %s\n", pageNum, char(tS), char(tE));
                end

                try
                    page = obj.Reader.fetchKlinesTimetable( ...
                        'StartTime', pageStart, ...
                        'EndTime',   pageEnd - 1);   % -1 ms: exclusive end
                    if height(page) > 0
                        pages{end+1} = page; %#ok<AGROW>
                    end
                catch ME
                    warning("BinanceFetcher:PageFailed", ...
                        "Page %d fetch failed: %s", pageNum, ME.message);
                end

                % Advance to the next page window
                pageStart = pageEnd;
            end

            % -------- Merge pages -------------------------------------------
            if isempty(pages)
                error("BinanceFetcher:NoData", ...
                    "No data returned for symbol=%s window=%s interval=%s.", ...
                    obj.Symbol, obj.Window, obj.Interval);
            end

            TT = vertcat(pages{:});
            % Remove duplicates caused by overlapping page boundaries
            [~, ia] = unique(TT.Time);
            TT = TT(ia, :);
            TT = sortrows(TT, 'Time');

            if obj.Verbose
                fprintf("[BinanceFetcher] Fetched %d candles over %d page(s).\n", ...
                    height(TT), pageNum);
            end
        end

        % -------------------------------------------------------------------
        function [tStart, tEnd] = computeWindow(obj)
            % COMPUTEWINDOW  Resolve window string to [tStart, tEnd] datetimes.

            tEnd   = datetime('now', 'TimeZone', 'UTC');
            tStart = tEnd - obj.parseWindow(obj.Window);
        end
    end

    % -----------------------------------------------------------------------
    methods (Access = private)

        function dur = parseWindow(~, windowStr)
            % PARSEWINDOW  Convert window shorthand or duration string to MATLAB duration.

            switch lower(windowStr)
                case '1d'
                    dur = days(1);
                case '3d'
                    dur = days(3);
                case '1w'
                    dur = days(7);
                case '1m'
                    dur = days(30);
                otherwise
                    % Try MATLAB's built-in duration parser: e.g., '48h', '2d'
                    try
                        dur = duration(windowStr);
                    catch
                        error("BinanceFetcher:InvalidWindow", ...
                            "Unrecognised window string '%s'. Use '1d','3d','1w','1m', " + ...
                            "or a valid MATLAB duration string (e.g. '48h').", windowStr);
                    end
            end
        end

        function ms = intervalToMilliseconds(~, intervalStr)
            % INTERVALTOMILLISECONDS  Convert Binance interval string to milliseconds.
            %   Reference: Binance kline interval definitions.

            % Map: interval string -> seconds
            map = struct( ...
                'x1m',  60,       'x3m',  180,      'x5m',  300, ...
                'x15m', 900,      'x30m', 1800,     'x1h',  3600, ...
                'x2h',  7200,     'x4h',  14400,    'x6h',  21600, ...
                'x8h',  28800,    'x12h', 43200,    'x1d',  86400, ...
                'x3d',  259200,   'x1w',  604800);

            key = ['x' lower(intervalStr)];
            if isfield(map, key)
                ms = map.(key) * 1000;
            else
                error("BinanceFetcher:UnknownInterval", ...
                    "Unrecognised Binance interval '%s'.", intervalStr);
            end
        end
    end
end
