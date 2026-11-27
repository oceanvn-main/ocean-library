classdef BinanceDataReader < OceanPipeline.Ingestion.IDataReader
    % BINANCEDATAREADER  Data ingestion reader for Binance REST API.
    %
    %   Fetches Spot and USDT-M Futures market data directly from Binance
    %   public REST endpoints (no API keys required for public endpoints).
    %
    %   Supported Data Kinds:
    %     1. Spot & Futures Candlesticks / Klines (OHLCV + Taker Volume)
    %     2. Order Book Depth Snapshots (Bids, Asks, Imbalance)
    %     3. Recent Tick Execution Trades
    %     4. 24-Hour Ticker Price Statistics
    %
    %   Syntax:
    %     reader = OceanPipeline.Ingestion.BinanceDataReader('Symbol', 'BTCUSDT', ...
    %                                                        'MarketType', 'spot', ...
    %                                                        'Interval', '1h', ...
    %                                                        'Limit', 500);
    %     TT = reader.fetchKlinesTimetable();
    %     depth = reader.fetchDepth();
    %     trades = reader.fetchRecentTrades();
    %
    %   See also: OceanPipeline.Ingestion.IDataReader, timetable, webread

    properties
        Symbol     (1,1) string = "BTCUSDT"
        MarketType (1,1) string = "spot"     % "spot" or "futures"
        Interval   (1,1) string = "1h"       % "1m","5m","15m","1h","4h","1d"
        Limit      (1,1) double = 500        % Max 1000 per REST request
        BaseUrl    (1,1) string = "https://api.binance.com"
        Timeout    (1,1) double = 15
    end

    methods
        function obj = BinanceDataReader(options)
            % BINANCEDATAREADER  Constructor with Name-Value arguments.
            arguments
                options.Symbol     (1,1) string = "BTCUSDT"
                options.MarketType (1,1) string = "spot"
                options.Interval   (1,1) string = "1h"
                options.Limit      (1,1) double = 500
                options.Timeout    (1,1) double = 15
                options.BaseUrl    (1,1) string = ""
            end

            obj.Symbol     = upper(options.Symbol);
            obj.MarketType = lower(options.MarketType);
            obj.Interval   = lower(options.Interval);
            obj.Limit      = options.Limit;
            obj.Timeout    = options.Timeout;

            if options.BaseUrl ~= ""
                obj.BaseUrl = options.BaseUrl;
            elseif obj.MarketType == "futures"
                obj.BaseUrl = "https://fapi.binance.com";
            else
                obj.BaseUrl = "https://api.binance.com";
            end
        end

        function src = getSource(obj)
            % GETSOURCE  Identify this reader source string.
            src = sprintf("BinanceREST:%s:%s:%s", obj.MarketType, obj.Symbol, obj.Interval);
        end

        function result = read(obj)
            % READ  Implements IDataReader.read contract.
            %   Returns struct(data, meta) where data is (N x 5) [Open, High, Low, Close, Volume].
            TT = obj.fetchKlinesTimetable();
            data = [TT.Open, TT.High, TT.Low, TT.Close, TT.Volume];
            header = {'Open', 'High', 'Low', 'Close', 'Volume'};
            meta = obj.buildMeta(data, obj.getSource(), header);
            meta.Time = TT.Time;
            result.data = data;
            result.meta = meta;
        end

        function TT = fetchKlinesTimetable(obj, options)
            % FETCHKLINESTIMETABLE  Fetch candlestick / kline data as a MATLAB timetable.
            arguments
                obj
                options.Symbol   (1,1) string = ""
                options.Interval (1,1) string = ""
                options.Limit    (1,1) double = 0
                options.StartTime double = []
                options.EndTime   double = []
            end

            symbol   = obj.Symbol;   if options.Symbol ~= "",   symbol   = upper(options.Symbol);   end
            interval = obj.Interval; if options.Interval ~= "", interval = lower(options.Interval); end
            limit    = obj.Limit;    if options.Limit > 0,      limit    = options.Limit;           end

            endpoint = "/api/v3/klines";
            if obj.MarketType == "futures"
                endpoint = "/fapi/v1/klines";
            end

            url = sprintf("%s%s?symbol=%s&interval=%s&limit=%d", ...
                obj.BaseUrl, endpoint, symbol, interval, limit);

            if ~isempty(options.StartTime)
                url = sprintf("%s&startTime=%.0f", url, options.StartTime);
            end
            if ~isempty(options.EndTime)
                url = sprintf("%s&endTime=%.0f", url, options.EndTime);
            end

            wopts = weboptions('Timeout', obj.Timeout);
            try
                rawResponse = webread(url, wopts);
            catch ME
                error("BinanceDataReader:FetchFailed", ...
                    "Failed to fetch Binance klines from %s: %s", url, ME.message);
            end

            % Process Binance array response
            % Format: [ OpenTime, Open, High, Low, Close, Volume, CloseTime, QuoteVolume, Trades, TakerBaseVol, TakerQuoteVol, Ignore ]
            n = length(rawResponse);
            if n == 0
                error("BinanceDataReader:EmptyData", "Binance API returned empty dataset.");
            end

            openTimeMs = zeros(n, 1);
            openPrice  = zeros(n, 1);
            highPrice  = zeros(n, 1);
            lowPrice   = zeros(n, 1);
            closePrice = zeros(n, 1);
            volume     = zeros(n, 1);
            quoteVol   = zeros(n, 1);
            numTrades  = zeros(n, 1);
            takerBase  = zeros(n, 1);
            takerQuote = zeros(n, 1);

            for i = 1:n
                item = rawResponse{i};
                openTimeMs(i) = item{1};
                openPrice(i)  = str2double(string(item{2}));
                highPrice(i)  = str2double(string(item{3}));
                lowPrice(i)   = str2double(string(item{4}));
                closePrice(i) = str2double(string(item{5}));
                volume(i)     = str2double(string(item{6}));
                quoteVol(i)   = str2double(string(item{8}));
                numTrades(i)  = str2double(string(item{9}));
                takerBase(i)  = str2double(string(item{10}));
                takerQuote(i) = str2double(string(item{11}));
            end

            dt = datetime(openTimeMs / 1000, 'ConvertFrom', 'posixtime', 'TimeZone', 'UTC');

            TT = timetable(dt, openPrice, highPrice, lowPrice, closePrice, volume, ...
                quoteVol, numTrades, takerBase, takerQuote, ...
                'VariableNames', {'Open', 'High', 'Low', 'Close', 'Volume', ...
                                  'QuoteAssetVolume', 'NumberOfTrades', ...
                                  'TakerBuyBaseVolume', 'TakerBuyQuoteVolume'});
            TT.Properties.DimensionNames{1} = 'Time';
            TT = sortrows(TT, 'Time');
        end

        function depth = fetchDepth(obj, options)
            % FETCHDEPTH  Fetch market order book snapshot.
            arguments
                obj
                options.Symbol (1,1) string = ""
                options.Limit  (1,1) double = 100
            end

            symbol = obj.Symbol; if options.Symbol ~= "", symbol = upper(options.Symbol); end
            limit  = options.Limit;

            endpoint = "/api/v3/depth";
            if obj.MarketType == "futures"
                endpoint = "/fapi/v1/depth";
            end

            url = sprintf("%s%s?symbol=%s&limit=%d", obj.BaseUrl, endpoint, symbol, limit);
            wopts = weboptions('Timeout', obj.Timeout);

            try
                raw = webread(url, wopts);
            catch ME
                error("BinanceDataReader:DepthFailed", ...
                    "Failed to fetch order book depth: %s", ME.message);
            end

            bidsCell = raw.bids;
            asksCell = raw.asks;

            nBids = length(bidsCell);
            nAsks = length(asksCell);

            bidPrice = zeros(nBids, 1);
            bidQty   = zeros(nBids, 1);
            for i = 1:nBids
                bidPrice(i) = str2double(string(bidsCell{i}{1}));
                bidQty(i)   = str2double(string(bidsCell{i}{2}));
            end

            askPrice = zeros(nAsks, 1);
            askQty   = zeros(nAsks, 1);
            for i = 1:nAsks
                askPrice(i) = str2double(string(asksCell{i}{1}));
                askQty(i)   = str2double(string(asksCell{i}{2}));
            end

            bidsTable = table(bidPrice, bidQty, 'VariableNames', {'Price', 'Quantity'});
            asksTable = table(askPrice, askQty, 'VariableNames', {'Price', 'Quantity'});

            bestBid = max(bidPrice);
            bestAsk = min(askPrice);
            spread  = bestAsk - bestBid;
            midPrice = (bestBid + bestAsk) / 2;

            totalBidVol = sum(bidQty);
            totalAskVol = sum(askQty);
            imbalance   = (totalBidVol - totalAskVol) / (totalBidVol + totalAskVol + 1e-12);

            depth.Symbol          = symbol;
            depth.MarketType      = obj.MarketType;
            depth.LastUpdateId    = raw.lastUpdateId;
            depth.Bids            = bidsTable;
            depth.Asks            = asksTable;
            depth.BestBid         = bestBid;
            depth.BestAsk         = bestAsk;
            depth.Spread          = spread;
            depth.MidPrice        = midPrice;
            depth.DepthImbalance  = imbalance;
            depth.Timestamp       = datetime('now', 'TimeZone', 'UTC');
        end

        function TT = fetchRecentTrades(obj, options)
            % FETCHRECENTTRADES  Fetch recent execution ticks as a timetable.
            arguments
                obj
                options.Symbol (1,1) string = ""
                options.Limit  (1,1) double = 500
            end

            symbol = obj.Symbol; if options.Symbol ~= "", symbol = upper(options.Symbol); end
            limit  = options.Limit;

            endpoint = "/api/v3/trades";
            if obj.MarketType == "futures"
                endpoint = "/fapi/v1/trades";
            end

            url = sprintf("%s%s?symbol=%s&limit=%d", obj.BaseUrl, endpoint, symbol, limit);
            wopts = weboptions('Timeout', obj.Timeout);

            try
                raw = webread(url, wopts);
            catch ME
                error("BinanceDataReader:TradesFailed", ...
                    "Failed to fetch recent trades: %s", ME.message);
            end

            n = length(raw);
            if n == 0
                error("BinanceDataReader:EmptyTrades", "No recent trades returned.");
            end

            tradeId   = zeros(n, 1);
            price     = zeros(n, 1);
            qty       = zeros(n, 1);
            quoteQty  = zeros(n, 1);
            timeMs    = zeros(n, 1);
            isBuyerMaker = false(n, 1);

            for i = 1:n
                if iscell(raw)
                    item = raw{i};
                else
                    item = raw(i);
                end
                tradeId(i)   = item.id;
                price(i)     = str2double(string(item.price));
                qty(i)       = str2double(string(item.qty));
                if isfield(item, 'quoteQty')
                    quoteQty(i) = str2double(string(item.quoteQty));
                else
                    quoteQty(i) = price(i) * qty(i);
                end
                timeMs(i)    = item.time;
                isBuyerMaker(i) = item.isBuyerMaker;
            end

            dt = datetime(timeMs / 1000, 'ConvertFrom', 'posixtime', 'TimeZone', 'UTC');

            TT = timetable(dt, tradeId, price, qty, quoteQty, isBuyerMaker, ...
                'VariableNames', {'TradeId', 'Price', 'Quantity', 'QuoteQuantity', 'IsBuyerMaker'});
            TT.Properties.DimensionNames{1} = 'Time';
            TT = sortrows(TT, 'Time');
        end

        function stats = fetchTicker24hr(obj, options)
            % FETCHTICKER24HR  Fetch 24-hour price change statistics.
            arguments
                obj
                options.Symbol (1,1) string = ""
            end

            symbol = obj.Symbol; if options.Symbol ~= "", symbol = upper(options.Symbol); end

            endpoint = "/api/v3/ticker/24hr";
            if obj.MarketType == "futures"
                endpoint = "/fapi/v1/ticker/24hr";
            end

            url = sprintf("%s%s?symbol=%s", obj.BaseUrl, endpoint, symbol);
            wopts = weboptions('Timeout', obj.Timeout);

            try
                raw = webread(url, wopts);
            catch ME
                error("BinanceDataReader:TickerFailed", ...
                    "Failed to fetch 24hr ticker: %s", ME.message);
            end

            stats.Symbol             = string(raw.symbol);
            stats.PriceChange        = str2double(string(raw.priceChange));
            stats.PriceChangePercent = str2double(string(raw.priceChangePercent));
            stats.WeightedAvgPrice   = str2double(string(raw.weightedAvgPrice));
            stats.LastPrice          = str2double(string(raw.lastPrice));
            stats.HighPrice          = str2double(string(raw.highPrice));
            stats.LowPrice           = str2double(string(raw.lowPrice));
            stats.Volume             = str2double(string(raw.volume));
            stats.QuoteVolume        = str2double(string(raw.quoteVolume));
        end
    end
end
