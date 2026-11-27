% EXP_BINANCE_TSA_PIPELINE
% End-to-End Binance Data Ingestion, Preprocessing, EDA, Population Spectrum,
% Autoregressive (AR) Model Fitting, and Multi-Step Forecasting.
%
% Objectives:
%   1. Fetch multi-kind data from Binance REST API (Spot Klines, Futures Klines, Orderbook, Recent Trades).
%   2. Regularise time series to strict hourly grid with causal missing value imputation.
%   3. Engineer time series features (Log returns, Realized Volatility, VWAP, Taker Ratio, Volume Z-Score).
%   4. Conduct Exploratory Data Analysis (ADF stationarity, ACF/PACF, Jarque-Bera test).
%   5. Compute Empirical Welch Spectral Density & Theoretical AR Population Spectrum.
%   6. Fit Autoregressive AR(p) model via Yule-Walker with automatic AIC/BIC lag selection.
%   7. Perform multi-step recursive forecasting on held-out Test set with 95% confidence bands.
%   8. Perform anti-leakage chronological Train/Validation/Test split with train-only scaling.
%   9. Save datasets to Medallion DataLake (Bronze/Silver Parquet files).
%  10. Visualize results exclusively using OceanViz.Core with Dark theme styling.
%
% Author: Ocean Library   Date: 2026-09-28

clc; clear; close all;
rng(42, 'twister');

fprintf("=====================================================================\n");
fprintf("   BINANCE TIME SERIES ANALYSIS (TSA) & SPECTRAL MODELING PIPELINE   \n");
fprintf("=====================================================================\n\n");

%% === 1. Data Ingestion via Binance REST API ===
SYMBOL     = "BTCUSDT";
INTERVAL   = "1h";
LIMIT_KLINE = 500;

fprintf("[Step 1] Ingesting multi-kind data from Binance REST API...\n");

% 1a. Spot Klines
readerSpot = OceanPipeline.Ingestion.BinanceDataReader(...
    'Symbol', SYMBOL, 'MarketType', 'spot', 'Interval', INTERVAL, 'Limit', LIMIT_KLINE);

try
    TT_spot = readerSpot.fetchKlinesTimetable();
    fprintf("   [SUCCESS] Spot Klines fetched: %d rows from %s to %s UTC\n", ...
        height(TT_spot), string(TT_spot.Time(1)), string(TT_spot.Time(end)));
catch ME
    fprintf("   [NOTICE] Live Binance Spot REST call failed (%s). Generating offline reference dataset.\n", ME.message);
    TT_spot = generateOfflineReferenceData(LIMIT_KLINE, hours(1));
end

% 1b. Futures Klines
readerFutures = OceanPipeline.Ingestion.BinanceDataReader(...
    'Symbol', SYMBOL, 'MarketType', 'futures', 'Interval', INTERVAL, 'Limit', LIMIT_KLINE);

try
    TT_fut = readerFutures.fetchKlinesTimetable();
    fprintf("   [SUCCESS] Futures Klines fetched: %d rows from %s to %s UTC\n", ...
        height(TT_fut), string(TT_fut.Time(1)), string(TT_fut.Time(end)));
catch ME
    fprintf("   [NOTICE] Live Binance Futures REST call failed (%s). Using synced reference dataset.\n", ME.message);
    TT_fut = TT_spot;
    TT_fut.Close = TT_fut.Close * 1.0005;
end

% 1c. Order Book Depth Snapshot
try
    depth = readerSpot.fetchDepth('Limit', 100);
    fprintf("   [SUCCESS] Order Book Snapshot: Best Bid = %.2f, Best Ask = %.2f, Spread = %.2f, Depth Imbalance = %.4f\n", ...
        depth.BestBid, depth.BestAsk, depth.Spread, depth.DepthImbalance);
catch ME
    fprintf("   [NOTICE] Order book snapshot unavailable (%s).\n", ME.message);
end

% 1d. Recent Execution Trades
try
    trades = readerSpot.fetchRecentTrades('Limit', 100);
    fprintf("   [SUCCESS] Recent Trades fetched: %d ticks from %s to %s UTC\n", ...
        height(trades), string(trades.Time(1)), string(trades.Time(end)));
catch ME
    fprintf("   [NOTICE] Recent trades unavailable (%s).\n", ME.message);
end

% 1e. 24-Hour Ticker Statistics
try
    ticker24h = readerSpot.fetchTicker24hr();
    fprintf("   [SUCCESS] 24hr Ticker Stats: PriceChange = %.2f (%.2f%%), High = %.2f, Low = %.2f, Vol = %.2f\n", ...
        ticker24h.PriceChange, ticker24h.PriceChangePercent, ticker24h.HighPrice, ticker24h.LowPrice, ticker24h.Volume);
catch ME
    fprintf("   [NOTICE] 24hr Ticker stats unavailable (%s).\n", ME.message);
end

%% === 2. Time Series Preprocessing & Feature Engineering ===
fprintf("\n[Step 2] Executing Preprocessing & Feature Engineering...\n");
processor = OceanPipeline.Transforms.BinanceTSAProcessor(...
    'TimeStep', INTERVAL, 'VolWindow', 24, 'MaxLag', 40, 'TrainRatio', 0.70, 'ValRatio', 0.15);

% 2a. Regularise timetable and apply causal missing value imputation
TT_reg = processor.regularizeTimetable(TT_spot, INTERVAL);
fprintf("   - Timetable regularised to grid '%s': %d samples.\n", INTERVAL, height(TT_reg));

% 2b. Compute time series features
TT_feat = processor.engineerFeatures(TT_reg, 'VolWindow', 24);
fprintf("   - Engineered Features: LogReturn, PriceDiff, RealizedVol, VolumeZScore, TakerBuyRatio, VWAP.\n");


%% 3. Anti Leakage Temporal Train / Val/ Test Split

fprintf("\n[Step 3] Performing Anti-Leakage Chronological Train / Val / Test Split...\n");
splits = processor.splitTimeSeries(TT_feat, 'TrainRatio', 0.70, 'ValRatio', 0.15);

TT_train = splits.TT_Train;
TT_val   = splits.TT_Val;
TT_test  = splits.TT_Test;

fprintf("   - Train Set: %d rows (%s to %s)\n", ...
    height(TT_train), string(TT_train.Time(1)), string(TT_train.Time(end)));
fprintf("   - Val Set:   %d rows (%s to %s)\n", ...
    height(TT_val), string(TT_val.Time(1)), string(TT_val.Time(end)));
fprintf("   - Test Set:  %d rows (%s to %s)\n", ...
    height(TT_test), string(TT_test.Time(1)), string(TT_test.Time(end)));
%% EDA and Population Spectrum

