# Binance TSA Pipeline Recipes

Copy-paste recipes for fetching Binance market data, preprocessing time series, running EDA diagnostics, and saving train/val/test splits to the Parquet DataLake.

---

## Recipe 1: Ingesting Spot Klines & Order Book Depth

```matlab
% Ingest Spot Klines & Order Book Depth from Binance REST API
reader = OceanPipeline.Ingestion.BinanceDataReader('Symbol', 'ETHUSDT', 'MarketType', 'spot', 'Interval', '15m', 'Limit', 500);

% Fetch Candlesticks
TT_klines = reader.fetchKlinesTimetable();

% Fetch Order Book Depth
depth = reader.fetchDepth('Limit', 100);
fprintf('Best Bid: %.2f | Best Ask: %.2f | Spread: %.2f | Imbalance: %.4f\n', ...
    depth.BestBid, depth.BestAsk, depth.Spread, depth.DepthImbalance);
```

---

## Recipe 2: Feature Engineering & Stationarity EDA

```matlab
% Regularise timetable & compute features
processor = OceanPipeline.Transforms.BinanceTSAProcessor('TimeStep', '15m', 'VolWindow', 20);

TT_reg  = processor.regularizeTimetable(TT_klines, '15m');
TT_feat = processor.engineerFeatures(TT_reg, 'VolWindow', 20);

% Run stationarity & distribution EDA on Log Returns
edaRet = processor.runEDA(TT_feat, 'LogReturn', 'MaxLag', 30);

fprintf('Mean: %.6f | Std: %.6f\n', edaRet.Mean, edaRet.Std);
fprintf('ADF Stationary: %s (P-Val: %.4f)\n', string(edaRet.ADF_IsStationary), edaRet.ADF_PValue);
fprintf('Jarque-Bera Normality P-Val: %.4e\n', edaRet.JarqueBeraPVal);
```

---

## Recipe 3: Anti-Leakage Split & Parquet DataLake Storage

```matlab
% Anti-leakage chronological split (70% Train, 15% Val, 15% Test)
splits = processor.splitTimeSeries(TT_feat, 'TrainRatio', 0.70, 'ValRatio', 0.15);

% Scale Val and Test using Train-only statistics (Zero Data Leakage)
trainLogRetScaled = splits.TT_TrainScale.LogReturn;
valLogRetScaled   = splits.TT_ValScale.LogReturn;
testLogRetScaled  = splits.TT_TestScale.LogReturn;

% Export to Medallion DataLake
processor.exportToLake(splits, 'binance_ethusdt_15m', './lake/');
```

---

## Recipe 4: Plotting Time Series EDA Dashboard with OceanViz

```matlab
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('Binance EDA Dashboard');
ax  = axes(fig);

dataTS.Y = edaRet.LogReturn;
dataTS.ACF = edaRet.ACF;
dataTS.PACF = edaRet.PACF;
dataTS.Lags = 30;
dataTS.ModelName = 'ETHUSDT 15m Log Returns';

viz.plot(ax, 'TimeSeries', dataTS, 'ShowLegend', true);
```
