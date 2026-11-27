# Candlestick Chart Render Recipes

Copy-pasteable MATLAB recipes for rendering OHLCV Candlestick & Volume charts.

---

## Recipe 1: Render Candlestick Chart from Binance Timetable

```matlab
% 1. Ingest Binance Spot Data
reader = OceanPipeline.Ingestion.BinanceDataReader(...
    'Symbol', 'BTCUSDT', ...
    'MarketType', 'spot', ...
    'Interval', '1h', ...
    'Limit', 300);

TT_spot = reader.fetchKlinesTimetable();

% 2. Render Candlestick Chart directly
fig = figure('Name', 'Binance BTC/USDT Spot Klines');
ax  = axes(fig);

OceanViz.Charts.CandlestickChart.plot(ax, TT_spot, ...
    'Title', 'BTC/USDT 1h Klines', ...
    'Theme', 'Dark');
```

---

## Recipe 2: Render Candlestick Chart via OceanViz.Core

```matlab
% 1. Create Core Visualizer Hub
viz = OceanViz.Core('Theme', 'Dark');

% 2. Prepare Data Struct
data.Open   = TT_spot.Open;
data.High   = TT_spot.High;
data.Low    = TT_spot.Low;
data.Close  = TT_spot.Close;
data.Volume = TT_spot.Volume;
data.Time   = TT_spot.Time;

% 3. Plot via Strategy Name
fig = viz.createFigure('Binance OHLCV Strategy');
ax  = axes(fig);
viz.plot(ax, 'Candlestick', data);
```

---

## Recipe 3: Customized Candlestick Chart Object Instance

```matlab
chart = OceanViz.Charts.CandlestickChart(...
    'Theme', 'Dark', ...
    'MAWindows', [5, 10, 20], ...
    'BullColor', [0.10, 0.85, 0.50], ...
    'BearColor', [0.95, 0.20, 0.30]);

fig = figure;
ax = axes(fig);
chart.draw(ax, TT_spot, 'Title', 'Customized Moving Average Windows');
```
