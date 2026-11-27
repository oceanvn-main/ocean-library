# CandlestickChart & CandlestickStrategy Renderer Guide

## Overview

`OceanViz.Charts.CandlestickChart` provides a performance-optimized graphics engine for rendering financial OHLCV candlestick and volume charts in MATLAB.

Regardless of the number of candles (N), the renderer maintains an **O(1) graphics handle complexity**:
- **High-Low Wicks**: Rendered via exactly 2 NaN-separated `line()` handle calls (1 Bullish, 1 Bearish).
- **Candle Bodies**: Rendered via exactly 2 4xM matrix `patch()` handle calls (1 Bullish, 1 Bearish).
- **Volume Bars**: Rendered via exactly 2 4xM matrix `patch()` handle calls (1 Bullish, 1 Bearish).

This avoids handle overhead and speeds up rendering by >50x compared to traditional `rectangle()` or `line()` loops.

---

## Usage

### 1. Static Convenient Call

You can call the static `plot` method directly with a `timetable`, `table`, or `struct`:

```matlab
fig = figure;
ax = axes(fig);
OceanViz.Charts.CandlestickChart.plot(ax, TT_spot, 'Title', 'BTC/USDT Spot Klines');
```

### 2. Object Instance Method

```matlab
chart = OceanViz.Charts.CandlestickChart(...
    'Theme', 'Dark', ...
    'MAWindows', [7, 25, 99], ...
    'ShowVolume', true);

chart.draw(ax, TT_spot, 'Title', 'Binance 1h Klines');
```

### 3. Integration with OceanViz.Core

```matlab
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('Binance OHLCV Chart');
ax = axes(fig);

data.Open   = TT_spot.Open;
data.High   = TT_spot.High;
data.Low    = TT_spot.Low;
data.Close  = TT_spot.Close;
data.Volume = TT_spot.Volume;
data.Time   = TT_spot.Time;

viz.plot(ax, 'Candlestick', data);
```

---

## Examples

### Input Data Formats Supported

`CandlestickChart` automatically parses and validates multiple data types:

1. **MATLAB `timetable`**: Expects variables `Open`, `High`, `Low`, `Close`, and optional `Volume`. Timestamps are extracted from `Time`.
2. **MATLAB `table`**: Expects columns `Open`, `High`, `Low`, `Close`, optional `Time`, and optional `Volume`.
3. **MATLAB `struct`**: Expects fields `.Open`, `.High`, `.Low`, `.Close`, optional `.Time`, and optional `.Volume`.

---

## See Also

- `OceanViz.Charts.CandlestickChart`
- `OceanViz.Strategies.CandlestickStrategy`
- `OceanViz.Core`
- `OceanPipeline.Ingestion.BinanceDataReader`
- `docs/recipes/candlestick_recipes.md`
