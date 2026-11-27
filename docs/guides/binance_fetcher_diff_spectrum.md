# BinanceFetcher & DifferenceSpectrumFilter Guide

## Overview

This guide covers two new components added to `+OceanPipeline` on 2026-10-02:

| Class | Package | Purpose |
|---|---|---|
| `BinanceFetcher` | `OceanPipeline.Ingestion` | Windowed Binance OHLCV data fetch with auto-pagination |
| `DifferenceSpectrumFilter` | `OceanPipeline.Transforms` | Nth-order differencing with before/after power spectrum analysis |

---

## BinanceFetcher

### Purpose

`BinanceDataReader` fetches a fixed number of candles (`Limit`). `BinanceFetcher` wraps it to let you say **"give me the last 1 week of hourly data"** — it computes the correct `startTime`/`endTime`, splits into Binance-safe page sizes (≤ 1000 per request), and merges the pages into one timetable.

### Supported Windows

| String | Duration |
|---|---|
| `'1d'` | 1 calendar day |
| `'3d'` | 3 calendar days |
| `'1w'` | 7 calendar days |
| `'1m'` | 30 calendar days |

Any valid MATLAB duration string (`'48h'`, `'2d'`) also works.

### Usage

```matlab
% Fetch 1 week of hourly BTCUSDT spot candles
fetcher = OceanPipeline.Ingestion.BinanceFetcher( ...
    'Symbol',     'BTCUSDT', ...
    'Window',     '1w', ...
    'Interval',   '1h', ...
    'MarketType', 'spot');
TT = fetcher.fetch();

% Fetch a custom date range
tStart = datetime(2024, 1, 1, 'TimeZone', 'UTC');
tEnd   = datetime(2024, 1, 15, 'TimeZone', 'UTC');
TT2 = fetcher.fetchRange(tStart, tEnd);
```

### Output Timetable

| Column | Type | Description |
|---|---|---|
| `Time` | datetime (UTC) | Candle open timestamp |
| `Open, High, Low, Close` | double | OHLC prices (USDT) |
| `Volume` | double | Base asset volume |
| `QuoteAssetVolume` | double | Quote asset volume |
| `NumberOfTrades` | double | Trades per candle |
| `TakerBuyBaseVolume` | double | Taker buy volume |
| `TakerBuyQuoteVolume` | double | Taker buy quote volume |

### Pagination Logic

```
startMs = posixtime(tStart) * 1000
endMs   = posixtime(tEnd)   * 1000
while pageStart < endMs:
    pageEnd = min(pageStart + MaxPerRequest * intervalMs, endMs)
    fetch page [pageStart, pageEnd)
    pageStart = pageEnd
```

---

## DifferenceSpectrumFilter

### Purpose

Applies the backshift-difference operator **(1-B)^d** to a signal and computes the one-sided Welch power spectral density (PSD) **before and after** differencing. The theoretical frequency-domain transfer function `|H(f)| = 2|sin(pi*f/fs)|` is overlaid for verification.

### Spectral Interpretation

| Order d | Transfer Function | Effect |
|---|---|---|
| 1 | `|H(f)| = 2|sin(pi*f/fs)|` | Removes DC; attenuates low-freq power |
| 2 | `|H(f)|^2` | Steeper low-freq attenuation (removes linear trends) |
| d | `|H(f)|^d` | Polynomial trend removal up to degree d-1 |

### Usage

```matlab
dsf = OceanPipeline.Transforms.DifferenceSpectrumFilter( ...
    'Order',          1, ...        % differencing order
    'Overlap',        0.5, ...      % Welch overlap fraction
    'WindowType',     'hann', ...   % Welch window
    'VerifyParseval', true);        % warn if Parseval error > 1%

report = dsf.filter(x, fs);        % x: signal vector, fs: sampling freq
dsf.printSummary(report);
```

### Report Fields

| Field | Description |
|---|---|
| `x_orig` | Original signal |
| `x_diff` | Differenced signal (length N-d) |
| `f_orig / psd_orig` | Frequency axis & PSD before differencing |
| `f_diff / psd_diff` | Frequency axis & PSD after differencing |
| `tf_gain` | Theoretical gain `|H(f)|^d` |
| `parseval_err_orig/diff` | Parseval % error (< 1% = acceptable) |
| `var_orig / var_diff` | Signal variance before/after |
| `summary` | Printable one-line summary |

### Parseval Verification

The one-sided Welch PSD should integrate to the signal variance:

```
integral_0^{fs/2} PSD(f) df  ≈  var(x)
```

A warning fires if the error exceeds 1%. Common causes: very short signals, window mismatch, or non-ergodic data.

---

## See Also

- `OceanPipeline.Ingestion.BinanceDataReader` — direct candle fetcher
- `OceanPipeline.Transforms.BinanceTSAProcessor` — full TSA preprocessing pipeline
- `OceanMath.Operators.DifferenceOperator1D` — sparse matrix implementation of (1-B)^d
- `OceanMath.Signal.SpectralAnalyzer` — general spectral analysis utilities
- Experiment: `exp_binance_fetch_and_diff_spectrum.m` — real Binance market data experiment
- Synthetic Experiment: `exp_synthetic_diff_spectrum_verify.m` — synthetic dataset filter verification experiment
