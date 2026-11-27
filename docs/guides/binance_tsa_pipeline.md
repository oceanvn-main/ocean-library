# Binance Time Series Analysis (TSA) Ingestion & Preprocessing Pipeline

## Overview

The `BinanceDataReader` and `BinanceTSAProcessor` provide a complete end-to-end data pipeline for fetching crypto market data directly from the Binance public REST API (Spot & USDT-M Futures), cleaning and regularising the time series, engineering financial time series features, performing Exploratory Data Analysis (EDA) & stationarity testing, and performing anti-leakage temporal Train/Validation/Test splitting.

The pipeline adheres strictly to the Ocean Library **MLOps Medallion Architecture** (`/datascience-pipeline`), ensuring data provenance and zero test-set leakage.

---

## Architecture & Components

```
           +---------------------------------------------+
           |           Binance REST API                  |
           | (Spot / Futures Klines, Depth, Trades)      |
           +---------------------------------------------+
                                  |
                                  v
           +---------------------------------------------+
           | OceanPipeline.Ingestion.BinanceDataReader   |
           |  - fetchKlinesTimetable()                   |
           |  - fetchDepth()                             |
           |  - fetchRecentTrades()                      |
           |  - fetchTicker24hr()                        |
           +---------------------------------------------+
                                  |
                                  v
           +---------------------------------------------+
           | OceanPipeline.Transforms.BinanceTSAProcessor|
           |  1. regularizeTimetable()  (Causal Fill)    |
           |  2. engineerFeatures()     (Returns/Vol/VWAP)|
           |  3. runEDA()               (ADF/ACF/PACF)   |
           |  4. splitTimeSeries()      (Chronological)  |
           |  5. exportToLake()         (Bronze/Silver)  |
           +---------------------------------------------+
                                  |
                                  v
           +---------------------------------------------+
           | OceanViz.Core  (Dark-Theme Dashboard)       |
           +---------------------------------------------+
```

---

## Features & Methods

### 1. `BinanceDataReader` (`+OceanPipeline/+Ingestion`)

Inherits from `OceanPipeline.Ingestion.IDataReader`.

- **`fetchKlinesTimetable('Symbol', 'BTCUSDT', 'Interval', '1h', 'Limit', 500)`**: Fetches OHLCV candlestick data into a MATLAB `timetable`.
- **`fetchDepth('Limit', 100)`**: Returns a struct containing Bids, Asks, Bid-Ask Spread, MidPrice, and Order Book Depth Imbalance.
- **`fetchRecentTrades('Limit', 500)`**: Returns a timetable of recent tick executions.
- **`fetchTicker24hr()`**: Returns 24-hour price change and volume statistics.

### 2. `BinanceTSAProcessor` (`+OceanPipeline/+Transforms`)

- **`regularizeTimetable(TT, timeStep)`**: Enforces evenly spaced timestamps (`retime`) and applies **causal forward-fill** (`fillmissing(..., 'previous')`) to prevent future-data leakage.
- **`engineerFeatures(TT, 'VolWindow', 24)`**: Computes:
  - Log Returns: r(t) = ln(Close(t) / Close(t-1))
  - Rolling Volatility (`RollingVol` / `RealizedVol` alias): sigma(t) = std(r_{t-W+1:t}) rolling standard deviation of log returns over window W
  - Volume Z-Score & Taker Buy Ratio: V_taker / V_total
  - Rolling Volume-Weighted Average Price (`VWAP` / `TypicalVWAP` / `CloseVWAP`):
    - **Exact Exchange VWAP** (`VWAP`): Uses Binance `QuoteAssetVolume` sum_i (trade_price_i * trade_qty_i) divided by sum_i (trade_qty_i) over 24h.
    - **Typical Price VWAP** (`TypicalVWAP`): Uses TP_t = (High_t + Low_t + Close_t) / 3 weighted by volume.
    - **Close-Weighted VWAP** (`CloseVWAP`): Uses Close_t weighted by volume.
- **`runEDA(TT, varName, 'MaxLag', 40)`** — returns a struct with the following fields:
  - **Stationarity**: `ADF_IsStationary`, `ADF_PValue`, `ADF_TestStatistic` via `adftest`
  - **Autocorrelation**: `ACF`, `ACFLags`, `PACF`, `PACFLags`, `ConfBound`
  - **Normality**: `JarqueBeraStat`, `JarqueBeraPVal` (chi-2 distribution, 2 DOF)
  - **Raw Periodogram** (exact FFT, Parseval-consistent one-sided): `Periodogram_Power`, `Periodogram_Freqs` (angular rad/sample)
  - **Welch PSD** (smoothed): `PSD_Freqs` (cycles/sample), `PSD_Power`, `PSD_AngularFreq` (rad/sample)
  - **Period Domain**: `PSD_PeriodHours` (converts angular frequency to period in hours for 1h data)
  - **Parseval Verification**: `Parseval_TotalArea` (numeric integration of raw periodogram), `Parseval_Variance` (population variance) — difference should be O(1e-8) or better
  - **Dominant Cycle**: `DominantPeriodHours`, `DominantPower` (peak frequency from Welch PSD)
- **`splitTimeSeries(TT, 'TrainRatio', 0.70, 'ValRatio', 0.15)`**: Chronologically splits data into Train, Validation, and Test sets and scales features using **train-only z-score parameters**.
- **`exportToLake(splits, datasetName, rootDir)`**: Saves datasets to Parquet format in the Medallion DataLake (Bronze raw & Silver processed).

---

## Quick Example

```matlab
% Instantiate Reader & Processor
reader = OceanPipeline.Ingestion.BinanceDataReader('Symbol', 'BTCUSDT', 'Interval', '1h', 'Limit', 500);
processor = OceanPipeline.Transforms.BinanceTSAProcessor('TimeStep', '1h', 'VolWindow', 24);

% Ingest
TT_raw = reader.fetchKlinesTimetable();

% Preprocess & Engineer Features
TT_reg  = processor.regularizeTimetable(TT_raw, '1h');
TT_feat = processor.engineerFeatures(TT_reg, 'VolWindow', 24);

% Run EDA Diagnostics
eda = processor.runEDA(TT_feat, 'LogReturn', 'MaxLag', 40);
fprintf('ADF Stationary: %s (P-Value: %.4f)\n', string(eda.ADF_IsStationary), eda.ADF_PValue);

% Split & Export
splits = processor.splitTimeSeries(TT_feat, 'TrainRatio', 0.70, 'ValRatio', 0.15);
processor.exportToLake(splits, 'binance_btcusdt', './lake/');
```

---

## Visualizations Generated

The experiment script `exp_binance_tsa_pipeline.m` generates 5 OceanViz dark-theme figures:

| Figure | Contents |
|--------|----------|
| **Figure 1** | Binance-style OHLCV Candlestick (MA7 / MA25 / MA99) + Volume bars |
| **Figure 2** | EDA Dashboard: Log Return series, ACF, PACF, Population Spectrum vs Welch PSD |
| **Figure 3** | AR(p) Model: AIC/BIC lag selection, multi-step forecast with 95% confidence bands, residual ACF |
| **Figure 4** | Engineered Features & Temporal Splits: Close+VWAP, Realized Volatility, Volume Z-Score |
| **Figure 5** | Power Spectrum Dashboard: Linear PSD, Log PSD (dB), Period-domain spectrum (with 24h/8h markers), Cumulative Parseval verification |

---

## Spectral Analysis Notes

- **Parseval's Theorem**: The total area under the raw FFT periodogram equals the signal variance. The pipeline verifies this numerically; a typical residual of < 1e-7 confirms correctness.
- **Period-Domain Spectrum** (Subplot 5C): Transforms angular frequency w to period T = 2*pi/w in hours, making dominant cycles directly interpretable against crypto trading cycles (24h daily, 8h funding rate, 4h session).
- **Dominant Period**: Identified as the peak of the smoothed Welch PSD in the period domain and printed to the console.
- **AR Population Spectrum**: The theoretical AR(p) spectrum, computed from the Yule-Walker coefficients as P(w) = sigma_e^2 / (2*pi * |A(e^{-jw})|^2), is overlaid on the empirical Welch PSD for model-fit assessment.

---

## See Also

- `OceanPipeline.Ingestion.BinanceDataReader`
- `OceanPipeline.Transforms.BinanceTSAProcessor`
- `OceanViz.Strategies.CandlestickStrategy`
- `Experiments/Scripts/Time Series Analysis/exp_binance_tsa_pipeline.m`
- `OceanViz.Core`
- `docs/recipes/binance_tsa_recipes.md`
