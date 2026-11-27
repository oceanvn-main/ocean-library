# Binance Long-Cycle Analysis

`OceanPipeline.Ingestion.RobustBinanceHistoryReader` reads explicit Binance
OHLCV ranges with bounded retries, closed-candle selection, incremental caching,
validation, and coverage provenance. It composes the existing
`BinanceDataReader`; existing ingestion APIs remain compatible.

For a proposed four-year BTC cycle, use native weekly candles. Intraday data
adds cost but no useful resolution at this horizon. A 12-year request currently
returns only the Binance BTCUSDT listing history (from August 2017), so coverage
metadata must accompany every result.

`OceanPipeline.Transforms.SelectiveFilterPipeline` exposes four sequential
layers: linear detrending, annual notch, half-year notch, and causal 13-week
smoothing. Each layer returns the aligned time series, raw periodogram, filter
coefficients, and cumulative transfer gain. Showing the gain is essential:
filters can amplify or suppress spectral peaks and must not be treated as
neutral evidence.

The experiment
`Experiments/Scripts/Time Series Analysis/exp_hamilton_ch6_selective_filter.m`
uses the OceanViz Scientific theme and the optimized Candlestick chart. It
reports the dominant period in a predeclared 2-6 year band, frequency
resolution, peak-to-band-median power, and the number of observable four-year
cycles.

With about 2.28 observable cycles, a peak near four years is descriptive and
does not establish a stable periodic law.
