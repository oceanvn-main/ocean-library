# Binance Long-Cycle Recipes

## Read cached weekly history

```matlab
reader = OceanPipeline.Ingestion.RobustBinanceHistoryReader( ...
    'Symbol','BTCUSDT', ...
    'Interval','1w', ...
    'CacheDirectory',fullfile(tempdir,'ocean-binance'));

endTime = datetime('now','TimeZone','UTC');
result = reader.readRange(endTime-calyears(12),endTime);
weekly = result.Data;
disp(result.Coverage)
disp(result.Quality)
```

The first call fetches Binance. Later calls reuse the validated cache and fetch
only an uncovered tail.

## Run the transparent filter pipeline

```matlab
pipeline = OceanPipeline.Transforms.SelectiveFilterPipeline( ...
    'SamplesPerYear',365.2425/7, ...
    'SmoothWindow',13, ...
    'SearchPeriodYears',[2 6], ...
    'TargetPeriodYears',4);

analysis = pipeline.run(weekly.Time,log(weekly.Close));
fprintf('Estimated long period: %.2f years\n', ...
    analysis.Audit.EstimatedPeriodYears);
```

Inspect `analysis.Layers(k).CumulativeGain` alongside each periodogram. Do not
interpret a post-filter peak without checking how the filter changes power at
that frequency.
