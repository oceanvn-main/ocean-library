# BinanceFetcher & DifferenceSpectrumFilter — Recipes

## Recipe 1: Fetch 1-day hourly data

```matlab
fetcher = OceanPipeline.Ingestion.BinanceFetcher( ...
    'Symbol', 'ETHUSDT', 'Window', '1d', 'Interval', '1h');
TT = fetcher.fetch();
disp(TT(1:5, {'Open','High','Low','Close','Volume'}));
```

## Recipe 2: Fetch 1-week 15-minute data

```matlab
fetcher = OceanPipeline.Ingestion.BinanceFetcher( ...
    'Symbol', 'BTCUSDT', 'Window', '1w', 'Interval', '15m', ...
    'MarketType', 'spot', 'Verbose', false);
TT = fetcher.fetch();
fprintf('Fetched %d candles\n', height(TT));
```

## Recipe 3: Fetch a custom date range

```matlab
fetcher = OceanPipeline.Ingestion.BinanceFetcher( ...
    'Symbol', 'BTCUSDT', 'Window', '1w', 'Interval', '1h');
tStart = datetime(2024, 6, 1, 'TimeZone', 'UTC');
tEnd   = datetime(2024, 6, 15, 'TimeZone', 'UTC');
TT = fetcher.fetchRange(tStart, tEnd);
```

## Recipe 4: Apply 1st-order difference spectrum filter

```matlab
% Assume TT is a timetable with Close column
logRet = diff(log(TT.Close));   % log-returns
fs = 1/3600;                    % 1 sample/hour in Hz

dsf = OceanPipeline.Transforms.DifferenceSpectrumFilter('Order', 1);
report = dsf.filter(logRet, fs);
dsf.printSummary(report);
```

## Recipe 5: Compare orders 1, 2, 3

```matlab
fs = 1/3600;
logRet = diff(log(TT.Close));

for d = 1:3
    dsf = OceanPipeline.Transforms.DifferenceSpectrumFilter('Order', d);
    r = dsf.filter(logRet, fs);
    fprintf('Order %d: var_diff=%.4e, dB_reduction=%.1f\n', ...
        d, r.var_diff, 10*log10(r.var_diff/r.var_orig));
end
```

## Recipe 6: Plot PSD before and after (manual)

```matlab
dsf = OceanPipeline.Transforms.DifferenceSpectrumFilter('Order', 2);
r = dsf.filter(logRet, 1/3600);

figure;
subplot(1,2,1);
semilogy(r.f_orig, r.psd_orig);
title('PSD — Original'); xlabel('Frequency (Hz)'); ylabel('PSD');
subplot(1,2,2);
semilogy(r.f_diff, r.psd_diff);
title('PSD — 2nd-order differenced'); xlabel('Frequency (Hz)');
```

## Recipe 7: Full pipeline (fetch + filter + report)

```matlab
% 1. Fetch 3-day hourly data
fetcher = OceanPipeline.Ingestion.BinanceFetcher( ...
    'Symbol', 'BTCUSDT', 'Window', '3d', 'Interval', '1h');
TT = fetcher.fetch();

% 2. Log-returns
logRet = diff(log(TT.Close));

% 3. Filter with order 1 and 2
for d = [1, 2]
    dsf = OceanPipeline.Transforms.DifferenceSpectrumFilter( ...
        'Order', d, 'VerifyParseval', true);
    r = dsf.filter(logRet, 1/3600);
    dsf.printSummary(r);
end

## Recipe 8: Synthetic verification suite

```matlab
% Run the full synthetic benchmark suite verifying trend elimination,
% spectral shaping accuracy, random walk stationarization, and gain shift.
run('Experiments/Scripts/Time Series Analysis/exp_synthetic_diff_spectrum_verify.m');
```
```
