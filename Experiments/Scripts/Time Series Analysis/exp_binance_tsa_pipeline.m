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

%% === 3. Anti-Leakage Temporal Train / Val / Test Split ===
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

%% === 4. Exploratory Data Analysis & Population Spectrum ===
fprintf("\n[Step 4] Running Exploratory Data Analysis & Population Spectrum Analysis...\n");

% 4a. EDA Diagnostics on Train Log Returns
yTrain = TT_train.LogReturn;
edaRet = processor.runEDA(TT_train, 'LogReturn', 'MaxLag', 40);

fprintf("   Train Log Return Diagnostics:\n");
fprintf("     * Mean: %.6f | Std: %.6f | Skewness: %.4f | Kurtosis: %.4f\n", ...
    edaRet.Mean, edaRet.Std, edaRet.Skewness, edaRet.Kurtosis);
fprintf("     * ADF Stationarity Test: Stationary = %s | P-Value = %.4e | Statistic = %.4f\n", ...
    string(edaRet.ADF_IsStationary), edaRet.ADF_PValue, edaRet.ADF_TestStatistic);
fprintf("     * Jarque-Bera Normality Test Stat: %.2f (P-Value = %.4e)\n", ...
    edaRet.JarqueBeraStat, edaRet.JarqueBeraPVal);
fprintf("     * Dominant Spectral Period: %.2f hours (Power = %.4e)\n", ...
    edaRet.DominantPeriodHours, edaRet.DominantPower);
fprintf("     * Parseval Theorem Verification: Total Area = %.6e | Sample Variance = %.6e (Diff = %.4e)\n", ...
    edaRet.Parseval_TotalArea, edaRet.Parseval_Variance, abs(edaRet.Parseval_TotalArea - edaRet.Parseval_Variance));

% 4b. Fit AR Model & Select Optimal Lag Order via BIC (OceanMath.TimeSeries.Models.AR)
%
% Design note: parfor is used only when a pool is ALREADY open.
% Pool cold-start cost (~2-5s) exceeds the sequential sweep (~15-30ms) for
% maxOrder=10, so blind parfor would be SLOWER. With an active pool the
% per-order fits are independent (AR is a value class) and execute in parallel.
maxOrder = 10;
aicVals  = zeros(maxOrder, 1);
bicVals  = zeros(maxOrder, 1);
arModels = cell(maxOrder, 1);

pool = gcp('nocreate'); % Query open pool — does NOT start one
if ~isempty(pool)
    % Pool is already open: parallel sweep at zero startup cost.
    % yTrain is broadcast (read-only); arModels is sliced output.
    parfor pSweep = 1:maxOrder
        m = OceanMath.TimeSeries.Models.AR.fit(yTrain, 'Order', pSweep);
        arModels{pSweep} = m;
        sigSq = m.NoiseStd^2;
        aicVals(pSweep) = length(yTrain) * log(sigSq) + 2 * (pSweep + 1);
        bicVals(pSweep) = length(yTrain) * log(sigSq) + (pSweep + 1) * log(length(yTrain));
    end
else
    % No active pool: sequential sweep (faster than spawning workers for 10 iters).
    for pSweep = 1:maxOrder
        arModels{pSweep} = OceanMath.TimeSeries.Models.AR.fit(yTrain, 'Order', pSweep);
        sigSq = arModels{pSweep}.NoiseStd^2;
        aicVals(pSweep) = length(yTrain) * log(sigSq) + 2 * (pSweep + 1);
        bicVals(pSweep) = length(yTrain) * log(sigSq) + (pSweep + 1) * log(length(yTrain));
    end
end

[~, bestP] = min(bicVals);
arModel = arModels{bestP}; % Reuse cached model — no extra Yule-Walker solve

fprintf("\n   [TSA Model] Fitted Autoregressive Model:\n");
fprintf("     * Optimal AR Order (BIC): p = %d\n", bestP);
fprintf("     * Equation: %s\n", arModel.getEquation());
fprintf("     * Estimated Noise Std (sigma_e): %.6f\n", arModel.NoiseStd);
fprintf("     * Stationarity Check: %s\n", string(arModel.isStationary()));

% 4c. Compute Theoretical AR Population Spectrum vs Empirical Welch Spectrum
w = linspace(0, pi, 256); % Normalized frequency [0, pi]
aPoly = [1, -arModel.ARCoeffs];
popSpectrum = (arModel.NoiseStd^2 / (2 * pi)) ./ (abs(polyval(aPoly, exp(-1i * w))).^2);

%% === 5. Multi-Step Forecasting on Test Set ===
fprintf("\n[Step 5] Evaluating Multi-Step Forecasts on Test Set...\n");

yTest = TT_test.LogReturn;
horizon = length(yTest);
[yHat, bands] = arModel.forecast(yTrain, horizon);

testErrors = yTest - yHat;
rmseTest = sqrt(mean(testErrors.^2));
maeTest  = mean(abs(testErrors));

insideBands = (yTest >= bands(:, 1)) & (yTest <= bands(:, 2));
coverage = sum(insideBands) / horizon * 100;

fprintf("   - Forecast Horizon: %d hours (Test Set)\n", horizon);
fprintf("   - Forecast RMSE:    %.6f\n", rmseTest);
fprintf("   - Forecast MAE:     %.6f\n", maeTest);
fprintf("   - 95%% Coverage:     %.1f%%\n", coverage);

%% === 6. Medallion DataLake Export (Parquet) ===
fprintf("\n[Step 6] Exporting to Medallion DataLake (Parquet)...\n");
lakeRes = processor.exportToLake(splits, "binance_" + lower(SYMBOL), "./lake/");
fprintf("   - DataLake Export Status: %s\n", lakeRes.Status);

%% === 7. Visualization with OceanViz Engine ===
fprintf("\n[Step 7] Rendering Visualizations with OceanViz Engine...\n");
viz = OceanViz.Core('Theme', 'Dark');
colors = viz.Colors;

% ---------------------------------------------------------------------
% Figure 1: Raw Data Binance OHLCV Candlestick & Volume Chart
% ---------------------------------------------------------------------
fig1 = viz.createFigure('Figure 1: Binance Raw Data Candlestick Chart');
axCandle = axes(fig1);

dataCandle.Open   = TT_spot.Open;
dataCandle.High   = TT_spot.High;
dataCandle.Low    = TT_spot.Low;
dataCandle.Close  = TT_spot.Close;
dataCandle.Volume = TT_spot.Volume;
dataCandle.Time   = TT_spot.Time;
dataCandle.Title  = sprintf("%s %s Binance Candlestick & Volume Chart", SYMBOL, INTERVAL);

viz.plot(axCandle, 'Candlestick', dataCandle, 'ShowLegend', true);

% ---------------------------------------------------------------------
% Figure 2: Time Series EDA & Population Spectrum Dashboard
% ---------------------------------------------------------------------
% ---------------------------------------------------------------------
% Figure 2: Time Series EDA & Population Spectrum Dashboard
% ---------------------------------------------------------------------
fig2 = viz.createFigure('Figure 2: Time Series EDA & Population Spectrum');
layoutEDA = tiledlayout(fig2, 2, 2, 'TileSpacing', 'loose', 'Padding', 'compact');

% Subplot 2A: Time Series Realization (Log Returns)
axLogRet = nexttile(layoutEDA, [1, 2]);
viz.styleAxes(axLogRet);
hold(axLogRet, 'on');
plot(axLogRet, TT_train.Time, yTrain, 'Color', colors.Primary, 'LineWidth', 1.0, 'DisplayName', 'Train Log Return');
title(axLogRet, sprintf('%s %s Log Returns Realization', SYMBOL, INTERVAL), 'Color', colors.Text);
xlabel(axLogRet, 'Time (UTC)', 'Color', colors.Text);
ylabel(axLogRet, 'Log Return r(t)', 'Color', colors.Text);
legend(axLogRet, 'Location', 'northwest', 'TextColor', colors.Text, 'Color', 'none', 'EdgeColor', 'none');
hold(axLogRet, 'off');

% Subplot 2B: ACF & PACF via OceanViz Engine
axACF = nexttile(layoutEDA);
dataACF.ACF       = edaRet.ACF;
dataACF.Lags      = edaRet.ACFLags;
dataACF.ConfBound = edaRet.ConfBound;
dataACF.Title     = "Autocorrelation Function (ACF)";
dataACF.YLabel    = "ACF \rho_k";

viz.plot(axACF, 'ACF', dataACF, 'ShowLegend', true);

% Subplot 2C: Population Spectrum vs Empirical Welch Spectrum
axSpec = nexttile(layoutEDA);
viz.styleAxes(axSpec);
hold(axSpec, 'on');

% Normalize Welch frequencies to [0, pi]
normFreqs = edaRet.PSD_Freqs * 2 * pi;
plot(axSpec, normFreqs, 10*log10(edaRet.PSD_Power + 1e-12), 'Color', colors.Primary, 'LineWidth', 1.2, 'DisplayName', 'Empirical Welch PSD');
plot(axSpec, w, 10*log10(popSpectrum + 1e-12), '--', 'Color', colors.Boundary, 'LineWidth', 1.8, 'DisplayName', sprintf('Theoretical AR(%d) Spectrum', bestP));

title(axSpec, 'Population Spectrum vs Empirical Welch PSD', 'Color', colors.Text);
xlabel(axSpec, 'Frequency \omega (rad/sample)', 'Color', colors.Text);
ylabel(axSpec, 'Power (dB)', 'Color', colors.Text);
legend(axSpec, 'Location', 'northeast', 'TextColor', colors.Text, 'Color', 'none', 'EdgeColor', 'none');
hold(axSpec, 'off');

% ---------------------------------------------------------------------
% Figure 3: TSA Model Fitting, Lag Selection & Forecasting
% ---------------------------------------------------------------------
fig3 = viz.createFigure('Figure 3: TSA AR(p) Model Fitting & Multi-Step Forecasting');
layoutTSA = tiledlayout(fig3, 2, 2, 'TileSpacing', 'loose', 'Padding', 'compact');

% Subplot 3A: Multi-Step Forecasting on Test Set
axFore = nexttile(layoutTSA, [1, 2]);
viz.styleAxes(axFore);
hold(axFore, 'on');

tTest = TT_test.Time;
fill(axFore, [tTest; flipud(tTest)], [bands(:, 2); flipud(bands(:, 1))], ...
    colors.Secondary, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', '95% Confidence Band');

plot(axFore, tTest, yTest, 'Color', colors.Text * 0.8, 'LineWidth', 1.0, 'LineStyle', ':', 'DisplayName', 'Actual Test Returns');
plot(axFore, tTest, yHat, 'Color', colors.Secondary, 'LineWidth', 1.5, 'DisplayName', sprintf('AR(%d) Forecast', bestP));

title(axFore, sprintf('Multi-Step Recursive Test Forecast (RMSE = %.6f, Coverage = %.1f%%)', rmseTest, coverage), ...
    'Color', colors.Text, 'FontSize', 11, 'FontWeight', 'bold');
xlabel(axFore, 'Time (UTC)', 'Color', colors.Text);
ylabel(axFore, 'Log Return r(t)', 'Color', colors.Text);
legend(axFore, 'Location', 'northwest', 'TextColor', colors.Text, 'Color', 'none', 'EdgeColor', 'none');
hold(axFore, 'off');

% Subplot 3B: Information Criteria Lag Selection (AIC & BIC)
axIC = nexttile(layoutTSA);
viz.styleAxes(axIC);
hold(axIC, 'on');

plot(axIC, 1:maxOrder, aicVals, '-o', 'Color', colors.Primary, 'LineWidth', 1.2, 'DisplayName', 'AIC');
plot(axIC, 1:maxOrder, bicVals, '-x', 'Color', colors.Boundary, 'LineWidth', 1.2, 'DisplayName', 'BIC');
xline(axIC, bestP, '--', 'Color', colors.Secondary, 'LineWidth', 1.2, 'DisplayName', sprintf('Optimal p=%d', bestP));

title(axIC, 'Information Criteria Lag Order Selection', 'Color', colors.Text);
xlabel(axIC, 'AR Order p', 'Color', colors.Text);
ylabel(axIC, 'Criterion Value', 'Color', colors.Text);
legend(axIC, 'Location', 'northeast', 'TextColor', colors.Text, 'Color', 'none', 'EdgeColor', 'none');
hold(axIC, 'off');

% Subplot 3C: Residuals ACF via OceanViz Engine (Model Fit Validation)
axRes = nexttile(layoutTSA);

residuals = zeros(length(yTrain) - bestP, 1);
for t = (bestP + 1):length(yTrain)
    residuals(t - bestP) = yTrain(t) - arModel.Constant - ...
        sum(arModel.ARCoeffs .* flip(yTrain(t-bestP : t-1)'));
end

maxLagRes = 20;
acfRes = OceanPipeline.Transforms.BinanceTSAProcessor.computeACF(residuals, maxLagRes);

dataResACF.ACF       = acfRes';
dataResACF.Lags      = 0:maxLagRes;
dataResACF.ConfBound = 1.96 / sqrt(length(residuals));
dataResACF.Title     = "Residuals Autocorrelation (Model Diagnostic)";
dataResACF.YLabel    = "Residual ACF";

viz.plot(axRes, 'ACF', dataResACF, 'ShowLegend', true);

% ---------------------------------------------------------------------
% Figure 4: Features & Temporal Train/Val/Test Splits
% ---------------------------------------------------------------------
fig4 = viz.createFigure('Figure 4: Binance Features & Temporal Splits');
tlayoutSplits = tiledlayout(fig4, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

% Price & VWAP
axP = nexttile(tlayoutSplits);
viz.styleAxes(axP);
hold(axP, 'on');
plot(axP, TT_feat.Time, TT_feat.Close, 'Color', colors.Primary, 'LineWidth', 1.2, 'DisplayName', 'Close Price');
plot(axP, TT_feat.Time, TT_feat.VWAP,  'Color', colors.Boundary, 'LineWidth', 1.2, 'DisplayName', '24h VWAP');
xline(axP, TT_train.Time(end), '--', 'Color', [0.2 0.8 0.4], 'LineWidth', 1.5, 'DisplayName', 'Train/Val Boundary');
xline(axP, TT_val.Time(end),   '--', 'Color', [1.0 0.5 0.2], 'LineWidth', 1.5, 'DisplayName', 'Val/Test Boundary');
title(axP, sprintf('%s Price & VWAP with Temporal Splits', SYMBOL), 'Color', colors.Text);
ylabel(axP, 'Price (USDT)', 'Color', colors.Text);
legend(axP, 'Location', 'northwest', 'TextColor', colors.Text, 'Color', 'none', 'EdgeColor', 'none');
hold(axP, 'off');

% Rolling Volatility (Rolling Std of Log Returns)
axV = nexttile(tlayoutSplits);
viz.styleAxes(axV);
hold(axV, 'on');
plot(axV, TT_feat.Time, TT_feat.RollingVol, 'Color', colors.Secondary, 'LineWidth', 1.2, 'DisplayName', 'Rolling Vol (24h)');
xline(axV, TT_train.Time(end), '--', 'Color', [0.2 0.8 0.4], 'LineWidth', 1.2, 'HandleVisibility', 'off');
xline(axV, TT_val.Time(end),   '--', 'Color', [1.0 0.5 0.2], 'LineWidth', 1.2, 'HandleVisibility', 'off');
title(axV, '24-Period Rolling Volatility \sigma_t (Std of Log Returns)', 'Color', colors.Text);
ylabel(axV, 'Vol (\sigma_t)', 'Color', colors.Text);
legend(axV, 'Location', 'northwest', 'TextColor', colors.Text, 'Color', 'none', 'EdgeColor', 'none');
hold(axV, 'off');

% Volume Z-Score & Taker Ratio
axZ = nexttile(tlayoutSplits);
viz.styleAxes(axZ);
hold(axZ, 'on');
plot(axZ, TT_feat.Time, TT_feat.VolumeZScore, 'Color', colors.Primary, 'LineWidth', 1.0, 'DisplayName', 'Volume Z-Score');
if ismember('TakerBuyRatio', TT_feat.Properties.VariableNames)
    plot(axZ, TT_feat.Time, TT_feat.TakerBuyRatio, 'Color', colors.Boundary, 'LineWidth', 1.0, 'DisplayName', 'Taker Buy Ratio');
end
yline(axZ, 0, 'Color', colors.Grid, 'LineWidth', 0.8, 'HandleVisibility', 'off');
xline(axZ, TT_train.Time(end), '--', 'Color', [0.2 0.8 0.4], 'LineWidth', 1.2, 'HandleVisibility', 'off');
xline(axZ, TT_val.Time(end),   '--', 'Color', [1.0 0.5 0.2], 'LineWidth', 1.2, 'HandleVisibility', 'off');
title(axZ, 'Volume Indicators (Z-Score & Taker Ratio)', 'Color', colors.Text);
xlabel(axZ, 'Time (UTC)', 'Color', colors.Text);
ylabel(axZ, 'Indicator', 'Color', colors.Text);
legend(axZ, 'Location', 'northwest', 'TextColor', colors.Text, 'Color', 'none', 'EdgeColor', 'none');
hold(axZ, 'off');
% ---------------------------------------------------------------------
% Figure 5: Power Spectrum & Spectral Diagnostics Dashboard
% ---------------------------------------------------------------------
fig5 = viz.createFigure('Figure 5: Power Spectrum & Spectral Analysis Dashboard');
layoutSpec = tiledlayout(fig5, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Subplot 5A: Raw Periodogram vs Welch PSD vs Theoretical AR Spectrum (Linear Scale)
axSpecLin = nexttile(layoutSpec);
viz.styleAxes(axSpecLin);
hold(axSpecLin, 'on');

plot(axSpecLin, edaRet.Periodogram_Freqs, edaRet.Periodogram_Power, 'Color', [0.4 0.4 0.5], 'LineWidth', 0.8, 'DisplayName', 'Raw FFT Periodogram');
plot(axSpecLin, edaRet.PSD_AngularFreq, edaRet.PSD_Power, 'Color', colors.Primary, 'LineWidth', 1.5, 'DisplayName', 'Welch PSD');
plot(axSpecLin, w, popSpectrum, '--', 'Color', colors.Boundary, 'LineWidth', 1.8, 'DisplayName', sprintf('AR(%d) Spectrum', bestP));

title(axSpecLin, 'Linear Power Spectral Density P(\omega)', 'Color', colors.Text);
xlabel(axSpecLin, 'Normalized Frequency \omega (rad/sample)', 'Color', colors.Text);
ylabel(axSpecLin, 'Power Density P(\omega)', 'Color', colors.Text);
legend(axSpecLin, 'Location', 'northeast', 'TextColor', colors.Text, 'Color', 'none', 'EdgeColor', 'none');
hold(axSpecLin, 'off');

% Subplot 5B: Log Power Spectral Density (dB/rad/sample)
axSpecDb = nexttile(layoutSpec);
viz.styleAxes(axSpecDb);
hold(axSpecDb, 'on');

plot(axSpecDb, edaRet.PSD_AngularFreq, 10*log10(edaRet.PSD_Power + 1e-12), 'Color', colors.Primary, 'LineWidth', 1.5, 'DisplayName', 'Welch PSD (dB)');
plot(axSpecDb, w, 10*log10(popSpectrum + 1e-12), '--', 'Color', colors.Boundary, 'LineWidth', 1.8, 'DisplayName', sprintf('AR(%d) Spectrum (dB)', bestP));

title(axSpecDb, 'Log Power Spectral Density (dB)', 'Color', colors.Text);
xlabel(axSpecDb, 'Normalized Frequency \omega (rad/sample)', 'Color', colors.Text);
ylabel(axSpecDb, 'Power (dB)', 'Color', colors.Text);
legend(axSpecDb, 'Location', 'northeast', 'TextColor', colors.Text, 'Color', 'none', 'EdgeColor', 'none');
hold(axSpecDb, 'off');

% Subplot 5C: Power Spectrum vs Period (Hours) & Dominant Cycle Identification
axPeriod = nexttile(layoutSpec);
viz.styleAxes(axPeriod);
hold(axPeriod, 'on');

validP = edaRet.PSD_PeriodHours > 0 & edaRet.PSD_PeriodHours <= 72;
plot(axPeriod, edaRet.PSD_PeriodHours(validP), edaRet.PSD_Power(validP), 'Color', colors.Secondary, 'LineWidth', 1.5, 'DisplayName', 'Power vs Period');

% Highlight standard trading cycle markers: 24h, 12h, 8h, 4h
xline(axPeriod, 24, ':', 'Color', [0.3 0.8 1.0], 'LineWidth', 1.2, 'DisplayName', '24h Daily Cycle');
xline(axPeriod, 8,  ':', 'Color', [1.0 0.8 0.3], 'LineWidth', 1.2, 'DisplayName', '8h Funding Cycle');

title(axPeriod, sprintf('Power Spectrum vs Period T (Dominant = %.1fh)', edaRet.DominantPeriodHours), 'Color', colors.Text);
xlabel(axPeriod, 'Period T (Hours)', 'Color', colors.Text);
ylabel(axPeriod, 'Power Density P(T)', 'Color', colors.Text);
legend(axPeriod, 'Location', 'northeast', 'TextColor', colors.Text, 'Color', 'none', 'EdgeColor', 'none');
hold(axPeriod, 'off');

% Subplot 5D: Cumulative Spectral Power & Parseval Theorem Verification
axCum = nexttile(layoutSpec);
viz.styleAxes(axCum);
hold(axCum, 'on');

df_w = edaRet.Periodogram_Freqs(2) - edaRet.Periodogram_Freqs(1);
cumPower = cumsum(edaRet.Periodogram_Power) * df_w;
plot(axCum, edaRet.Periodogram_Freqs, cumPower, 'Color', colors.Primary, 'LineWidth', 1.5, 'DisplayName', 'Integrated Spectral CDF');
yline(axCum, edaRet.Parseval_Variance, '--', 'Color', colors.Secondary, 'LineWidth', 1.5, 'DisplayName', sprintf('Sample Variance \\sigma^2 = %.6e', edaRet.Parseval_Variance));

title(axCum, sprintf('Cumulative Spectral Power (Parseval Area = %.6e)', edaRet.Parseval_TotalArea), 'Color', colors.Text);
xlabel(axCum, 'Normalized Frequency \omega (rad/sample)', 'Color', colors.Text);
ylabel(axCum, 'Cumulative Variance', 'Color', colors.Text);
legend(axCum, 'Location', 'southeast', 'TextColor', colors.Text, 'Color', 'none', 'EdgeColor', 'none');
hold(axCum, 'off');

fprintf("\n=====================================================================\n");
fprintf("   BINANCE TSA & SPECTRAL MODELING PIPELINE COMPLETE   \n");
fprintf("=====================================================================\n");

%% --- Helper Function for Offline Dataset Generation ---
function TT = generateOfflineReferenceData(N, timeStep)
    tEnd = datetime('now', 'TimeZone', 'UTC');
    tStart = tEnd - N * timeStep;
    dt = (tStart:timeStep:tEnd)';
    dt = dt(1:N);

    S0 = 65000;
    mu = 0.0001;
    sigma = 0.015;

    r = mu + sigma * randn(N, 1);
    prices = S0 * exp(cumsum(r));

    highP = prices .* (1 + 0.005 * rand(N, 1));
    lowP  = prices .* (1 - 0.005 * rand(N, 1));
    openP = [S0; prices(1:end-1)];
    vol   = 100 + 50 * randn(N, 1); vol(vol < 10) = 10;
    qVol  = vol .* prices;
    nTrades = round(500 + 200 * randn(N, 1));
    takerBase = vol .* (0.45 + 0.1 * rand(N, 1));
    takerQuote = takerBase .* prices;

    TT = timetable(dt, openP, highP, lowP, prices, vol, qVol, nTrades, takerBase, takerQuote, ...
        'VariableNames', {'Open', 'High', 'Low', 'Close', 'Volume', ...
                          'QuoteAssetVolume', 'NumberOfTrades', ...
                          'TakerBuyBaseVolume', 'TakerBuyQuoteVolume'});
    TT.Properties.DimensionNames{1} = 'Time';
end
