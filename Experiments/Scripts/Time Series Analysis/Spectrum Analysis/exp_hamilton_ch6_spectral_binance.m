% EXP_HAMILTON_CH6_SPECTRAL_BINANCE
% Replication of Hamilton Chapter 6 Spectral Analysis (Figures 6.3 - 6.6)
% using real Binance BTC/USDT Daily Price & Return Data.
%
% Theoretical & Empirical Objectives:
%   1. Ingest real Binance BTC/USDT daily close price data (1000 daily klines).
%   2. Replicate Hamilton Figure 6.3/6.4: Non-stationary level series y_t = ln(Price)
%      and its sample periodogram I_y(omega), showing massive zero-frequency peak.
%   3. Replicate Hamilton Figure 6.5: First-difference log returns x_t = 100 * [ln(y_t) - ln(y_{t-1})]
%      and its periodogram I_x(omega), illustrating high-pass filtering (h_x(omega) = 4 sin^2(omega/2)).
%   4. Replicate Hamilton Figure 6.6: 30-day growth rate w_t = 100 * [ln(y_t) - ln(y_{t-30})]
%      and its periodogram I_w(omega), revealing filter ripples & zero-nulls (h_w(omega) = 4 sin^2(30 omega / 2)).
%   5. Fit AR(p) model via Yule-Walker and compare sample periodogram, Welch PSD,
%      and theoretical population AR spectrum.
%   6. Verify Parseval's Spectral Energy Theorem (Integral of Periodogram = Sample Variance).
%   7. Export processed series to Medallion DataLake (Parquet) for Python Phase.
%   8. Render figures exclusively using OceanViz.Core engine with Dark theme styling.
%
% Author: Ocean Library   Date: 2026-09-29

clc; clear; close all;
rng(42, 'twister');

fprintf("=====================================================================\n");
fprintf("   HAMILTON CH.6 SPECTRAL ANALYSIS (BINANCE BTC DAILY REAL DATA)    \n");
fprintf("=====================================================================\n\n");

%% === 1. Data Ingestion via Binance REST API ===
SYMBOL   = "BTCUSDT";
INTERVAL = "1d";
LIMIT    = 1000; % ~1000 days of daily close prices

fprintf("[Step 1] Ingesting %d daily klines from Binance REST API (%s)...\n", LIMIT, SYMBOL);

reader = OceanPipeline.Ingestion.BinanceDataReader(...
    'Symbol', SYMBOL, 'MarketType', 'spot', 'Interval', INTERVAL, 'Limit', LIMIT);

try
    TT_raw = reader.fetchKlinesTimetable();
    fprintf("   [SUCCESS] Live Binance Daily Data fetched: %d rows (%s to %s UTC)\n", ...
        height(TT_raw), string(TT_raw.Time(1)), string(TT_raw.Time(end)));
catch ME
    fprintf("   [NOTICE] Live REST call failed (%s). Generating offline reference dataset.\n", ME.message);
    TT_raw = generateOfflineDailyReferenceData(LIMIT);
end

%% === 2. Transformations & Differencing (Hamilton Ch. 6 Formulations) ===
fprintf("\n[Step 2] Engineering Series & Filter Transformations...\n");

% Extract prices & timestamps
timeVector = TT_raw.Time;
prices = TT_raw.Close;
N = length(prices);

% 2a. Raw Log Level: y_t = ln(Price_t) (Hamilton Fig 6.3 Level Analog)
y_level = log(prices);

% 2b. First-Difference Log Return: x_t = 100 * (y_t - y_{t-1}) (Hamilton Fig 6.5 Analog)
x_ret = [NaN; 100 * diff(y_level)];

% 2c. 30-Day Growth Rate: w_t = 100 * (y_t - y_{t-30}) (Hamilton Fig 6.6 Analog)
K_LAG = 30;
w_growth = [nan(K_LAG, 1); 100 * (y_level(K_LAG+1:end) - y_level(1:end-K_LAG))];

% Create structured timetable
TT_spectral = timetable(timeVector, prices, y_level, x_ret, w_growth, ...
    'VariableNames', {'Close', 'LogLevel', 'LogReturn', 'Growth30D'});

% Extract clean valid vectors (drop NaNs)
x_clean = x_ret(~isnan(x_ret));
w_clean = w_growth(~isnan(w_growth));
y_clean = y_level;

fprintf("   - Log Level y_t:       %d samples\n", length(y_clean));
fprintf("   - 1-Day Return x_t:    %d samples (Mean = %.4f%%, Std = %.4f%%)\n", ...
    length(x_clean), mean(x_clean), std(x_clean));
fprintf("   - 30-Day Growth w_t:   %d samples (Mean = %.4f%%, Std = %.4f%%)\n", ...
    length(w_clean), mean(w_clean), std(w_clean));

%% === 3. Computation of Sample Periodograms (Wiener-Khinchin FFT) ===
fprintf("\n[Step 3] Computing Sample Periodograms I(omega)...\n");

[w_grid, Iy] = computePeriodogram(y_clean - mean(y_clean));
[~,      Ix] = computePeriodogram(x_clean - mean(x_clean));
[~,      Iw] = computePeriodogram(w_clean - mean(w_clean));

% 3b. Theoretical Filter Transfer Functions (Hamilton 6.2.2 & 6.2.3)
% Differencing Filter h_x(omega) = |1 - exp(-1i * omega)|^2 = 4 * sin^2(omega / 2)
h_diff = 4 * (sin(w_grid / 2)).^2;

% 30-Day Difference Filter h_w(omega) = |1 - exp(-1i * 30 * omega)|^2 = 4 * sin^2(30 * omega / 2)
h_30d  = 4 * (sin(K_LAG * w_grid / 2)).^2;

%% === 4. AR Population Spectrum Fitting ===
fprintf("\n[Step 4] Fitting AR(p) Model & Theoretical Population Spectrum...\n");

arModel = OceanMath.TimeSeries.Models.AR.fit(x_clean, 'Order', 4);
arOrder = length(arModel.ARCoeffs);
fprintf("   - Fitted AR(%d) Model for Log Returns x_t:\n", arOrder);
fprintf("     * Equation: %s\n", arModel.getEquation());
fprintf("     * Innovation Variance sigma_e^2: %.6f\n", arModel.NoiseStd^2);

aPoly = [1, -arModel.ARCoeffs];
popSpectrum_AR = (arModel.NoiseStd^2 / (2 * pi)) ./ (abs(polyval(aPoly, exp(-1i * w_grid))).^2);

% Compute Empirical Welch Spectrum for comparison
[pxx_welch, f_welch] = pwelch(x_clean - mean(x_clean), round(length(x_clean)/4), [], length(w_grid)*2, 1);
w_welch = f_welch * 2 * pi;
welchPSD = pxx_welch / (2 * pi);

%% === 5. Verification of Parseval's Spectral Identity ===
fprintf("\n[Step 5] Verifying Parseval's Energy Conservation Identity...\n");

% Numerical integration of Periodogram over [0, pi]
dw = w_grid(2) - w_grid(1);
area_Ix = sum(Ix) * dw;
var_x   = var(x_clean, 1); % Population variance

fprintf("   - Integrated Periodogram Area (0 to pi): %.6f\n", area_Ix);
fprintf("   - Sample Variance Var(x_t):              %.6f\n", var_x);
fprintf("   - Parseval Identity Absolute Error:      %.4e\n", abs(area_Ix - var_x));

%% === 6. Medallion DataLake Export (Parquet) ===
fprintf("\n[Step 6] Exporting Processed Dataset to DataLake...\n");
lakeDir = "./lake/processed";
if ~exist(lakeDir, 'dir')
    mkdir(lakeDir);
end
parquetPath = fullfile(lakeDir, "binance_btc_daily_hamilton.parquet");
parquetwrite(parquetPath, timetable2table(TT_spectral));
fprintf("   - Saved: %s\n", parquetPath);

%% === 7. OceanViz Visualization Engine Rendering ===
fprintf("\n[Step 7] Rendering Hamilton Ch.6 Figures via OceanViz Engine...\n");
viz = OceanViz.Core('Theme', 'Dark');
colors = viz.Colors;

% ---------------------------------------------------------------------
% Figure 1: Raw Binance Daily OHLCV Candlestick & Volume Chart
% ---------------------------------------------------------------------
fig1 = viz.createFigure('Figure 1: Binance BTC/USDT Daily Candlestick Chart');
axCandle = axes(fig1);

OceanViz.Charts.CandlestickChart.plot(axCandle, TT_raw, ...
    'Title', sprintf("%s %s Binance Daily OHLCV Candlestick & Volume Chart", SYMBOL, INTERVAL), ...
    'Theme', 'Dark', ...
    'Colors', colors);

% ---------------------------------------------------------------------
% Figure 2: Hamilton Fig 6.3 & 6.4 — Level Series y_t & Periodogram I_y(w)
% ---------------------------------------------------------------------
fig2 = viz.createFigure('Figure 2: Hamilton Fig 6.3/6.4 - BTC Log Level & Periodogram');
layout2 = tiledlayout(fig2, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

ax2A = nexttile(layout2);
viz.styleAxes(ax2A);
hold(ax2A, 'on');
plot(ax2A, timeVector, y_level, 'Color', colors.Primary, 'LineWidth', 1.2, 'DisplayName', 'Log Level y_t = ln(Price)');
title(ax2A, 'Hamilton Fig 6.3 Analog: BTC/USDT Daily Log Level Series y_t', 'Color', colors.Text);
xlabel(ax2A, 'Date', 'Color', colors.Text);
ylabel(ax2A, 'ln(Price)', 'Color', colors.Text);
legend(ax2A, 'Location', 'northwest', 'TextColor', colors.Text, 'Color', 'none', 'EdgeColor', 'none');
hold(ax2A, 'off');

ax2B = nexttile(layout2);
viz.styleAxes(ax2B);
hold(ax2B, 'on');
plot(ax2B, w_grid, 10*log10(Iy + 1e-12), 'Color', colors.Secondary, 'LineWidth', 1.2, 'DisplayName', 'Sample Periodogram I_y(\omega)');
title(ax2B, 'Hamilton Fig 6.4 Analog: Periodogram of Non-Stationary Level y_t (Zero-Frequency Dominance)', 'Color', colors.Text);
xlabel(ax2B, 'Frequency \omega (rad/sample)', 'Color', colors.Text);
ylabel(ax2B, 'Power (dB)', 'Color', colors.Text);
legend(ax2B, 'Location', 'northeast', 'TextColor', colors.Text, 'Color', 'none', 'EdgeColor', 'none');
hold(ax2B, 'off');

% ---------------------------------------------------------------------
% Figure 3: Hamilton Fig 6.5 — Log Return x_t & High-Pass Differencing Spectrum
% ---------------------------------------------------------------------
fig3 = viz.createFigure('Figure 3: Hamilton Fig 6.5 - 1-Day Log Return & Spectrum');
layout3 = tiledlayout(fig3, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

tCleanX = timeVector(~isnan(x_ret));
ax3A = nexttile(layout3);
viz.styleAxes(ax3A);
hold(ax3A, 'on');
plot(ax3A, tCleanX, x_clean, 'Color', colors.Primary, 'LineWidth', 1.0, 'DisplayName', '1-Day Return x_t (%)');
yline(ax3A, 0, ':', 'Color', colors.Grid, 'LineWidth', 0.8, 'HandleVisibility', 'off');
title(ax3A, 'Hamilton Fig 6.5 Top: 1-Day Log Return Series x_t = 100 * (y_t - y_{t-1})', 'Color', colors.Text);
xlabel(ax3A, 'Date', 'Color', colors.Text);
ylabel(ax3A, 'Return x_t (%)', 'Color', colors.Text);
legend(ax3A, 'Location', 'northwest', 'TextColor', colors.Text, 'Color', 'none', 'EdgeColor', 'none');
hold(ax3A, 'off');

ax3B = nexttile(layout3);
viz.styleAxes(ax3B);
hold(ax3B, 'on');
plot(ax3B, w_grid, 10*log10(Ix + 1e-12), 'Color', colors.Primary, 'LineWidth', 1.0, 'DisplayName', 'Sample Periodogram I_x(\omega)');
plot(ax3B, w_grid, 10*log10(popSpectrum_AR + 1e-12), '--', 'Color', colors.Boundary, 'LineWidth', 1.8, 'DisplayName', sprintf('Theoretical AR(%d) Spectrum', arOrder));
plot(ax3B, w_welch, 10*log10(welchPSD + 1e-12), ':', 'Color', colors.Secondary, 'LineWidth', 1.5, 'DisplayName', 'Welch Smoothed PSD');
title(ax3B, 'Hamilton Fig 6.5 Bottom: Periodogram & Theoretical Spectrum of Differenced Series x_t', 'Color', colors.Text);
xlabel(ax3B, 'Frequency \omega (rad/sample)', 'Color', colors.Text);
ylabel(ax3B, 'Power (dB)', 'Color', colors.Text);
legend(ax3B, 'Location', 'northeast', 'TextColor', colors.Text, 'Color', 'none', 'EdgeColor', 'none');
hold(ax3B, 'off');

% ---------------------------------------------------------------------
% Figure 4: Hamilton Fig 6.6 — 30-Day Growth w_t & Filter Ripples Spectrum
% ---------------------------------------------------------------------
fig4 = viz.createFigure('Figure 4: Hamilton Fig 6.6 - 30-Day Growth Rate & Filter Ripples');
layout4 = tiledlayout(fig4, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

tCleanW = timeVector(~isnan(w_growth));
ax4A = nexttile(layout4);
viz.styleAxes(ax4A);
hold(ax4A, 'on');
plot(ax4A, tCleanW, w_clean, 'Color', colors.Secondary, 'LineWidth', 1.2, 'DisplayName', '30-Day Growth Rate w_t (%)');
yline(ax4A, 0, ':', 'Color', colors.Grid, 'LineWidth', 0.8, 'HandleVisibility', 'off');
title(ax4A, 'Hamilton Fig 6.6 Top: 30-Day Growth Rate Series w_t = 100 * (y_t - y_{t-30})', 'Color', colors.Text);
xlabel(ax4A, 'Date', 'Color', colors.Text);
ylabel(ax4A, 'Growth w_t (%)', 'Color', colors.Text);
legend(ax4A, 'Location', 'northwest', 'TextColor', colors.Text, 'Color', 'none', 'EdgeColor', 'none');
hold(ax4A, 'off');

ax4B = nexttile(layout4);
viz.styleAxes(ax4B);
hold(ax4B, 'on');
plot(ax4B, w_grid, 10*log10(Iw + 1e-12), 'Color', colors.Secondary, 'LineWidth', 1.2, 'DisplayName', 'Sample Periodogram I_w(\omega)');

% Mark theoretical spectral zero nulls at omega_j = 2 * pi * j / 30
nullFreqs = (2 * pi / K_LAG) * (1:floor(K_LAG / 2));
for k = 1:length(nullFreqs)
    if k == 1
        xline(ax4B, nullFreqs(k), ':', 'Color', colors.Boundary, 'LineWidth', 1.0, 'DisplayName', 'Filter Zero Nulls (\omega = 2\pi j / 30)');
    else
        xline(ax4B, nullFreqs(k), ':', 'Color', colors.Boundary, 'LineWidth', 1.0, 'HandleVisibility', 'off');
    end
end
title(ax4B, 'Hamilton Fig 6.6 Bottom: Periodogram of 30-Day Difference (Spurious Filter Nulls & Ripples)', 'Color', colors.Text);
xlabel(ax4B, 'Frequency \omega (rad/sample)', 'Color', colors.Text);
ylabel(ax4B, 'Power (dB)', 'Color', colors.Text);
legend(ax4B, 'Location', 'northeast', 'TextColor', colors.Text, 'Color', 'none', 'EdgeColor', 'none');
hold(ax4B, 'off');

% ---------------------------------------------------------------------
% Figure 5: Theoretical Filter Gain Functions Comparison
% ---------------------------------------------------------------------
fig5 = viz.createFigure('Figure 5: Spectral Filter Gain Transfer Functions');
ax5 = axes(fig5);
viz.styleAxes(ax5);
hold(ax5, 'on');

plot(ax5, w_grid, h_diff, 'Color', colors.Primary, 'LineWidth', 1.8, 'DisplayName', '1-Day Differencing Gain h_x(\omega) = 4 sin^2(\omega / 2)');
plot(ax5, w_grid, h_30d,  'Color', colors.Secondary, 'LineWidth', 1.2, 'DisplayName', '30-Day Differencing Gain h_w(\omega) = 4 sin^2(30\omega / 2)');

title(ax5, 'Theoretical Frequency Response Gain Comparison h(\omega)', 'Color', colors.Text);
xlabel(ax5, 'Frequency \omega (rad/sample)', 'Color', colors.Text);
ylabel(ax5, 'Power Gain |H(\omega)|^2', 'Color', colors.Text);
legend(ax5, 'Location', 'northeast', 'TextColor', colors.Text, 'Color', 'none', 'EdgeColor', 'none');
hold(ax5, 'off');

fprintf("\n=====================================================================\n");
fprintf("   HAMILTON CH.6 SPECTRAL EXPERIMENT COMPLETE   \n");
fprintf("=====================================================================\n");

%% --- Helper Functions ---
function [w_grid, I_sample] = computePeriodogram(z)
    % Compute sample periodogram I(omega) via FFT
    % I(omega) = (1 / (2 * pi * N)) * |sum_{t=1}^N z_t * exp(-1i * omega * (t-1))|^2
    N = length(z);
    N_fft = 2^nextpow2(2 * N);
    Z_fft = fft(z, N_fft);
    
    % One-sided frequency grid [0, pi]
    numPoints = floor(N_fft / 2) + 1;
    dw = 2 * pi / N_fft;
    w_grid = (0:numPoints-1)' * dw;
    
    Z_half = Z_fft(1:numPoints);
    I_sample = (1 / (2 * pi * N)) * (abs(Z_half).^2);
    
    % Account for negative frequencies (factor of 2 except DC and Nyquist)
    if numPoints > 1
        if mod(N_fft, 2) == 0
            I_sample(2:end-1) = 2 * I_sample(2:end-1); % Even N_fft: Nyquist at end is unique
        else
            I_sample(2:end)   = 2 * I_sample(2:end);   % Odd N_fft: No exact Nyquist bin, double all positive freqs
        end
    end
end

function TT = generateOfflineDailyReferenceData(N)
    tEnd = datetime('now', 'TimeZone', 'UTC');
    tStart = tEnd - caldays(N);
    dt = (tStart:caldays(1):tEnd)';
    dt = dt(1:N);

    S0 = 30000;
    mu = 0.001;
    sigma = 0.025;

    r = mu + sigma * randn(N, 1);
    prices = S0 * exp(cumsum(r));

    highP = prices .* (1 + 0.01 * rand(N, 1));
    lowP  = prices .* (1 - 0.01 * rand(N, 1));
    openP = [S0; prices(1:end-1)];
    vol   = 5000 + 2000 * randn(N, 1); vol(vol < 100) = 100;
    qVol  = vol .* prices;
    nTrades = round(20000 + 5000 * randn(N, 1));
    takerBase = vol .* (0.50 + 0.05 * rand(N, 1));
    takerQuote = takerBase .* prices;

    TT = timetable(dt, openP, highP, lowP, prices, vol, qVol, nTrades, takerBase, takerQuote, ...
        'VariableNames', {'Open', 'High', 'Low', 'Close', 'Volume', ...
                          'QuoteAssetVolume', 'NumberOfTrades', ...
                          'TakerBuyBaseVolume', 'TakerBuyQuoteVolume'});
    TT.Properties.DimensionNames{1} = 'Time';
end
