% EXP_POPULATION_SPECTRUM  Calculates the Population Spectrum of Daily Sunspots
%
% This experiment follows the Matlab_Python_TimeSeries_wl_exp workflow.
% It demonstrates robust time series ingestion, spectral analysis (pwelch/periodogram),
% baseline ARIMA modeling, and Parquet export for Python MLOps.
%
% Data Source: SILSO Daily Sunspot Number (Royal Observatory of Belgium)

clc; clear; close all;
rng(42, 'twister');

% Add project root to path to resolve +OceanViz
rootPath = fileparts(fileparts(fileparts(fileparts(fileparts(mfilename('fullpath'))))));
addpath(rootPath);

fprintf('=== Phase 1: MATLAB Time Series Experiment ===\n');
%% Step 1.1: Data Ingestion & Native Structures
fprintf('1.1 Fetching and formatting data...\n');
url = 'https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv';
options = weboptions('Timeout', 30);
filename = 'sunspot_daily.csv';
websave(filename, url, options);

% The SILSO daily CSV format: Year, Month, Day, DecimalYear, SunspotNumber, StdDev, Observations, Definitive
opts = detectImportOptions(filename);
% Force standard names
opts.VariableNames = {'Year', 'Month', 'Day', 'DecimalYear', 'SunspotNumber', 'StdDev', 'Observations', 'Definitive'};
data = readtable(filename, opts);

% Create proper datetime and timetable
timestamps = datetime(data.Year, data.Month, data.Day);
TT = timetable(timestamps, data.SunspotNumber, 'VariableNames', {'SunspotNumber'});

% Clean data (missing/error values are marked as -1 in this dataset)
TT.SunspotNumber(TT.SunspotNumber < 0) = NaN;
TT = fillmissing(TT, 'previous'); % Causal imputation

% Filter data from 1950 onwards for relevant modern cycles
TT = TT(timerange('1950-01-01', '2023-12-31'), :);

%% Step 1.2: Exploratory Data Analysis & Feature Extraction
fprintf('1.2 Performing Exploratory Data Analysis...\n');
try
    [h, pValue] = adftest(TT.SunspotNumber);
    fprintf('ADF Test: h=%d, p-value=%.4f (h=1 indicates stationary)\n', h, pValue);
catch
    fprintf('[Info] adftest requires Econometrics Toolbox. Skipping stationarity test.\n');
end

% Use OceanViz for plotting
try
    viz = OceanViz.Core('Theme', 'Dark');
    fig1 = viz.createFigure('Sunspot Time Series & ACF');
    tiledlayout(2, 1);
    
    ax1 = nexttile;
    plot(ax1, TT.timestamps, TT.SunspotNumber, 'Color', viz.Colors.Primary);
    title(ax1, 'Daily Sunspot Number (1950-2023)');
    viz.styleAxes(ax1);
    
    ax2 = nexttile;
    if exist('autocorr', 'file')
        autocorr(TT.SunspotNumber, 'NumLags', 365*20); % 20 years lag
    else
        [c, lags] = xcorr(TT.SunspotNumber - mean(TT.SunspotNumber), 365*20, 'coeff');
        stem(ax2, lags(lags>=0), c(lags>=0), 'Marker', 'none');
        xlabel(ax2, 'Lag'); ylabel(ax2, 'Sample Autocorrelation');
    end
    title(ax2, 'Autocorrelation (20 Years)');
    viz.styleAxes(ax2);
catch ME
    fprintf('[OceanViz Error] %s\n', ME.message);
    fprintf('[OceanViz] Missing OceanViz, using fallback plot.\n');
    figure;
    subplot(2,1,1); plot(TT.timestamps, TT.SunspotNumber); title('Daily Sunspots');
    subplot(2,1,2); 
    if exist('autocorr', 'file')
        autocorr(TT.SunspotNumber, 'NumLags', 365*20);
    else
        [c, lags] = xcorr(TT.SunspotNumber - mean(TT.SunspotNumber), 365*20, 'coeff');
        stem(lags(lags>=0), c(lags>=0), 'Marker', 'none');
        xlabel('Lag'); ylabel('Sample Autocorrelation');
    end
end

%% Step 1.3: Classical Modeling (Baseline ARIMA)
fprintf('1.3 Fitting baseline classical model (ARIMA)...\n');
% Train/Test split: 80% Train, 20% Test
splitIdx = floor(0.8 * height(TT));
TT_train = TT(1:splitIdx, :);
TT_test  = TT(splitIdx+1:end, :);

% Subsample to monthly for faster baseline ARIMA estimation
TT_train_monthly = retime(TT_train, 'monthly', 'mean');

try
    % Fit a basic ARMA(2,2) to model the cycle dynamics
    mdl = arima(2, 0, 2);
    EstMdl = estimate(mdl, TT_train_monthly.SunspotNumber, 'Display', 'off');
    fprintf('Estimated ARMA model variance: %.4f\n', EstMdl.Variance);
    
    % Subsample test to monthly for comparison
    TT_test_monthly = retime(TT_test, 'monthly', 'mean');
    n_fcast = height(TT_test_monthly); % Forecast for full test window
    
    % Forecast n_fcast steps
    [Y_fcast, ~] = forecast(EstMdl, n_fcast, 'Y0', TT_train_monthly.SunspotNumber);
    fcast_dates = TT_train_monthly.timestamps(end) + calmonths(1:n_fcast)';
    
    try
        fig3 = viz.createFigure('Forecast');
        ax4 = gca;
        plot(ax4, TT_test.timestamps, TT_test.SunspotNumber, 'Color', viz.Colors.Primary, 'DisplayName', 'Actual (Test)');
        hold(ax4, 'on');
        plot(ax4, fcast_dates, Y_fcast, 'Color', viz.Colors.Secondary, 'LineWidth', 1.5, 'DisplayName', 'Predicted (ARMA)');
        xlim(ax4, [TT_test.timestamps(1), fcast_dates(end)]);
        title(ax4, sprintf('Sunspot Number Forecast (%d months)', n_fcast));
        xlabel(ax4, 'Date');
        ylabel(ax4, 'Sunspot Number');
        legend(ax4, 'Location', 'best');
        viz.styleAxes(ax4);
        exportgraphics(fig3, 'lake/processed/matlab_forecast.png', 'Resolution', 150);
    catch
        fig3 = figure;
        plot(TT_test.timestamps, TT_test.SunspotNumber, 'DisplayName', 'Actual (Test)');
        hold on;
        plot(fcast_dates, Y_fcast, 'LineWidth', 1.5, 'DisplayName', 'Predicted (ARMA)');
        xlim([TT_test.timestamps(1), fcast_dates(end)]);
        title(sprintf('Sunspot Number Forecast (%d months)', n_fcast));
        xlabel('Date');
        ylabel('Sunspot Number');
        legend('Location', 'best');
        exportgraphics(fig3, 'lake/processed/matlab_forecast.png', 'Resolution', 150);
    end
catch
    fprintf('[Info] arima requires Econometrics Toolbox. Skipping baseline modeling.\n');
end

%% Step 1.4: Population Spectrum — Textbook Autocovariance Method (Wiener-Khinchin)
fprintf('1.4 Calculating Population Spectrum (Power Spectral Density)...\n');
% Implements the textbook sample analogue of the population spectrum:
%
%   S(w) = (1/2*pi) * [ gamma(0) + 2 * sum_{k=1}^{L} gamma(k)*cos(k*w) ]
%
% where gamma(k) = (1/N) * sum_{t=1}^{N-k} x(t)*x(t+k)  [biased autocovariance]
% omega in [0, pi] rad/sample;  fs = 1 sample/day.

fs = 1; % 1 sample per day

% Remove mean to focus on oscillatory variance (exclude DC component)
sunspots_centered = TT_train.SunspotNumber - mean(TT_train.SunspotNumber);

% --- VERIFICATION: Area under spectrum equals total variance (Parseval) ---
% Using periodogram with rectwin for an exact equality check.
% Windowing in pwelch alters normalization; rectwin keeps it exact.
[pxx_exact, f_exact] = periodogram(sunspots_centered, ...
    rectwin(length(sunspots_centered)), length(sunspots_centered), fs);
area_psd       = bandpower(pxx_exact, f_exact, 'psd');
total_variance = var(sunspots_centered, 1); % Population variance (1/N)

fprintf('Verification:\n');
fprintf('  Total Area under Spectrum: %.4f\n', area_psd);
fprintf('  Total Variance:            %.4f\n', total_variance);
fprintf('  Difference:                %.4e\n', abs(area_psd - total_variance));
% -------------------------------------------------------------------------

% --- Biased Sample Autocovariance via FFT (Wiener-Khinchin shortcut) ---
% The direct for-loop is O(N^2) — ~19000^2 operations, impractical.
% FFT-based computation is O(N log N) and gives the IDENTICAL biased result:
%   gamma(k) = IFFT{ |FFT(x)|^2 } / N
%
% Truncate to max_lag: 3 full solar cycles (~33 yrs) is sufficient to
% resolve the 11-year peak without the full N-1 lag sum.
x        = sunspots_centered(:);
N        = length(x);
max_lag  = min(N-1, floor(3 * 11 * 365.25)); % ~12,045 lags

X_fft    = fft(x, 2^nextpow2(2*N));          % zero-pad to next pow2 >= 2N
gammaHat = real(ifft(X_fft .* conj(X_fft))) / N; % full circular autocorrelation / N
gammaHat = gammaHat(1:max_lag+1);            % lags 0..max_lag  [max_lag+1 x 1]

% --- Discrete Angular Frequency Axis & Vectorized Spectrum ---
% omega in [0, pi] rad/sample; 2000 evaluation points
omega = linspace(0, pi, 2000);  % [1 x 2000]

% S(w) = (1/2*pi) * [ gamma(0) + 2 * gamma(1:L)' * cos([1:L]' * omega) ]
% The [max_lag x 2000] cosine matrix is built implicitly by outer product.
k_vec = (1:max_lag)';                                    % [max_lag x 1]
S = gammaHat(1) + 2 * (gammaHat(2:end)' * cos(k_vec * omega)); % [1 x 2000]
S = S / (2*pi);                                          % units: variance * day

% --- Convert omega to Period axis ---
% f (cycles/day)  = omega / (2*pi)
% period (days)   = 2*pi / omega
% period (years)  = 2*pi / (omega * 365.25)
period_years = (2*pi ./ omega) / 365.25;  % inf at omega=0 (DC)

% Restrict to 0.5..40 years for plotting (exclude DC spike)
valid = omega > 0 & period_years >= 0.5 & period_years <= 40;

try
    fig2 = viz.createFigure('Population Spectrum');
    ax3 = gca;
    plot(ax3, period_years(valid), 10*log10(S(valid)), ...
        'LineWidth', 1.5, 'Color', viz.Colors.Secondary, ...
        'DisplayName', 'Sample Spectrum S(\omega)');
    xlabel(ax3, 'Period (Years)');
    ylabel(ax3, 'Power Spectral Density (dB \cdot day)');
    title(ax3, 'Population Spectrum — Autocovariance Method (Wiener-Khinchin)');
    legend(ax3, 'Location', 'best');
    viz.styleAxes(ax3);

    % Find dominant peak in the 5-20 year solar cycle band
    solar_band  = valid & period_years >= 5 & period_years <= 20;
    [~, loc]    = max(S(solar_band));
    solar_per   = period_years(solar_band);
    fprintf('Dominant cycle detected at: %.2f years\n', solar_per(loc));
catch ME
    fprintf('[OceanViz Error in PSD] %s\n', ME.message);
    figure;
    plot(period_years(valid), 10*log10(S(valid)), 'LineWidth', 1.5, ...
        'DisplayName', 'Sample Spectrum S(\omega)');
    xlim([0.5, 40]);
    title('Population Spectrum (Autocovariance Method)');
    xlabel('Period (Years)'); ylabel('PSD (dB \cdot day)');
    legend('Location', 'best');
end

%% Step 1.5: Data Export for Python
fprintf('1.5 Exporting data to Parquet for Python Phase...\n');
if ~exist('lake/processed', 'dir')
    mkdir('lake/processed');
end
parquetwrite('lake/processed/train.parquet', timetable2table(TT_train));
parquetwrite('lake/processed/test.parquet', timetable2table(TT_test));

fprintf('MATLAB Phase Complete. Proceed to Python script for Phase 2.\n');
