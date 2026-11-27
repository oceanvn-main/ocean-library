% EXP_HAMILTON_CH6_SELECTIVE_FILTER
% Binance BTC/USDT weekly selective-filter + phase/coherence experiment.
%
% Goals:
%   1. Confirm several-year spectral concentration.
%   2. Reconstruct the several-year band while preserving Fourier phase.
%   3. Fit a ~4.2-year harmonic and estimate calendar trough/peak dates.
%   4. Slide a trailing ~4.2-year window to study local phase.
%   5. Use a longer window to test whether the preferred period is stable.
%   6. Optionally compare the spectral peak against an AR(1) red-noise null.
%
% IMPORTANT:
%   - Frequency-domain peak strength tells us "how much".
%   - Harmonic/Fourier phase tells us "where in calendar time".
%   - A ~4.2-year rolling window contains only ~1 cycle, so use it mainly
%     for PHASE LOCALIZATION, not precise frequency estimation.

clc; clear; close all;

%% Configuration
SYMBOL = "BTCUSDT";
INTERVAL = "1w";
REQUEST_YEARS = 12;

EXPORT_FIGURES = true;
EXPORT_DPI = 160;

% Spectral hypothesis / exploratory range
REFERENCE_PERIOD_YEARS = 4.20;       % predeclared phase reference
SEARCH_PERIOD_YEARS = [2 6];

% Several-year reconstruction band
LONG_BAND_PERIOD_YEARS = [3.5 5.0];

% Short rolling window: phase localization
ROLLING_PHASE_WINDOW_YEARS = 4.20;
ROLLING_STEP_WEEKS = 4;

% Long rolling window: period stability
ROLLING_PERIOD_WINDOW_YEARS = 8.0;
PERIOD_SCAN_YEARS = 3.5:0.025:5.0;

% Optional null-model audit
RUN_RED_NOISE_NULL = true;
NUM_SURROGATES = 500;
RNG_SEED = 1847;

SAMPLES_PER_YEAR = 365.2425 / 7;

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(fileparts(scriptDir)));
resultDir = fullfile(scriptDir,'Results');
cacheDir = fullfile(resultDir,'cache');

if ~exist(resultDir,'dir'), mkdir(resultDir); end
addpath(projectRoot);

viz = OceanViz.Core('Theme','Scientific');
palette = viz.getColorSequence(6);

%% Robust weekly Binance history
requestEnd = datetime('now','TimeZone','UTC');
requestStart = requestEnd - calyears(REQUEST_YEARS);

reader = OceanPipeline.Ingestion.RobustBinanceHistoryReader( ...
    'Symbol', SYMBOL, ...
    'MarketType', 'spot', ...
    'Interval', INTERVAL, ...
    'MaxRetries', 3, ...
    'CacheDirectory', cacheDir, ...
    'ClosedCandlesOnly', true);

market = reader.readRange(requestStart,requestEnd,'UseCache',true);

TT = market.Data;
coverage = market.Coverage;

fprintf('\nCoverage audit\n');
fprintf('  Requested: %s to %s (%.2f years)\n', ...
    string(coverage.RequestedStart), ...
    string(coverage.RequestedEnd), ...
    coverage.RequestedYears);

fprintf('  Returned : %s to %s (%.2f years, %d weekly candles)\n', ...
    string(coverage.ActualStart), ...
    string(coverage.ActualEnd), ...
    coverage.ActualYears, ...
    coverage.ObservationCount);

fprintf('  Observable four-year cycles: %.2f\n', ...
    coverage.FourYearCycles);

if coverage.StartTruncated
    fprintf(['  NOTE: Binance BTC/USDT begins after the requested start. ' ...
        'A full 12-year Binance history does not exist.\n']);
end

%% Original signal
figCandle = viz.createFigure( ...
    'BTC/USDT Weekly Candlestick — Original Signal');

figCandle.Position = [60 50 1450 800];

ax = axes(figCandle);

OceanViz.Charts.CandlestickChart.plot(ax,TT, ...
    'Title', sprintf( ...
        '%s Weekly Binance Spot — Original OHLCV Signal', SYMBOL), ...
    'Theme', 'Scientific', ...
    'Colors', viz.Colors, ...
    'MAWindows', [13 52 208]);

chartAxes = findall(figCandle,'Type','axes');

for k = 1:numel(chartAxes)
    datetick(chartAxes(k),'x','yyyy','keeplimits');
end

%% Selective filtering and spectral audit
filterPipeline = OceanPipeline.Transforms.SelectiveFilterPipeline( ...
    'SamplesPerYear', SAMPLES_PER_YEAR, ...
    'SmoothWindow', 13, ...
    'SearchPeriodYears', SEARCH_PERIOD_YEARS, ...
    'TargetPeriodYears', REFERENCE_PERIOD_YEARS);

analysis = filterPipeline.run(TT.Time,log(TT.Close));

layers = analysis.Layers;
audit = analysis.Audit;

fprintf('\nSeveral-year spectral audit\n');

fprintf('  Dominant period in 2-6 year band : %.3f years\n', ...
    audit.EstimatedPeriodYears);

fprintf('  Peak / band-median power         : %.3f\n', ...
    audit.PeakToBandMedian);

fprintf('  Frequency resolution             : %.5f cycles/year\n', ...
    audit.FrequencyResolution);

fprintf('  Observed target cycles           : %.2f\n', ...
    audit.ObservedTargetCycles);

fprintf('  Reference frequency              : %.4f cycles/year\n', ...
    1 / REFERENCE_PERIOD_YEARS);

fprintf('  Estimated frequency              : %.4f cycles/year\n', ...
    1 / audit.EstimatedPeriodYears);

%% Existing time representation
figTime = viz.createFigure( ...
    'BTC Four-Year Cycle — Filtered Time Layers');

figTime.Position = [80 40 1400 940];

tlTime = tiledlayout( ...
    figTime,4,1, ...
    'TileSpacing','compact', ...
    'Padding','compact');

for k = 1:4
    ax = nexttile(tlTime,k);

    colororder(ax,palette(k,:));

    plot( ...
        ax, ...
        layers(k).Time, ...
        layers(k).Signal, ...
        'LineWidth',1.0);

    viz.styleAxes(ax);
    box(ax,'on');

    yline( ...
        ax,0,':', ...
        'Color',viz.Colors.Grid, ...
        'HandleVisibility','off');

    title(ax,layers(k).Name,'Color',viz.Colors.Text);
    ylabel(ax,'Filtered log price');

    if k < 4
        ax.XTickLabel = [];
    else
        xlabel(ax,'Date');
    end
end

title(tlTime,{ ...
    'BTC/USDT Weekly Signal After Each Selective-Filtering Layer', ...
    'Trend, annual, half-year, and sub-quarter components removed sequentially'}, ...
    'Color',viz.Colors.Text, ...
    'FontWeight','bold');

%% Existing frequency representation
figSpectrum = viz.createFigure( ...
    'BTC Four-Year Cycle — Filtered Spectra');

figSpectrum.Position = [100 60 1400 820];

tlSpectrum = tiledlayout( ...
    figSpectrum,2,2, ...
    'TileSpacing','compact', ...
    'Padding','compact');

for k = 1:4
    ax = nexttile(tlSpectrum,k);

    colororder(ax,palette(k,:));

    yyaxis(ax,'left');

    plot( ...
        ax, ...
        layers(k).Frequency, ...
        layers(k).Periodogram, ...
        'LineWidth',1.05);

    ylabel(ax,'Raw periodogram');

    yyaxis(ax,'right');

    plot( ...
        ax, ...
        layers(k).Frequency, ...
        layers(k).CumulativeGain, ...
        '--', ...
        'Color',viz.Colors.Tertiary, ...
        'LineWidth',0.9);

    ylabel(ax,'Filter gain');

    viz.styleAxes(ax);
    box(ax,'on');

    % Predeclared ~4.2-year reference
    xline( ...
        ax,1/REFERENCE_PERIOD_YEARS,':', ...
        'Color',viz.Colors.Text, ...
        'LineWidth',1.0, ...
        'HandleVisibility','off');

    % Empirically estimated several-year peak
    xline( ...
        ax,1/audit.EstimatedPeriodYears,'--', ...
        'Color',viz.Colors.Tertiary, ...
        'LineWidth',1.0, ...
        'HandleVisibility','off');

    xlim(ax,[0 2]);

    title(ax,layers(k).Name,'Color',viz.Colors.Text);
    xlabel(ax,'Frequency (cycles/year)');
end

title(tlSpectrum,{ ...
    'BTC/USDT Weekly Spectrum After Each Filtering Layer', ...
    sprintf( ...
        ['Dotted: %.2f-year reference; dashed: estimated %.2f-year period'], ...
        REFERENCE_PERIOD_YEARS, ...
        audit.EstimatedPeriodYears)}, ...
    'Color',viz.Colors.Text, ...
    'FontWeight','bold');

%% ================================================================
%  NEW 1: PHASE-PRESERVING SEVERAL-YEAR BAND RECONSTRUCTION
% ================================================================

% Primary source for timing:
% use only the detrended layer because later filters may introduce
% phase delay unless they are explicitly zero-phase.
phaseTime = layers(1).Time;
phaseSignal = layers(1).Signal;

valid = isfinite(phaseSignal);

phaseTime = phaseTime(valid);
phaseSignal = phaseSignal(valid);

bandSignal = fftBandReconstruct( ...
    phaseSignal, ...
    SAMPLES_PER_YEAR, ...
    LONG_BAND_PERIOD_YEARS);

figBand = viz.createFigure( ...
    'BTC Several-Year Band — Phase-Preserving Reconstruction');

figBand.Position = [120 70 1400 760];

tlBand = tiledlayout( ...
    figBand,2,1, ...
    'TileSpacing','compact', ...
    'Padding','compact');

ax = nexttile(tlBand,1);

plot( ...
    ax,phaseTime,phaseSignal, ...
    'LineWidth',0.85, ...
    'DisplayName','Detrended log price');

hold(ax,'on');

plot( ...
    ax,phaseTime,bandSignal, ...
    'LineWidth',1.8, ...
    'DisplayName',sprintf( ...
        '%.1f-%.1f year Fourier band', ...
        LONG_BAND_PERIOD_YEARS(1), ...
        LONG_BAND_PERIOD_YEARS(2)));

viz.styleAxes(ax);
box(ax,'on');
legend(ax,'Location','best');

ylabel(ax,'Log-price deviation');

title(ax, ...
    'Several-Year Component Reconstructed from Complex Fourier Coefficients');

% Plot only the reconstructed cycle separately
ax = nexttile(tlBand,2);

plot( ...
    ax,phaseTime,bandSignal, ...
    'LineWidth',1.7);

yline( ...
    ax,0,':', ...
    'Color',viz.Colors.Grid, ...
    'HandleVisibility','off');

viz.styleAxes(ax);
box(ax,'on');

ylabel(ax,'Band component');
xlabel(ax,'Date');

title(ax, ...
    ['This reconstruction preserves Fourier phase; ' ...
     'periodogram magnitude alone would not']);

%% ================================================================
%  NEW 2: GLOBAL FIXED-PERIOD HARMONIC FIT
% ================================================================

globalAnchor = phaseTime(1);

% Predeclared 4.2-year model
globalReferenceFit = fitFixedPeriodHarmonic( ...
    phaseTime, ...
    phaseSignal, ...
    REFERENCE_PERIOD_YEARS, ...
    globalAnchor);

% Exploratory fit using period found by the spectrum
globalEstimatedFit = fitFixedPeriodHarmonic( ...
    phaseTime, ...
    phaseSignal, ...
    audit.EstimatedPeriodYears, ...
    globalAnchor);

fprintf('\nGlobal harmonic phase audit\n');

fprintf('\n  Predeclared %.2f-year harmonic\n', ...
    REFERENCE_PERIOD_YEARS);

fprintf('    Amplitude                   : %.5f\n', ...
    globalReferenceFit.Amplitude);

fprintf('    Incremental variance fit    : %.4f\n', ...
    globalReferenceFit.FitStrength);

fprintf('    Previous fitted trough      : %s\n', ...
    string(globalReferenceFit.PreviousTrough));

fprintf('    Next fitted trough          : %s\n', ...
    string(globalReferenceFit.NextTrough));

fprintf('    Current cycle fraction      : %.3f\n', ...
    globalReferenceFit.CycleFraction);

fprintf('\n  Exploratory %.3f-year harmonic\n', ...
    audit.EstimatedPeriodYears);

fprintf('    Amplitude                   : %.5f\n', ...
    globalEstimatedFit.Amplitude);

fprintf('    Incremental variance fit    : %.4f\n', ...
    globalEstimatedFit.FitStrength);

fprintf('    Previous fitted trough      : %s\n', ...
    string(globalEstimatedFit.PreviousTrough));

fprintf('    Next fitted trough          : %s\n', ...
    string(globalEstimatedFit.NextTrough));

fprintf('    Current cycle fraction      : %.3f\n', ...
    globalEstimatedFit.CycleFraction);

%% Global harmonic visualization
%% Global harmonic visualization — clearer phase encoding
figGlobalPhase = viz.createFigure( ...
    'BTC Several-Year Harmonic — Calendar Phase');

figGlobalPhase.Position = [140 80 1450 780];

ax = axes(figGlobalPhase);
hold(ax,'on');

% -------------------------------------------------------------------------
% Visual encoding
% -------------------------------------------------------------------------
signalColor   = palette(1,:);   % observed detrended signal
harmonicColor = palette(3,:);   % fitted several-year harmonic

troughColor   = palette(2,:);   % cool / distinct
peakColor     = palette(4,:);   % warm / distinct

% -------------------------------------------------------------------------
% Main signals
% -------------------------------------------------------------------------
plot( ...
    ax, ...
    phaseTime, ...
    phaseSignal, ...
    'Color',signalColor, ...
    'LineWidth',0.85, ...
    'DisplayName','Detrended signal');

plot( ...
    ax, ...
    phaseTime, ...
    globalReferenceFit.CycleComponent, ...
    'Color',harmonicColor, ...
    'LineWidth',2.2, ...
    'DisplayName',sprintf( ...
        'Fixed %.2f-year harmonic', ...
        REFERENCE_PERIOD_YEARS));

% -------------------------------------------------------------------------
% Turning points
% -------------------------------------------------------------------------
[troughTimes,peakTimes] = harmonicTurningPoints( ...
    globalAnchor, ...
    phaseTime(1), ...
    phaseTime(end), ...
    REFERENCE_PERIOD_YEARS, ...
    globalReferenceFit.Phase);

% Add one labeled trough line for legend
if ~isempty(troughTimes)
    xline( ...
        ax,troughTimes(1),':', ...
        'Color',troughColor, ...
        'LineWidth',2.2, ...
        'DisplayName','Fitted trough');

    for k = 2:numel(troughTimes)
        xline( ...
            ax,troughTimes(k),':', ...
            'Color',troughColor, ...
            'LineWidth',2.2, ...
            'HandleVisibility','off');
    end
end

% Add one labeled peak line for legend
if ~isempty(peakTimes)
    xline( ...
        ax,peakTimes(1),'--', ...
        'Color',peakColor, ...
        'LineWidth',2.0, ...
        'DisplayName','Fitted peak');

    for k = 2:numel(peakTimes)
        xline( ...
            ax,peakTimes(k),'--', ...
            'Color',peakColor, ...
            'LineWidth',2.0, ...
            'HandleVisibility','off');
    end
end

% -------------------------------------------------------------------------
% Axes styling
% -------------------------------------------------------------------------
yline( ...
    ax,0,'-', ...
    'Color',viz.Colors.Grid, ...
    'LineWidth',0.8, ...
    'HandleVisibility','off');

viz.styleAxes(ax);
box(ax,'on');

xlabel(ax,'Date');
ylabel(ax,'Detrended log-price residual');

legend(ax, ...
    'Location','northeast', ...
    'Box','on');

title(ax,{ ...
    sprintf('Calendar Phase of a Fixed %.2f-Year Harmonic', ...
        REFERENCE_PERIOD_YEARS), ...
    'Dotted = fitted troughs     |     Dashed = fitted peaks'});

%% ================================================================
%  NEW 3: TRAILING ~4.2-YEAR ROLLING PHASE ANALYSIS
% ================================================================

rollingDetrended = rollingPhaseAudit( ...
    layers(1).Time, ...
    layers(1).Signal, ...
    REFERENCE_PERIOD_YEARS, ...
    ROLLING_PHASE_WINDOW_YEARS, ...
    ROLLING_STEP_WEEKS, ...
    globalAnchor);

% Robustness comparison:
% only interpret timing from this if SelectiveFilterPipeline is zero-phase.
rollingFinal = rollingPhaseAudit( ...
    layers(end).Time, ...
    layers(end).Signal, ...
    REFERENCE_PERIOD_YEARS, ...
    ROLLING_PHASE_WINDOW_YEARS, ...
    ROLLING_STEP_WEEKS, ...
    globalAnchor);

figRollingPhase = viz.createFigure( ...
    'BTC Rolling 4.2-Year Phase Audit');

figRollingPhase.Position = [150 60 1450 900];

tlPhase = tiledlayout( ...
    figRollingPhase,3,1, ...
    'TileSpacing','compact', ...
    'Padding','compact');

% Fit strength
ax = nexttile(tlPhase,1);

plot( ...
    ax, ...
    rollingDetrended.EndTime, ...
    rollingDetrended.FitStrength, ...
    'LineWidth',1.5, ...
    'DisplayName','Detrended layer');

hold(ax,'on');

plot( ...
    ax, ...
    rollingFinal.EndTime, ...
    rollingFinal.FitStrength, ...
    '--', ...
    'LineWidth',1.2, ...
    'DisplayName','Final filtered layer');

viz.styleAxes(ax);
box(ax,'on');

ylabel(ax,'Incremental fit');

legend(ax,'Location','best');

title(ax, ...
    ['Rolling harmonic strength: fraction of local residual variation ' ...
     'explained beyond intercept + local trend']);

% Cycle phase
ax = nexttile(tlPhase,2);

plot( ...
    ax, ...
    rollingDetrended.EndTime, ...
    rollingDetrended.CycleFraction, ...
    'LineWidth',1.5, ...
    'DisplayName','Detrended layer');

hold(ax,'on');

plot( ...
    ax, ...
    rollingFinal.EndTime, ...
    rollingFinal.CycleFraction, ...
    '--', ...
    'LineWidth',1.2, ...
    'DisplayName','Final filtered layer');

yline(ax,0,':','HandleVisibility','off');
yline(ax,0.5,':','HandleVisibility','off');
yline(ax,1,':','HandleVisibility','off');

ylim(ax,[0 1]);

viz.styleAxes(ax);
box(ax,'on');

ylabel(ax,'Cycle fraction');

title(ax,{ ...
    'Local Phase of the Fixed 4.2-Year Harmonic', ...
    '0 = trough/start; 0.5 = peak; 1 = next trough/end'});

% Implied current-cycle trough date
ax = nexttile(tlPhase,3);

plot( ...
    ax, ...
    rollingDetrended.EndTime, ...
    rollingDetrended.PreviousTrough, ...
    'LineWidth',1.5);

hold(ax,'on');

plot( ...
    ax, ...
    rollingFinal.EndTime, ...
    rollingFinal.PreviousTrough, ...
    '--', ...
    'LineWidth',1.1);

viz.styleAxes(ax);
box(ax,'on');

xlabel(ax,'Rolling-window end date');
ylabel(ax,'Implied cycle-start trough');

title(ax, ...
    ['If phase is coherent, neighboring windows should imply ' ...
     'similar trough dates rather than jumping randomly']);

%% Latest trailing-window interpretation
if ~isempty(rollingDetrended)
    latest = rollingDetrended(end,:);

    fprintf('\nLatest trailing %.2f-year phase estimate\n', ...
        ROLLING_PHASE_WINDOW_YEARS);

    fprintf('  Window end           : %s\n', ...
        string(latest.EndTime));

    fprintf('  Previous trough      : %s\n', ...
        string(latest.PreviousTrough));

    fprintf('  Next trough          : %s\n', ...
        string(latest.NextTrough));

    fprintf('  Cycle fraction       : %.3f\n', ...
        latest.CycleFraction);

    fprintf('  Incremental fit      : %.4f\n', ...
        latest.FitStrength);

    fprintf('  Harmonic amplitude   : %.5f\n', ...
        latest.Amplitude);
end

%% ================================================================
%  NEW 4: LONGER-WINDOW PERIOD-STABILITY SCAN
% ================================================================

periodScan = rollingPeriodScan( ...
    layers(1).Time, ...
    layers(1).Signal, ...
    PERIOD_SCAN_YEARS, ...
    ROLLING_PERIOD_WINDOW_YEARS, ...
    ROLLING_STEP_WEEKS, ...
    globalAnchor);

if ~isempty(periodScan.EndTime)

    figPeriodScan = viz.createFigure( ...
        'BTC Several-Year Period Stability');

    figPeriodScan.Position = [170 70 1450 880];

    tlScan = tiledlayout( ...
        figPeriodScan,3,1, ...
        'TileSpacing','compact', ...
        'Padding','compact');

    % Heatmap of harmonic strength
    ax = nexttile(tlScan,1);

    xNum = datenum(periodScan.EndTime);

    imagesc( ...
        ax, ...
        xNum, ...
        PERIOD_SCAN_YEARS, ...
        periodScan.Score.');

    axis(ax,'xy');

    datetick(ax,'x','yyyy','keeplimits');

    colorbar(ax);

    ylabel(ax,'Candidate period (years)');

    title(ax,{ ...
        sprintf('Rolling %.1f-Year Period Scan', ...
            ROLLING_PERIOD_WINDOW_YEARS), ...
        'Color = incremental variance explained by harmonic'});

    hold(ax,'on');

    plot( ...
        ax, ...
        datenum(periodScan.EndTime), ...
        periodScan.BestPeriod, ...
        'k-', ...
        'LineWidth',1.4);

    % Best period
    ax = nexttile(tlScan,2);

    plot( ...
        ax, ...
        periodScan.EndTime, ...
        periodScan.BestPeriod, ...
        'LineWidth',1.5);

    yline( ...
        ax, ...
        REFERENCE_PERIOD_YEARS, ...
        ':', ...
        'HandleVisibility','off');

    ylim(ax,[min(PERIOD_SCAN_YEARS) max(PERIOD_SCAN_YEARS)]);

    viz.styleAxes(ax);
    box(ax,'on');

    ylabel(ax,'Best period (years)');

    title(ax, ...
        'Does the preferred several-year period remain near the same value?');

    % Best fit strength
    ax = nexttile(tlScan,3);

    plot( ...
        ax, ...
        periodScan.EndTime, ...
        periodScan.BestScore, ...
        'LineWidth',1.5);

    viz.styleAxes(ax);
    box(ax,'on');

    ylabel(ax,'Best incremental fit');
    xlabel(ax,'Window end');

    title(ax, ...
        'A stable period is not meaningful if its harmonic fit is weak');
else
    fprintf('\nPeriod-stability scan skipped: not enough history.\n');
end

%% ================================================================
%  NEW 5: OPTIONAL AR(1) RED-NOISE NULL TEST
% ================================================================

if RUN_RED_NOISE_NULL

    rng(RNG_SEED);

    nullSignal = layers(1).Signal;
    nullSignal = nullSignal(isfinite(nullSignal));

    nullAudit = redNoiseSpectralNull( ...
        nullSignal, ...
        SAMPLES_PER_YEAR, ...
        SEARCH_PERIOD_YEARS, ...
        NUM_SURROGATES);

    fprintf('\nAR(1) red-noise null test\n');

    fprintf('  Estimated AR(1) phi              : %.4f\n', ...
        nullAudit.Phi);

    fprintf('  Observed peak/band-median ratio  : %.4f\n', ...
        nullAudit.ObservedStatistic);

    fprintf('  Monte Carlo p-value              : %.4f\n', ...
        nullAudit.PValue);

    fprintf(['  Interpretation: this asks whether a several-year spectral ' ...
        'peak at least this concentrated can arise from a persistent ' ...
        'but non-periodic AR(1)-type process.\n']);
end

%% Export
if EXPORT_FIGURES

    exportgraphics( ...
        figCandle, ...
        fullfile(resultDir,'exp_binance_weekly_candlestick.png'), ...
        'Resolution',EXPORT_DPI);

    exportgraphics( ...
        figTime, ...
        fullfile(resultDir,'exp_binance_filter_time_layers.png'), ...
        'Resolution',EXPORT_DPI);

    exportgraphics( ...
        figSpectrum, ...
        fullfile(resultDir,'exp_binance_filter_spectra.png'), ...
        'Resolution',EXPORT_DPI);

    exportgraphics( ...
        figBand, ...
        fullfile(resultDir,'exp_binance_fouryear_band_reconstruction.png'), ...
        'Resolution',EXPORT_DPI);

    exportgraphics( ...
        figGlobalPhase, ...
        fullfile(resultDir,'exp_binance_global_cycle_phase.png'), ...
        'Resolution',EXPORT_DPI);

    exportgraphics( ...
        figRollingPhase, ...
        fullfile(resultDir,'exp_binance_rolling_phase.png'), ...
        'Resolution',EXPORT_DPI);

    if exist('figPeriodScan','var')
        exportgraphics( ...
            figPeriodScan, ...
            fullfile(resultDir,'exp_binance_period_stability.png'), ...
            'Resolution',EXPORT_DPI);
    end
end

fprintf('\nInterpretation guardrails\n');

fprintf(['  1. A periodogram peak identifies a frequency/time scale, ' ...
    'not calendar phase.\n']);

fprintf(['  2. The fixed harmonic estimates phase, but does not prove ' ...
    'a stable causal market cycle.\n']);

fprintf(['  3. A %.2f-year window contains only about one %.2f-year cycle; ' ...
    'use it mainly for phase localization.\n'], ...
    ROLLING_PHASE_WINDOW_YEARS, ...
    REFERENCE_PERIOD_YEARS);

fprintf(['  4. Period stability requires longer windows and multiple ' ...
    'repetitions; Binance provides only %.2f observable four-year cycles.\n'], ...
    coverage.FourYearCycles);

fprintf(['  5. Timing from aggressively filtered layers is trustworthy only ' ...
    'if those filters are zero-phase or their phase response is corrected.\n']);

%% ========================================================================
% LOCAL FUNCTIONS
% ========================================================================

function bandSignal = fftBandReconstruct(signal,fs,periodBandYears)
% Reconstruct a time-domain component from a selected frequency band
% while retaining the original complex Fourier coefficients (phase included).

    y = signal(:);

    y = y - mean(y,'omitnan');

    N = numel(y);

    Y = fft(y);

    f = (0:N-1)' * fs / N;

    % Convert DFT frequencies to signed frequencies
    fSigned = f;

    maskNegative = fSigned > fs/2;
    fSigned(maskNegative) = fSigned(maskNegative) - fs;

    fLow = 1 / max(periodBandYears);
    fHigh = 1 / min(periodBandYears);

    keep = ...
        abs(fSigned) >= fLow & ...
        abs(fSigned) <= fHigh;

    Yband = zeros(size(Y));
    Yband(keep) = Y(keep);

    bandSignal = real(ifft(Yband));
end

function fit = fitFixedPeriodHarmonic(time,y,periodYears,globalAnchor)
% Fit:
%
% y(t) = a0 + a1*t_local
%        + alpha*cos(2*pi*t_global/P)
%        + delta*sin(2*pi*t_global/P)
%        + error
%
% The local trend absorbs slow drift.
% The global time coordinate makes phase/calendar dates comparable.

    y = y(:);
    time = time(:);

    valid = isfinite(y);

    y = y(valid);
    time = time(valid);

    tGlobal = days(time - globalAnchor) / 365.2425;

    tLocal = tGlobal - mean(tGlobal);

    omega = 2*pi / periodYears;

    C = cos(omega*tGlobal);
    S = sin(omega*tGlobal);

    Xbase = [ ...
        ones(size(y)), ...
        tLocal];

    Xfull = [ ...
        Xbase, ...
        C, ...
        S];

    betaBase = Xbase \ y;
    betaFull = Xfull \ y;

    yBase = Xbase * betaBase;
    yFull = Xfull * betaFull;

    cycleComponent = ...
        betaFull(3)*C + ...
        betaFull(4)*S;

    SSEbase = sum((y - yBase).^2);
    SSEfull = sum((y - yFull).^2);

    fitStrength = 1 - SSEfull / SSEbase;

    alpha = betaFull(3);
    deltaCoeff = betaFull(4);

    amplitude = hypot(alpha,deltaCoeff);

    % a*cos(wt) + b*sin(wt)
    % = R*cos(wt - phase)
    phase = atan2(deltaCoeff,alpha);

    tEndYears = tGlobal(end);

    % trough condition:
    % omega*t - phase = pi + 2*pi*k
    kPrevious = floor( ...
        (omega*tEndYears - phase - pi) / (2*pi));

    previousTroughYears = ...
        (phase + pi + 2*pi*kPrevious) / omega;

    nextTroughYears = ...
        previousTroughYears + periodYears;

    previousTrough = ...
        globalAnchor + days(365.2425*previousTroughYears);

    nextTrough = ...
        globalAnchor + days(365.2425*nextTroughYears);

    cycleFraction = ...
        (tEndYears - previousTroughYears) / periodYears;

    cycleFraction = mod(cycleFraction,1);

    fit = struct();

    fit.PeriodYears = periodYears;
    fit.Amplitude = amplitude;
    fit.Phase = phase;
    fit.Alpha = alpha;
    fit.Delta = deltaCoeff;

    fit.FitStrength = fitStrength;

    fit.CycleComponent = cycleComponent;
    fit.FittedSignal = yFull;

    fit.PreviousTrough = previousTrough;
    fit.NextTrough = nextTrough;

    fit.CycleFraction = cycleFraction;
end

function T = rollingPhaseAudit( ...
    time,y,periodYears,windowYears,stepWeeks,globalAnchor)

    time = time(:);
    y = y(:);

    valid = isfinite(y);

    time = time(valid);
    y = y(valid);

    if isempty(time)
        T = table();
        return;
    end

    windowDuration = days(365.2425*windowYears);

    firstPossible = time(1) + windowDuration;

    candidateIdx = find(time >= firstPossible);

    candidateIdx = candidateIdx(1:stepWeeks:end);

    nWindows = numel(candidateIdx);

    endTime = repmat(datetime('NaT','TimeZone','UTC'),nWindows,1);
    previousTrough = endTime;
    nextTrough = endTime;

    amplitude = nan(nWindows,1);
    fitStrength = nan(nWindows,1);
    cycleFraction = nan(nWindows,1);

    keep = false(nWindows,1);

    for r = 1:nWindows

        idxEnd = candidateIdx(r);

        tEnd = time(idxEnd);
        tStart = tEnd - windowDuration;

        idx = time >= tStart & time <= tEnd;

        % Rough minimum expected observations
        expectedN = windowYears * 365.2425 / 7;

        if nnz(idx) < 0.80*expectedN
            continue;
        end

        fit = fitFixedPeriodHarmonic( ...
            time(idx), ...
            y(idx), ...
            periodYears, ...
            globalAnchor);

        endTime(r) = tEnd;
        previousTrough(r) = fit.PreviousTrough;
        nextTrough(r) = fit.NextTrough;

        amplitude(r) = fit.Amplitude;
        fitStrength(r) = fit.FitStrength;
        cycleFraction(r) = fit.CycleFraction;

        keep(r) = true;
    end

    T = table( ...
        endTime(keep), ...
        previousTrough(keep), ...
        nextTrough(keep), ...
        amplitude(keep), ...
        fitStrength(keep), ...
        cycleFraction(keep), ...
        'VariableNames',{ ...
            'EndTime', ...
            'PreviousTrough', ...
            'NextTrough', ...
            'Amplitude', ...
            'FitStrength', ...
            'CycleFraction'});
end

function scan = rollingPeriodScan( ...
    time,y,periodGrid,windowYears,stepWeeks,globalAnchor)

    time = time(:);
    y = y(:);

    valid = isfinite(y);

    time = time(valid);
    y = y(valid);

    windowDuration = days(365.2425*windowYears);

    if time(end) - time(1) < windowDuration
        scan = struct( ...
            'EndTime',datetime.empty(0,1), ...
            'Score',[], ...
            'BestPeriod',[], ...
            'BestScore',[]);
        return;
    end

    firstPossible = time(1) + windowDuration;

    candidateIdx = find(time >= firstPossible);
    candidateIdx = candidateIdx(1:stepWeeks:end);

    nWindows = numel(candidateIdx);
    nPeriods = numel(periodGrid);

    score = nan(nWindows,nPeriods);

    endTime = repmat(datetime('NaT','TimeZone','UTC'),nWindows,1);

    for r = 1:nWindows

        idxEnd = candidateIdx(r);

        tEnd = time(idxEnd);
        tStart = tEnd - windowDuration;

        idx = time >= tStart & time <= tEnd;

        endTime(r) = tEnd;

        for j = 1:nPeriods

            fit = fitFixedPeriodHarmonic( ...
                time(idx), ...
                y(idx), ...
                periodGrid(j), ...
                globalAnchor);

            score(r,j) = fit.FitStrength;
        end
    end

    [bestScore,bestIdx] = max(score,[],2,'omitnan');

    bestPeriod = periodGrid(bestIdx).';

    scan = struct();

    scan.EndTime = endTime;
    scan.Score = score;
    scan.BestPeriod = bestPeriod;
    scan.BestScore = bestScore;
end

function [troughTimes,peakTimes] = harmonicTurningPoints( ...
    anchor,tMin,tMax,periodYears,phase)

    omega = 2*pi / periodYears;

    tMinYears = days(tMin-anchor)/365.2425;
    tMaxYears = days(tMax-anchor)/365.2425;

    kGrid = -20:40;

    troughYears = ...
        (phase + pi + 2*pi*kGrid) / omega;

    peakYears = ...
        (phase + 2*pi*kGrid) / omega;

    troughYears = troughYears( ...
        troughYears >= tMinYears & ...
        troughYears <= tMaxYears);

    peakYears = peakYears( ...
        peakYears >= tMinYears & ...
        peakYears <= tMaxYears);

    troughTimes = ...
        anchor + days(365.2425*troughYears);

    peakTimes = ...
        anchor + days(365.2425*peakYears);
end

function audit = redNoiseSpectralNull( ...
    signal,fs,searchPeriodYears,nSurrogates)

    y = signal(:);

    y = y - mean(y);

    xLag = y(1:end-1);
    xNow = y(2:end);

    phi = ...
        (xLag' * xNow) / ...
        (xLag' * xLag);

    residual = ...
        xNow - phi*xLag;

    sigmaInnovation = ...
        sqrt(mean(residual.^2));

    observedStatistic = bandPeakStatistic( ...
        y, ...
        fs, ...
        searchPeriodYears);

    N = numel(y);

    surrogateStatistic = nan(nSurrogates,1);

    burnIn = max(500,5*N);

    for b = 1:nSurrogates

        e = ...
            sigmaInnovation * ...
            randn(N + burnIn,1);

        x = zeros(N + burnIn,1);

        if abs(phi) < 1
            x(1) = ...
                sigmaInnovation / ...
                sqrt(1-phi^2) * ...
                randn;
        end

        for t = 2:numel(x)
            x(t) = phi*x(t-1) + e(t);
        end

        x = x(burnIn+1:end);

        surrogateStatistic(b) = ...
            bandPeakStatistic( ...
                x, ...
                fs, ...
                searchPeriodYears);
    end

    pValue = ...
        (1 + sum( ...
            surrogateStatistic >= observedStatistic)) ...
        / ...
        (nSurrogates + 1);

    audit = struct();

    audit.Phi = phi;
    audit.ObservedStatistic = observedStatistic;
    audit.SurrogateStatistic = surrogateStatistic;
    audit.PValue = pValue;
end

function statistic = bandPeakStatistic( ...
    signal,fs,searchPeriodYears)

    y = signal(:);

    y = y - mean(y);

    N = numel(y);

    Y = fft(y);

    nPositive = floor(N/2) + 1;

    f = ...
        (0:nPositive-1)' * ...
        fs / N;

    P = ...
        abs(Y(1:nPositive)).^2 / N;

    fLow = 1 / max(searchPeriodYears);
    fHigh = 1 / min(searchPeriodYears);

    idx = ...
        f >= fLow & ...
        f <= fHigh;

    bandPower = P(idx);

    statistic = ...
        max(bandPower) / ...
        median(bandPower);
end