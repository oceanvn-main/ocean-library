% EXP_HAMILTON_CH6_SELECTIVE_FILTER
% Binance BTC/USDT weekly selective-filter experiment for a four-year cycle.

clc; clear; close all;

%% Configuration
SYMBOL="BTCUSDT";
INTERVAL="1w";
REQUEST_YEARS=12;
EXPORT_FIGURES=true;
EXPORT_DPI=160;

scriptDir=fileparts(mfilename('fullpath'));
projectRoot=fileparts(fileparts(fileparts(scriptDir)));
resultDir=fullfile(scriptDir,'Results');
cacheDir=fullfile(resultDir,'cache');
if ~exist(resultDir,'dir'), mkdir(resultDir); end
addpath(projectRoot);

viz=OceanViz.Core('Theme','Scientific');
palette=viz.getColorSequence(4);

%% Robust weekly Binance history
requestEnd=datetime('now','TimeZone','UTC');
requestStart=requestEnd-calyears(REQUEST_YEARS);
reader=OceanPipeline.Ingestion.RobustBinanceHistoryReader( ...
    'Symbol',SYMBOL,'MarketType','spot','Interval',INTERVAL, ...
    'MaxRetries',3,'CacheDirectory',cacheDir,'ClosedCandlesOnly',true);
market=reader.readRange(requestStart,requestEnd,'UseCache',true);
TT=market.Data;
coverage=market.Coverage;

fprintf('\nCoverage audit\n');
fprintf('  Requested: %s to %s (%.2f years)\n', ...
    string(coverage.RequestedStart),string(coverage.RequestedEnd),coverage.RequestedYears);
fprintf('  Returned : %s to %s (%.2f years, %d weekly candles)\n', ...
    string(coverage.ActualStart),string(coverage.ActualEnd), ...
    coverage.ActualYears,coverage.ObservationCount);
fprintf('  Observable four-year cycles: %.2f\n',coverage.FourYearCycles);
if coverage.StartTruncated
    fprintf(['  NOTE: Binance BTC/USDT begins after the requested start. ' ...
        'A full 12-year Binance history does not exist.\n']);
end

%% Original signal: OceanViz Candlestick component
figCandle=viz.createFigure('BTC/USDT Weekly Candlestick — Original Signal');
figCandle.Position=[60 50 1450 800];
ax=axes(figCandle);
OceanViz.Charts.CandlestickChart.plot(ax,TT, ...
    'Title',sprintf('%s Weekly Binance Spot — Original OHLCV Signal',SYMBOL), ...
    'Theme','Scientific','Colors',viz.Colors,'MAWindows',[13 52 208]);

% CandlestickChart is optimized for generic intervals. For this long weekly
% range, replace its intraday date labels with calendar years.
chartAxes=findall(figCandle,'Type','axes');
for k=1:numel(chartAxes)
    datetick(chartAxes(k),'x','yyyy','keeplimits');
end

%% Reusable selective filtering and spectral audit
filterPipeline=OceanPipeline.Transforms.SelectiveFilterPipeline( ...
    'SamplesPerYear',365.2425/7,'SmoothWindow',13, ...
    'SearchPeriodYears',[2 6],'TargetPeriodYears',4);
analysis=filterPipeline.run(TT.Time,log(TT.Close));
layers=analysis.Layers;
audit=analysis.Audit;

fprintf('\nFour-year spectral audit\n');
fprintf('  Dominant period in predeclared 2-6 year band: %.2f years\n', ...
    audit.EstimatedPeriodYears);
fprintf('  Peak / band-median power: %.2f\n',audit.PeakToBandMedian);
fprintf('  Frequency resolution: %.4f cycles/year\n',audit.FrequencyResolution);
fprintf('  Observed target cycles: %.2f\n',audit.ObservedTargetCycles);

%% Time representation after every removal layer
figTime=viz.createFigure('BTC Four-Year Cycle — Filtered Time Layers');
figTime.Position=[80 40 1400 940];
tlTime=tiledlayout(figTime,4,1,'TileSpacing','compact','Padding','compact');
for k=1:4
    ax=nexttile(tlTime,k);
    colororder(ax,palette(k,:));
    plot(ax,layers(k).Time,layers(k).Signal,'LineWidth',1.0);
    viz.styleAxes(ax); box(ax,'on');
    yline(ax,0,':','Color',viz.Colors.Grid,'HandleVisibility','off');
    title(ax,layers(k).Name,'Color',viz.Colors.Text);
    ylabel(ax,'Filtered log price');
    if k<4, ax.XTickLabel=[]; else, xlabel(ax,'Date'); end
end
title(tlTime,{ ...
    'BTC/USDT Weekly Signal After Each Selective-Filtering Layer', ...
    'Trend, annual, half-year, and sub-quarter components removed sequentially'}, ...
    'Color',viz.Colors.Text,'FontWeight','bold');

%% Frequency representation after every removal layer
figSpectrum=viz.createFigure('BTC Four-Year Cycle — Filtered Spectra');
figSpectrum.Position=[100 60 1400 820];
tlSpectrum=tiledlayout(figSpectrum,2,2,'TileSpacing','compact','Padding','compact');
for k=1:4
    ax=nexttile(tlSpectrum,k);
    colororder(ax,palette(k,:));
    yyaxis(ax,'left');
    plot(ax,layers(k).Frequency,layers(k).Periodogram,'LineWidth',1.05);
    ylabel(ax,'Raw periodogram');
    yyaxis(ax,'right');
    plot(ax,layers(k).Frequency,layers(k).CumulativeGain,'--', ...
        'Color',viz.Colors.Tertiary,'LineWidth',0.9);
    ylabel(ax,'Filter gain');
    viz.styleAxes(ax); box(ax,'on');
    xline(ax,0.25,':','Color',viz.Colors.Text,'LineWidth',1.0, ...
        'HandleVisibility','off');
    xlim(ax,[0 2]);
    title(ax,layers(k).Name,'Color',viz.Colors.Text);
    xlabel(ax,'Frequency (cycles/year)');
end
title(tlSpectrum,{ ...
    'BTC/USDT Weekly Spectrum After Each Filtering Layer', ...
    sprintf(['Dotted reference: 0.25 cycles/year; estimated long period: ' ...
        '%.2f years'],audit.EstimatedPeriodYears)}, ...
    'Color',viz.Colors.Text,'FontWeight','bold');

%% Export
if EXPORT_FIGURES
    exportgraphics(figCandle,fullfile(resultDir,'exp_binance_weekly_candlestick.png'), ...
        'Resolution',EXPORT_DPI);
    exportgraphics(figTime,fullfile(resultDir,'exp_binance_filter_time_layers.png'), ...
        'Resolution',EXPORT_DPI);
    exportgraphics(figSpectrum,fullfile(resultDir,'exp_binance_filter_spectra.png'), ...
        'Resolution',EXPORT_DPI);
end

fprintf(['\nInterpretation guardrail: a peak near four years is descriptive, ' ...
    'not proof of a stable cycle. Binance provides only %.2f observable ' ...
    'four-year cycles in this market.\n'],coverage.FourYearCycles);
