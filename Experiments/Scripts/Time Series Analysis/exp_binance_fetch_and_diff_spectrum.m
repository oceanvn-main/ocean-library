% EXP_BINANCE_FETCH_AND_DIFF_SPECTRUM
% =========================================================================
% Experiment: Windowed Binance Ingestion + Difference-Based Spectrum Filtering
%
% Objectives:
%   1. Fetch BTCUSDT close-price data for four user-selectable windows
%      (1d, 3d, 1w, 1m) using OceanPipeline.Ingestion.BinanceFetcher.
%   2. Compute log-returns from close prices (stationarising transformation).
%   3. Apply 1st-, 2nd-, and 3rd-order difference operators to the log-return
%      series using OceanPipeline.Transforms.DifferenceSpectrumFilter.
%   4. Compare the one-sided power spectral density (Welch) before and after
%      each differencing order; overlay the theoretical |H(f)|^d gain.
%   5. Verify Parseval's theorem for each PSD estimate.
%   6. Report: variance reduction table, Parseval errors, summary stats.
%
% Assumptions:
%   - Live Binance REST data is fetched; script falls back gracefully if the
%     API is unreachable (uses a synthetic Brownian-motion + seasonal signal).
%   - Log-returns are computed causally: r_t = log(C_t / C_{t-1}).
%   - No future information is used at any processing stage.
%   - All spectral estimates use the Welch/Hann window (VerifyParseval=true).
%
% Results location: Experiments/Scripts/Time Series Analysis/Results/
%
% Author: Ocean Library   Date: 2026-10-02

clc; clear; close all;
rng(42, 'twister');

fprintf("==========================================================\n");
fprintf("  BINANCE FETCH + DIFFERENCE SPECTRUM FILTER EXPERIMENT  \n");
fprintf("==========================================================\n\n");

% -------------------------------------------------------------------------
% 0. Configuration
% -------------------------------------------------------------------------
SYMBOL       = "BTCUSDT";
INTERVAL     = "1h";          % Binance kline interval
% Windows to test — run through all four so results are comparable
WINDOWS      = ["1d", "3d", "1w", "1m"];
WINDOW_LABEL = ["1 Day", "3 Days", "1 Week", "1 Month"];

DIFF_ORDERS  = [1, 2, 3];     % Differencing orders to test

RESULT_DIR   = fullfile("Experiments", "Scripts", "Time Series Analysis", "Results");
if ~exist(RESULT_DIR, 'dir'), mkdir(RESULT_DIR); end

% -------------------------------------------------------------------------
% 1. Fetch Data (all four windows in advance; keep longest for spectrum work)
% -------------------------------------------------------------------------
fprintf("[Step 1] Fetching Binance OHLCV data for all windows...\n\n");

TT_all = cell(numel(WINDOWS), 1);

for wi = 1:numel(WINDOWS)
    window = WINDOWS(wi);
    fetcher = OceanPipeline.Ingestion.BinanceFetcher( ...
        'Symbol',     SYMBOL, ...
        'Window',     window, ...
        'Interval',   INTERVAL, ...
        'MarketType', 'spot', ...
        'Verbose',    true);

    try
        TT_all{wi} = fetcher.fetch();
        fprintf("   [OK] Window=%s : %d candles | %s  -->  %s  UTC\n\n", ...
            window, height(TT_all{wi}), ...
            char(TT_all{wi}.Time(1)), char(TT_all{wi}.Time(end)));
    catch ME
        fprintf("   [NOTICE] Live fetch failed for window=%s: %s\n", window, ME.message);
        fprintf("            Using synthetic fallback data.\n\n");
        TT_all{wi} = generateFallbackData(window, INTERVAL);
    end
end

% -------------------------------------------------------------------------
% 2. Log-Return Computation (applied per window)
% -------------------------------------------------------------------------
fprintf("[Step 2] Computing log-returns (causal, no look-ahead)...\n");

logRet_all = cell(numel(WINDOWS), 1);
fs_Hz      = 1 / 3600;        % 1 sample per hour => fs = 1/3600 Hz

for wi = 1:numel(WINDOWS)
    TT = TT_all{wi};
    closePrice = TT.Close;
    % r_t = log(C_t) - log(C_{t-1}) — first observation dropped (no previous)
    logRet = diff(log(closePrice));
    logRet_all{wi} = logRet;

    fprintf("   Window=%-4s | N_candles=%4d | N_returns=%4d | " + ...
            "mean=%.5f | std=%.5f\n", ...
        WINDOWS(wi), numel(closePrice), numel(logRet), ...
        mean(logRet), std(logRet));
end
fprintf("\n");

% -------------------------------------------------------------------------
% 3. Difference Spectrum Filtering (on the 1-week window as primary series)
% -------------------------------------------------------------------------
fprintf("[Step 3] Applying difference spectrum filter (Orders 1,2,3) on 1-week returns...\n\n");

% Use 1-week data as the primary case for detailed spectrum analysis
idx_primary = find(WINDOWS == "1w", 1);
x_primary   = logRet_all{idx_primary};

dsf_reports = cell(numel(DIFF_ORDERS), 1);

for oi = 1:numel(DIFF_ORDERS)
    d = DIFF_ORDERS(oi);
    dsf = OceanPipeline.Transforms.DifferenceSpectrumFilter( ...
        'Order',          d, ...
        'Overlap',        0.5, ...
        'WindowType',     'hann', ...
        'VerifyParseval', true);

    report = dsf.filter(x_primary, fs_Hz);
    dsf_reports{oi} = report;

    fprintf("  Order %d:\n", d);
    fprintf("    %s\n\n", report.summary);
end

% -------------------------------------------------------------------------
% 4. Repeat spectrum filter on ALL windows (order=1 only) for comparison
% -------------------------------------------------------------------------
fprintf("[Step 4] First-difference spectrum on all windows (order=1)...\n\n");

all_window_reports = cell(numel(WINDOWS), 1);

for wi = 1:numel(WINDOWS)
    x = logRet_all{wi};
    if numel(x) < 8
        fprintf("   Window=%-4s  [SKIP] too few samples (%d).\n", WINDOWS(wi), numel(x));
        continue;
    end

    dsf = OceanPipeline.Transforms.DifferenceSpectrumFilter( ...
        'Order',          1, ...
        'Overlap',        0.5, ...
        'WindowType',     'hann', ...
        'VerifyParseval', true);

    all_window_reports{wi} = dsf.filter(x, fs_Hz);

    r = all_window_reports{wi};
    fprintf("   Window=%-6s | N_orig=%4d | var_orig=%.4e | " + ...
            "var_1diff=%.4e | reduction=%.1f dB | " + ...
            "Parseval_orig=%.2f%%\n", ...
        WINDOWS(wi), numel(r.x_orig), r.var_orig, r.var_diff, ...
        10*log10(r.var_diff / (r.var_orig + eps)), ...
        r.parseval_err_orig);
end
fprintf("\n");

% -------------------------------------------------------------------------
% 5. Verification Table (Parseval + Variance Reduction)
% -------------------------------------------------------------------------
fprintf("[Step 5] Verification table (Parseval + variance reduction per order)...\n\n");

fprintf("  %-8s | %-12s | %-12s | %-12s | %-15s | %-15s\n", ...
    "Order", "N_orig", "N_diff", "Var_orig", "Var_diff", "Parseval_err(diff)");
fprintf("  %s\n", repmat('-', 1, 85));

for oi = 1:numel(DIFF_ORDERS)
    r = dsf_reports{oi};
    fprintf("  %-8d | %-12d | %-12d | %-12.4e | %-15.4e | %-15.3f%%\n", ...
        r.order, numel(r.x_orig), numel(r.x_diff), ...
        r.var_orig, r.var_diff, r.parseval_err_diff);
end
fprintf("\n");

% -------------------------------------------------------------------------
% 6. Visualization  (OceanViz.Core — Dark Theme)
% -------------------------------------------------------------------------
fprintf("[Step 6] Generating diagnostic plots...\n");

try
    viz = OceanViz.Core('Theme', 'Dark');
catch
    viz = [];   % graceful degradation if OceanViz unavailable
end

% ---- Figure 1: All-window close-price + log-return overview ---------------
fig1 = figure('Name', 'Binance Multi-Window Overview', ...
              'Color', [0.059 0.067 0.090], ...
              'Position', [50 50 1400 800]);

nW = numel(WINDOWS);
for wi = 1:nW
    TT = TT_all{wi};
    lr = logRet_all{wi};

    % Close price
    ax = subplot(nW, 2, (wi-1)*2 + 1);
    plot(TT.Time, TT.Close, 'Color', '#7c83fd', 'LineWidth', 1.2);
    title(sprintf('%s BTCUSDT Close — %s', INTERVAL, WINDOW_LABEL(wi)), ...
        'Color', '#e8eaf6', 'FontSize', 9);
    ylabel('Price (USDT)', 'Color', '#a0a8c0', 'FontSize', 8);
    set(ax, 'Color', [0.102 0.114 0.153], 'XColor', '#a0a8c0', 'YColor', '#a0a8c0', ...
        'GridColor', [0.165 0.180 0.235], 'XGrid', 'on', 'YGrid', 'on');
    legend(sprintf('N=%d candles', height(TT)), 'TextColor', '#e8eaf6', ...
        'Color', [0.102 0.114 0.153], 'EdgeColor', 'none', 'FontSize', 7);

    % Log-return
    ax2 = subplot(nW, 2, (wi-1)*2 + 2);
    plot(TT.Time(2:end), lr, 'Color', '#7cfd9a', 'LineWidth', 0.8);
    title(sprintf('Log-Return — %s', WINDOW_LABEL(wi)), ...
        'Color', '#e8eaf6', 'FontSize', 9);
    ylabel('Log-Return', 'Color', '#a0a8c0', 'FontSize', 8);
    set(ax2, 'Color', [0.102 0.114 0.153], 'XColor', '#a0a8c0', 'YColor', '#a0a8c0', ...
        'GridColor', [0.165 0.180 0.235], 'XGrid', 'on', 'YGrid', 'on');
end

sgtitle(sprintf('Binance %s BTCUSDT — %d Windows Overview', INTERVAL, nW), ...
    'Color', '#e8eaf6', 'FontSize', 13, 'FontWeight', 'bold');

exportgraphics(fig1, fullfile(RESULT_DIR, 'exp_diff_spectrum_window_overview.png'), ...
    'Resolution', 150);
fprintf("   Saved: exp_diff_spectrum_window_overview.png\n");

% ---- Figure 2: Spectrum comparison per differencing order (1-week) --------
fig2 = figure('Name', 'Difference Spectrum Filter (1-Week)', ...
              'Color', [0.059 0.067 0.090], ...
              'Position', [80 80 1500 900]);

COLORS = {'#7c83fd', '#fd7c7c', '#7cfd9a', '#fdd97c'};
ORDER_NAMES = {'Original', '1st-order diff', '2nd-order diff', '3rd-order diff'};

nOrders = numel(DIFF_ORDERS);

for oi = 1:nOrders
    r = dsf_reports{oi};

    % --- Left column: time-domain original vs differenced ----------------
    ax_t = subplot(nOrders, 3, (oi-1)*3 + 1);
    hold on;
    plot(r.x_orig, 'Color', COLORS{1}, 'LineWidth', 0.8, 'DisplayName', 'Original');
    x_padded = [NaN(r.order, 1); r.x_diff];   % align time-axis with original
    plot(x_padded, 'Color', COLORS{oi+1}, 'LineWidth', 0.8, ...
        'DisplayName', sprintf('Order %d diff', r.order));
    hold off;
    title(sprintf('Time Domain — Order %d', r.order), ...
        'Color', '#e8eaf6', 'FontSize', 9);
    ylabel('Amplitude', 'Color', '#a0a8c0', 'FontSize', 8);
    legend('FontSize', 7, 'TextColor', '#e8eaf6', 'Color', [0.102 0.114 0.153], ...
        'EdgeColor', 'none');
    set(ax_t, 'Color', [0.102 0.114 0.153], 'XColor', '#a0a8c0', 'YColor', '#a0a8c0', ...
        'GridColor', [0.165 0.180 0.235], 'XGrid', 'on', 'YGrid', 'on');

    % --- Middle column: PSD comparison (dB scale) ------------------------
    ax_p = subplot(nOrders, 3, (oi-1)*3 + 2);
    hold on;
    semilogy(r.f_orig, r.psd_orig, 'Color', COLORS{1}, 'LineWidth', 1.2, ...
        'DisplayName', 'PSD (original)');
    semilogy(r.f_diff, r.psd_diff, 'Color', COLORS{oi+1}, 'LineWidth', 1.2, ...
        'DisplayName', sprintf('PSD (order %d diff)', r.order));
    hold off;
    title(sprintf('Welch PSD Comparison — Order %d', r.order), ...
        'Color', '#e8eaf6', 'FontSize', 9);
    xlabel('Frequency (Hz)', 'Color', '#a0a8c0', 'FontSize', 8);
    ylabel('PSD (V^2/Hz)', 'Color', '#a0a8c0', 'FontSize', 8);
    legend('FontSize', 7, 'TextColor', '#e8eaf6', 'Color', [0.102 0.114 0.153], ...
        'EdgeColor', 'none');
    set(ax_p, 'Color', [0.102 0.114 0.153], 'XColor', '#a0a8c0', 'YColor', '#a0a8c0', ...
        'GridColor', [0.165 0.180 0.235], 'XGrid', 'on', 'YGrid', 'on');

    % --- Right column: Theoretical |H(f)|^d overlay ----------------------
    ax_h = subplot(nOrders, 3, (oi-1)*3 + 3);
    hold on;
    yyaxis left;
    semilogy(r.f_orig, r.psd_orig, 'Color', COLORS{1}, 'LineWidth', 1.0, ...
        'DisplayName', 'PSD (original)');
    set(ax_h, 'YColor', COLORS{1});
    ylabel('PSD (V^2/Hz)', 'Color', '#a0a8c0', 'FontSize', 8);

    yyaxis right;
    plot(r.f_orig, r.tf_gain, 'Color', '#fdd97c', 'LineWidth', 1.2, ...
        'LineStyle', '--', 'DisplayName', sprintf('|H(f)|^{%d}', r.order));
    set(ax_h, 'YColor', '#fdd97c');
    ylabel(sprintf('|H(f)|^{%d}', r.order), 'Color', '#fdd97c', 'FontSize', 8);
    hold off;

    title(sprintf('Gain |H(f)|^{%d} Overlay — Order %d', r.order, r.order), ...
        'Color', '#e8eaf6', 'FontSize', 9);
    xlabel('Frequency (Hz)', 'Color', '#a0a8c0', 'FontSize', 8);
    legend('FontSize', 7, 'TextColor', '#e8eaf6', 'Color', [0.102 0.114 0.153], ...
        'EdgeColor', 'none');
    set(ax_h, 'Color', [0.102 0.114 0.153], 'XColor', '#a0a8c0', ...
        'GridColor', [0.165 0.180 0.235], 'XGrid', 'on', 'YGrid', 'on');
end

sgtitle(sprintf('Difference Spectrum Filter — BTCUSDT 1w Log-Returns (fs=%.4g Hz)', fs_Hz), ...
    'Color', '#e8eaf6', 'FontSize', 13, 'FontWeight', 'bold');

exportgraphics(fig2, fullfile(RESULT_DIR, 'exp_diff_spectrum_orders.png'), ...
    'Resolution', 150);
fprintf("   Saved: exp_diff_spectrum_orders.png\n");

% ---- Figure 3: All-window first-difference spectrum overlay ---------------
fig3 = figure('Name', 'First-Difference PSD — All Windows', ...
              'Color', [0.059 0.067 0.090], ...
              'Position', [100 100 1100 600]);

ax3a = subplot(1, 2, 1);
hold on;
WIN_COLORS = {'#7c83fd', '#fd7c7c', '#7cfd9a', '#fdd97c'};
for wi = 1:nW
    r = all_window_reports{wi};
    if isempty(r), continue; end
    semilogy(r.f_orig, r.psd_orig, 'Color', WIN_COLORS{wi}, 'LineWidth', 1.2, ...
        'DisplayName', sprintf('%s original', WINDOW_LABEL(wi)));
end
hold off;
title('PSD Before Differencing', 'Color', '#e8eaf6', 'FontSize', 10);
xlabel('Frequency (Hz)', 'Color', '#a0a8c0', 'FontSize', 8);
ylabel('PSD (V^2/Hz)', 'Color', '#a0a8c0', 'FontSize', 8);
legend('FontSize', 7, 'TextColor', '#e8eaf6', 'Color', [0.102 0.114 0.153], 'EdgeColor', 'none');
set(ax3a, 'Color', [0.102 0.114 0.153], 'XColor', '#a0a8c0', 'YColor', '#a0a8c0', ...
    'GridColor', [0.165 0.180 0.235], 'XGrid', 'on', 'YGrid', 'on');

ax3b = subplot(1, 2, 2);
hold on;
for wi = 1:nW
    r = all_window_reports{wi};
    if isempty(r), continue; end
    semilogy(r.f_diff, r.psd_diff, 'Color', WIN_COLORS{wi}, 'LineWidth', 1.2, ...
        'DisplayName', sprintf('%s 1st-diff', WINDOW_LABEL(wi)));
end
hold off;
title('PSD After 1st-Order Differencing', 'Color', '#e8eaf6', 'FontSize', 10);
xlabel('Frequency (Hz)', 'Color', '#a0a8c0', 'FontSize', 8);
ylabel('PSD (V^2/Hz)', 'Color', '#a0a8c0', 'FontSize', 8);
legend('FontSize', 7, 'TextColor', '#e8eaf6', 'Color', [0.102 0.114 0.153], 'EdgeColor', 'none');
set(ax3b, 'Color', [0.102 0.114 0.153], 'XColor', '#a0a8c0', 'YColor', '#a0a8c0', ...
    'GridColor', [0.165 0.180 0.235], 'XGrid', 'on', 'YGrid', 'on');

sgtitle('Welch PSD: All Windows — Before vs After 1st-Difference', ...
    'Color', '#e8eaf6', 'FontSize', 13, 'FontWeight', 'bold');

exportgraphics(fig3, fullfile(RESULT_DIR, 'exp_diff_spectrum_windows.png'), ...
    'Resolution', 150);
fprintf("   Saved: exp_diff_spectrum_windows.png\n\n");

% -------------------------------------------------------------------------
% 7. Final Report
% -------------------------------------------------------------------------
fprintf("==========================================================\n");
fprintf("  EXPERIMENT RESULTS SUMMARY\n");
fprintf("==========================================================\n\n");

fprintf("  [1] DATA FETCH SUMMARY\n");
fprintf("  %-8s | %-12s | %-24s | %-24s\n", ...
    "Window", "N candles", "Start (UTC)", "End (UTC)");
fprintf("  %s\n", repmat('-', 1, 75));
for wi = 1:nW
    TT = TT_all{wi};
    if ~isempty(TT) && height(TT) > 0
        fprintf("  %-8s | %-12d | %-24s | %-24s\n", ...
            WINDOWS(wi), height(TT), char(TT.Time(1)), char(TT.Time(end)));
    else
        fprintf("  %-8s | %-12s | %-24s | %-24s\n", ...
            WINDOWS(wi), "N/A", "N/A", "N/A");
    end
end

fprintf("\n  [2] DIFFERENCE FILTER VERIFICATION (1-week log-returns, fs=%.4g Hz)\n", fs_Hz);
fprintf("  %-8s | %-10s | %-10s | %-14s | %-14s | %-10s | %-10s\n", ...
    "Order", "N_orig", "N_diff", "Var_orig", "Var_diff", "dB_reduc", "Parseval%");
fprintf("  %s\n", repmat('-', 1, 85));
for oi = 1:numel(DIFF_ORDERS)
    r = dsf_reports{oi};
    dB = 10*log10(r.var_diff / (r.var_orig + eps));
    fprintf("  %-8d | %-10d | %-10d | %-14.4e | %-14.4e | %-10.2f | %-10.3f%%\n", ...
        r.order, numel(r.x_orig), numel(r.x_diff), ...
        r.var_orig, r.var_diff, dB, r.parseval_err_diff);
end

fprintf("\n  [3] PARSEVAL VERIFICATION\n");
fprintf("  All Parseval errors within acceptable threshold (< 1%%): ");
allOK = true;
for oi = 1:numel(DIFF_ORDERS)
    r = dsf_reports{oi};
    if r.parseval_err_orig > 1.0 || r.parseval_err_diff > 1.0
        allOK = false;
    end
end
if allOK
    fprintf("PASS\n");
else
    fprintf("SOME WARNINGS (check individual reports above)\n");
end

fprintf("\n  [4] PLOT FILES\n");
fprintf("  -> exp_diff_spectrum_window_overview.png\n");
fprintf("  -> exp_diff_spectrum_orders.png\n");
fprintf("  -> exp_diff_spectrum_windows.png\n");
fprintf("\n==========================================================\n");
fprintf("  EXPERIMENT COMPLETE\n");
fprintf("==========================================================\n");

% =========================================================================
% LOCAL HELPER FUNCTIONS
% =========================================================================

function TT = generateFallbackData(window, interval)
% GENERATEFALLBACKDATA  Synthetic OHLCV timetable for offline testing.
%
%   Generates a Geometric Brownian Motion price series with an embedded
%   seasonal cycle so that spectral content is non-trivial.
%
%   This function is ONLY called when the live Binance REST API fails.
%   It does NOT encode any real market data or known outcomes.

    % Parse window size
    windowDays = struct('1d', 1, '3d', 3, '1w', 7, '1m', 30);
    w_lower = lower(window);
    if isfield(windowDays, w_lower)
        nDays = windowDays.(w_lower);
    else
        nDays = 7;   % default fallback
    end

    % Interval -> step size (hours)
    intStepHours = struct('x1m', 1/60, 'x5m', 5/60, 'x15m', 0.25, ...
                          'x30m', 0.5, 'x1h', 1, 'x4h', 4, 'x1d', 24);
    ikey = ['x' lower(interval)];
    if isfield(intStepHours, ikey)
        stepH = intStepHours.(ikey);
    else
        stepH = 1;
    end

    N = floor(nDays * 24 / stepH);
    N = max(N, 20);   % minimum sensible length

    % Synthetic GBM with daily seasonal cycle
    mu    = 0.0002;    % drift per step
    sigma = 0.012;     % vol per step
    S0    = 60000;     % starting price (USD)

    t = (0:N-1)' * stepH;               % hours
    season = 0.002 * sin(2*pi * t / 24); % daily cycle

    % Log-price path
    logReturns = mu + sigma * randn(N, 1) + season;
    logPrice   = cumsum([log(S0); logReturns]);
    price      = exp(logPrice(1:N));

    % OHLCV columns
    Open   = price;
    Close  = price .* (1 + 0.002*randn(N,1));
    High   = max(Open, Close) .* (1 + abs(0.003*randn(N,1)));
    Low    = min(Open, Close) .* (1 - abs(0.003*randn(N,1)));
    Volume = abs(500 + 100*randn(N,1));

    tEnd   = datetime('now', 'TimeZone', 'UTC');
    tStart = tEnd - days(nDays);
    dt     = linspace(posixtime(tStart), posixtime(tEnd), N)';
    timestamps = datetime(dt, 'ConvertFrom', 'posixtime', 'TimeZone', 'UTC');

    TT = timetable(timestamps, Open, High, Low, Close, Volume, ...
        'VariableNames', {'Open', 'High', 'Low', 'Close', 'Volume'});
    TT.Properties.DimensionNames{1} = 'Time';
end
