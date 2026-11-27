%% exp_freq_distinguishable.m
%
% Test 3: The N Distinguishable Frequencies
%
% For period N, there are EXACTLY N distinct discrete frequencies.
% At k = N, the pattern is identical to k = 0.
% Layout: 3 rows x 4 cols = k=0..11  (for N=8)
%   Row 1: k=0..3   (blue   — reference zone)
%   Row 2: k=4..7   (purple — approaching / at limit)
%   Row 3: k=8..11  (red    — ALIAS zone: identical to row 1 patterns)
%
% Author : Ocean Library — Signal Processing Experiments
% Date   : 2026-05-19
% -----------------------------------------------------------------------

clc; clear; close all;

%% COLOR SCHEME
BG       = [0.10 0.10 0.14];
GRID_C   = [0.25 0.25 0.30];
TEXT_C   = [0.85 0.85 0.90];
C_REF    = [0.25 0.65 0.95];   % blue   — reference zone (row 1)
C_MID    = [0.75 0.55 0.95];   % purple — mid zone      (row 2)
C_ALIAS  = [1.00 0.38 0.38];   % red    — alias zone    (row 3)
C_GOLD   = [1.00 0.80 0.20];   % highlight / special

AX_REF   = [0.05 0.07 0.14];
AX_MID   = [0.09 0.07 0.13];
AX_ALIAS = [0.16 0.06 0.06];

%% PARAMETERS
N_period = 8;
N_samp   = 32;     % samples to display per subplot
n        = (0:N_samp-1)';

% Show k = 0..11  (N + 3 extra aliases)
k_list = 0:11;     % 12 subplots -> 3 rows x 4 cols

%% FIGURE
fig = figure('Color', BG, 'Position', [30 30 1380 820]);

% ---- Top info bar (replaces in-plot legend) ----------------------------
annotation(fig, 'textbox', [0 0.955 1 0.042], ...
    'String', ['Test 3: N = 8 Distinguishable Frequencies   |   ' ...
               'Blue = unique (k<4)   Purple = limit zone (k=4..7)   ' ...
               'RED = alias zone (k>=8) — patterns repeat from k=0'], ...
    'Color', TEXT_C, 'FontSize', 9.5, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'EdgeColor', 'none', 'BackgroundColor', [0.13 0.13 0.18]);

% ---- Zone separator labels on left margin ------------------------------
zone_labels = {'Reference zone', 'Limit zone', 'ALIAS zone'};
zone_colors = {C_REF, C_MID, C_ALIAS};
zone_y      = [0.935, 0.625, 0.315];  % approximate row centers (normalized)
for z = 1:3
    annotation(fig, 'textbox', [0.002 zone_y(z)-0.045 0.035 0.09], ...
        'String', zone_labels{z}, 'Color', zone_colors{z}, 'FontSize', 8, ...
        'FontWeight', 'bold', 'EdgeColor', 'none', 'BackgroundColor', 'none', ...
        'Rotation', 90, 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle');
end

% ---- Subplots ----------------------------------------------------------
for idx = 1:numel(k_list)
    k        = k_list(idx);
    w0       = 2*pi*k / N_period;
    k_wrap   = mod(k, N_period);   % which unique k this aliases to
    sig      = cos(w0 .* n);

    % Determine zone
    if k < N_period/2
        s_clr = C_REF;  ax_bg = AX_REF;   zone = 'ref';
    elseif k < N_period
        s_clr = C_MID;  ax_bg = AX_MID;   zone = 'mid';
    else
        s_clr = C_ALIAS; ax_bg = AX_ALIAS; zone = 'alias';
    end

    ax = subplot(3, 4, idx);
    set(ax, 'Color', ax_bg, 'XColor', GRID_C, 'YColor', GRID_C, ...
            'GridColor', GRID_C, 'GridAlpha', 0.5, ...
            'XLim', [-0.5 N_samp-0.5], 'YLim', [-1.45 1.45], ...
            'FontSize', 7.5);
    grid(ax, 'on'); box(ax, 'on'); hold(ax, 'on');

    % Zero reference line
    yline(ax, 0, '-', 'Color', [0.40 0.40 0.45], 'LineWidth', 0.7);

    % Discrete stems
    for i = 1:N_samp
        plot(ax, [n(i) n(i)], [0, sig(i)], '-', ...
             'Color', [s_clr 0.50], 'LineWidth', 1.1);
    end
    scatter(ax, n, sig, 28, s_clr, 'filled', ...
            'MarkerEdgeColor', 'w', 'LineWidth', 0.5);

    % ---- Title (no in-plot text banners) ---------------------------------
    if strcmp(zone, 'alias')
        title(ax, sprintf('k = %d  =  k = %d  (ALIAS)', k, k_wrap), ...
              'Color', C_GOLD, 'FontSize', 9, 'FontWeight', 'bold');

    elseif k == 0
        title(ax, 'k = 0   (DC: all samples = +1)', ...
              'Color', C_GOLD, 'FontSize', 9, 'FontWeight', 'bold');

    elseif k == N_period/2
        title(ax, sprintf('k = %d   (pi limit: +1,-1,+1,-1...)', k), ...
              'Color', C_GOLD, 'FontSize', 9, 'FontWeight', 'bold');

    else
        title(ax, sprintf('k = %d   w0 = 2*pi*%d/%d', k, k, N_period), ...
              'Color', TEXT_C, 'FontSize', 9, 'FontWeight', 'bold');
    end

    if idx > 8
        xlabel(ax, 'n', 'Color', TEXT_C, 'FontSize', 8);
    else
        set(ax, 'XTickLabel', []);
    end

    hold(ax, 'off');
end


%% COMMAND WINDOW REPORT
fprintf('\n');
fprintf('======================================================\n');
fprintf('  Test 3: Distinguishable Frequencies  (N = %d)\n', N_period);
fprintf('======================================================\n');
fprintf('  k    w0              Zone       Equivalent to\n');
fprintf('------------------------------------------------------\n');
rows = {'Reference','Reference','Reference','Reference', ...
        'Limit','Limit','Limit','Limit', ...
        'ALIAS','ALIAS','ALIAS','ALIAS'};
for idx = 1:numel(k_list)
    k    = k_list(idx);
    w0   = 2*pi*k/N_period;
    kwrap = mod(k,N_period);
    if k < N_period
        eq = 'unique';
    else
        eq = sprintf('= k = %d', kwrap);
    end
    fprintf('  %-4d  %.4f rad/samp  %-9s  %s\n', k, w0, rows{idx}, eq);
end
fprintf('======================================================\n');
fprintf('Identity: e^(j*(w0+2*pi)*n) = e^(j*w0*n) for all n.\n\n');
