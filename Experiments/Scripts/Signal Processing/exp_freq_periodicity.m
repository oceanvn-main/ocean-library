%% exp_freq_periodicity.m
%
% Test 2: The Periodicity Failure
%
% Compares a periodic discrete signal (w0 = pi/4, rational w0/2pi = 1/8)
% against the classic textbook aperiodic case (w0 = 1, irrational w0/2pi).
%
% Proof panel: |x[n] - x[0]| must return to EXACTLY 0 at period N.
% The aperiodic case NEVER touches 0 for any integer n > 0.
%
% Author : Ocean Library — Signal Processing Experiments
% Date   : 2026-05-19
% -----------------------------------------------------------------------

clc; clear; close all;

%% COLOR SCHEME
BG     = [0.10 0.10 0.14];
AX_BG  = [0.07 0.07 0.11];
GRID_C = [0.25 0.25 0.30];
TEXT_C = [0.85 0.85 0.90];
C_CONT = [0.30 0.85 0.50];
C_PER  = [0.25 0.65 0.95];
C_NPER = [1.00 0.40 0.40];
C_GOLD = [1.00 0.80 0.20];

%% PARAMETERS
N_show   = 200;
n        = (0:N_show-1)';
t_cont   = linspace(0, N_show-1, 4000)';
w0_per   = pi/4;
period_N = 8;
w0_fail  = 1;
x_per    = cos(w0_per  .* n);
x_fail   = cos(w0_fail .* n);

%% FIGURE 1: Three-panel comparison
fig1 = figure('Color', BG, 'Position', [50 50 1200 580]);

%% Panel 1
ax1 = subplot(2,1,1);
setupAx(ax1, AX_BG, GRID_C, [-1 N_show], [-1.4 1.5]);
hold(ax1,'on');
plot(ax1, t_cont, cos(w0_per.*t_cont), '-', 'Color',[C_CONT 0.35], 'LineWidth',1.0);
for k=1:N_show
    plot(ax1,[n(k) n(k)],[0 x_per(k)],'-','Color',[C_PER 0.55],'LineWidth',0.8);
end
scatter(ax1, n, x_per, 14, C_PER, 'filled');
for m=0:floor(N_show/period_N)
    xline(ax1, m*period_N,'--','Color',C_GOLD,'LineWidth',1.2,'Alpha',0.8);
end
title(ax1, ...
    sprintf('cos(pi/4 n)     w0/(2*pi) = 1/8     N = %d',period_N), ...
    'Color',C_PER,'FontSize',10,'FontWeight','bold');
yLabel = sprintf('x[n]');
ylabel(ax1, yLabel,'Color',TEXT_C,'FontSize',9,'FontWeight','bold');
hold(ax1,'off');

%% Panel 2
ax2 = subplot(2,1,2);
setupAx(ax2,[0.10 0.07 0.07],GRID_C,[-1 N_show],[-1.4 1.5]);
hold(ax2,'on');
plot(ax2, t_cont, cos(w0_fail.*t_cont),'-','Color',[C_CONT 0.35],'LineWidth',1.0);
for k=1:N_show
    plot(ax2,[n(k) n(k)],[0 x_fail(k)],'-','Color',[C_NPER 0.55],'LineWidth',0.8);
end
scatter(ax2, n, x_fail, 14, C_NPER, 'filled');
yline(ax2, 1,'--','Color',C_GOLD,'LineWidth',1.3,'Alpha',0.85);
text(ax2, N_show*0.78, 1.10, 'y = 1  (x[0])','Color',C_GOLD,'FontSize',8.5,'FontWeight','bold');
title(ax2,'cos(n)     w0/(2*pi) = 1/(2*pi) = 0.15915...', ...
    'Color',C_NPER,'FontSize',10,'FontWeight','bold');
ylabel(ax2,'x[n]','Color',TEXT_C,'FontSize',9,'FontWeight','bold');
hold(ax2,'off');


sgtitle(fig1, 'Test 2: Periodicity   |   w0/(2*pi) must be rational', ...
    'Color',TEXT_C,'FontSize',13,'FontWeight','bold');

%% FIGURE 2: Overview + Zoom-in evidence
% Row 1: full signal view with star markers at n = 0, 8, 16, 24
% Rows 2-4: zoomed windows at each candidate period (n=8, 16, 24)
%   Left  = cos(pi/4 n)   Right = cos(n)
% The dot position relative to the gold y=1 line IS the evidence.

check_n  = [8, 16, 24];         % candidate periods to zoom into
zoom_hw  = 4;                   % half-window: show n ± 4 samples
n_over   = (0:3*period_N)';     % overview: first 3 periods
t_over   = linspace(0, 3*period_N, 600)';

fig2 = figure('Color', BG, 'Position', [40 40 1200 860]);

signals = {w0_per, C_PER, AX_BG; w0_fail, C_NPER, [0.10 0.07 0.07]};
sig_labels = {'cos(pi/4 n)   w0 = pi/4', 'cos(n)   w0 = 1 rad/samp'};

%% Row 1: Overview
for p = 1:2
    w0p = signals{p,1};  clr = signals{p,2};  bg = signals{p,3};
    sig = cos(w0p .* n_over);

    ax = subplot(4, 2, p);
    setupAx(ax, bg, GRID_C, [-0.5 3*period_N+0.5], [-1.50 1.65]);
    hold(ax, 'on');

    plot(ax, t_over, cos(w0p.*t_over), '-', 'Color',[C_CONT 0.35], 'LineWidth',1.1);
    for i = 1:numel(n_over)
        plot(ax,[n_over(i) n_over(i)],[0 sig(i)],'-','Color',[clr 0.35],'LineWidth',0.9);
    end
    scatter(ax, n_over, sig, 16, clr, 'filled');

    % Gold y=1 reference
    yline(ax, 1, '-', 'Color', C_GOLD, 'LineWidth', 1.5, 'Alpha', 0.8);

    % Star markers at check points + value readout
    for ni = [0, check_n]
        vi = cos(w0p * ni);
        scatter(ax, ni, vi, 120, C_GOLD, 'p', 'filled', ...
                'MarkerEdgeColor','w', 'LineWidth', 0.6);
        text(ax, ni, vi+0.16, sprintf('%.3f', vi), ...
             'Color', C_GOLD, 'FontSize', 7.5, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'center');
    end

    title(ax, sig_labels{p}, 'Color', clr, 'FontSize', 10, 'FontWeight', 'bold');
    ylabel(ax, 'x[n]', 'Color', TEXT_C, 'FontSize', 8);
    set(ax, 'XTickLabel', []);
    hold(ax, 'off');
end

%% Rows 2-4: Zoom-in at each check_n
for row = 1:numel(check_n)
    n_check = check_n(row);
    n_zoom  = (n_check - zoom_hw : n_check + zoom_hw)';
    t_zoom  = linspace(n_check - zoom_hw, n_check + zoom_hw, 400)';

    for p = 1:2
        w0p = signals{p,1};  clr = signals{p,2};  bg = signals{p,3};
        sig_z = cos(w0p .* n_zoom);
        v_at_check = cos(w0p * n_check);

        ax = subplot(4, 2, (row)*2 + p);
        setupAx(ax, bg, GRID_C, [n_check-zoom_hw-0.3, n_check+zoom_hw+0.3], [-1.45 1.45]);
        hold(ax, 'on');

        % Continuous curve
        plot(ax, t_zoom, cos(w0p.*t_zoom), '-', 'Color',[C_CONT 0.40], 'LineWidth',1.2);

        % Discrete stems (all samples in window)
        for i = 1:numel(n_zoom)
            plot(ax, [n_zoom(i) n_zoom(i)], [0, sig_z(i)], '-', ...
                 'Color', [clr 0.45], 'LineWidth', 1.0);
        end
        scatter(ax, n_zoom, sig_z, 30, clr, 'filled');

        % Gold y=1 reference line
        yline(ax, 1, '-', 'Color', C_GOLD, 'LineWidth', 2.2, 'Alpha', 0.95);

        % Large highlighted dot at exactly n_check
        scatter(ax, n_check, v_at_check, 200, C_GOLD, 'p', 'filled', ...
                'MarkerEdgeColor', 'w', 'LineWidth', 1.0);

        % Vertical drop-line showing the gap (computed from data, no label)
        if abs(v_at_check - 1) > 1e-10
            plot(ax, [n_check, n_check], [v_at_check, 1], '-', ...
                 'Color', [C_NPER 0.85], 'LineWidth', 2.5);
            % Gap value label — data only, no verdict
            text(ax, n_check + 0.3, (v_at_check+1)/2, ...
                 sprintf('%.4f', 1 - v_at_check), ...
                 'Color', C_NPER, 'FontSize', 9, 'FontWeight', 'bold');
        end

        % x[n_check] readout inside plot
        text(ax, n_check - zoom_hw + 0.2, -1.28, ...
             sprintf('x[%d] = %.6f', n_check, v_at_check), ...
             'Color', TEXT_C, 'FontSize', 9, 'FontWeight', 'bold', ...
             'BackgroundColor', bg);

        title(ax, sprintf('Zoom: n = %d', n_check), ...
              'Color', TEXT_C, 'FontSize', 9, 'FontWeight', 'bold');

        if row == numel(check_n)
            xlabel(ax, 'n', 'Color', TEXT_C, 'FontSize', 8);
        else
            set(ax, 'XTickLabel', []);
        end
        ylabel(ax, 'x[n]', 'Color', TEXT_C, 'FontSize', 8);
        hold(ax, 'off');
    end
end

sgtitle(fig2, ...
    sprintf('x[n] at n = 0, 8, 16, 24   |   Gold line = y = 1   |   Gold star = x[n_check]'), ...
    'Color', TEXT_C, 'FontSize', 11, 'FontWeight', 'bold');

%% Print numerical table
fprintf('\nNumerical check at candidate period positions:\n');
fprintf('%-6s  %-20s  %-20s\n', 'n', 'cos(pi/4*n)', 'cos(n)');
fprintf('%s\n', repmat('-',1,50));
for ni = [0, check_n]
    fprintf('%-6d  %-20.10f  %-20.10f\n', ni, cos(w0_per*ni), cos(w0_fail*ni));
end

%% HELPER
function setupAx(ax, bg, gc, xlim, ylim)
    set(ax,'Color',bg,'XColor',gc,'YColor',gc,'GridColor',gc,'GridAlpha',0.5, ...
           'XLim',xlim,'YLim',ylim);
    grid(ax,'on'); box(ax,'on');
end
