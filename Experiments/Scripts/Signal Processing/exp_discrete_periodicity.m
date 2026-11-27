%% exp_discrete_periodicity.m
%
% Experiment: Periodicity in Discrete-Time vs Continuous-Time Sinusoids
%
% THEORY:
%   Continuous time:   x(t) = cos(w0*t)  is ALWAYS periodic, period T = 2*pi/w0
%
%   Discrete time:     x[n] = cos(w0*n)  is periodic with period N ONLY IF:
%
%       w0 * N = 2*pi*k   for some integers N and k
%   =>  w0 / (2*pi) = k/N   must be a RATIONAL number
%
%   If w0/(2*pi) is IRRATIONAL, no such integer N exists -> NOT periodic.
%
%   Famous failure case (from Oppenheim & Schafer):
%       x3[n] = cos(n)   where w0 = 1 rad/sample
%       1/(2*pi) is irrational -> x3[n] is NOT periodic in discrete time.
%
% Author : Ocean Library — Signal Processing Experiments
% Date   : 2026-05-19
% -----------------------------------------------------------------------

clc; clear; close all;

%% -----------------------------------------------------------------------
%  COLOR SCHEME
% -----------------------------------------------------------------------
BG      = [0.10 0.10 0.14];
AX_BG   = [0.07 0.07 0.11];
GRID_C  = [0.25 0.25 0.30];
TEXT_C  = [0.85 0.85 0.90];
C_CONT  = [0.30 0.85 0.50];   % continuous curve — green
C_PER   = [0.25 0.65 0.95];   % periodic discrete — blue
C_NPER  = [1.00 0.42 0.42];   % non-periodic discrete — red
C_ENV   = [1.00 0.80 0.20];   % period marker — gold

%% -----------------------------------------------------------------------
%  CASES DEFINITION
% -----------------------------------------------------------------------
% Each case:  { label, w0, is_rational, period_N (0 if not periodic) }

cases = {
    'w_0 = pi/4  (rational: 1/8)',       pi/4,         true,  8;
    'w_0 = pi/6  (rational: 1/12)',      pi/6,         true,  12;
    'w_0 = 2*pi/7  (rational: 1/7)',     2*pi/7,       true,  7;
    'w_0 = 1 rad/samp  (irrational)',    1,            false, 0;
    'w_0 = sqrt(2)  (irrational)',       sqrt(2),      false, 0;
    'w_0 = pi/sqrt(2)  (irrational)',    pi/sqrt(2),   false, 0;
};

N_show    = 80;    % discrete samples to display
n         = (0:N_show-1)';
t_cont    = linspace(0, N_show-1, 2000)';  % fine grid for continuous curve

%% -----------------------------------------------------------------------
%  FIGURE 1: Side-by-side — Periodic vs Non-Periodic (overview)
% -----------------------------------------------------------------------
fig1 = figure('Color', BG, 'Position', [50 50 1300 900]);

n_periodic     = 3;   % first 3 cases
n_nonperiodic  = 3;   % last 3 cases

for i = 1:6
    label    = cases{i,1};
    w0       = cases{i,2};
    is_rat   = cases{i,3};
    period_N = cases{i,4};

    x_disc = cos(w0 .* n);
    x_cont = cos(w0 .* t_cont);

    if i <= 3
        col_idx = i;
        ax = subplot(3, 2, (col_idx-1)*2 + 1);
    else
        col_idx = i - 3;
        ax = subplot(3, 2, (col_idx-1)*2 + 2);
    end

    styleAxes(ax, AX_BG, GRID_C);
    hold(ax, 'on');

    % Continuous overlay (always periodic — reference)
    plot(ax, t_cont, x_cont, '-', 'Color', [C_CONT 0.35], 'LineWidth', 1.0);

    % Discrete stems
    if is_rat
        dot_color = C_PER;
    else
        dot_color = C_NPER;
    end

    for k = 1:N_show
        plot(ax, [n(k) n(k)], [0, x_disc(k)], '-', ...
             'Color', [dot_color 0.6], 'LineWidth', 0.9);
    end
    scatter(ax, n, x_disc, 16, dot_color, 'filled');

    % Mark period boundaries for rational cases
    if is_rat && period_N > 0
        for m = 0 : floor(N_show / period_N)
            xline(ax, m * period_N, '--', 'Color', C_ENV, 'LineWidth', 1.0, ...
                  'Alpha', 0.7);
        end
        title_str = sprintf('%s\nPeriodic:  N = %d', label, period_N);
        title(ax, title_str, 'Color', C_PER, 'FontSize', 8.5, 'FontWeight', 'bold');
    else
        title_str = sprintf('%s\nNOT periodic in discrete time', label);
        title(ax, title_str, 'Color', C_NPER, 'FontSize', 8.5, 'FontWeight', 'bold');
    end

    xlabel(ax, 'n', 'Color', TEXT_C, 'FontSize', 8);
    ylabel(ax, 'x[n]', 'Color', TEXT_C, 'FontSize', 8);
    set(ax, 'XLim', [-1 N_show], 'YLim', [-1.35 1.35]);
    hold(ax, 'off');
end

% Column headers
annotation(fig1, 'textbox', [0.02 0.95 0.45 0.04], ...
    'String', 'PERIODIC Cases  (w0/2pi is rational)', ...
    'Color', C_PER, 'FontSize', 11, 'FontWeight', 'bold', ...
    'EdgeColor', 'none', 'BackgroundColor', 'none');
annotation(fig1, 'textbox', [0.52 0.95 0.46 0.04], ...
    'String', 'NON-PERIODIC Cases  (w0/2pi is irrational)', ...
    'Color', C_NPER, 'FontSize', 11, 'FontWeight', 'bold', ...
    'EdgeColor', 'none', 'BackgroundColor', 'none');

sgtitle(fig1, 'Discrete-Time Periodicity Condition: w0/2pi must be rational', ...
    'Color', TEXT_C, 'FontSize', 13, 'FontWeight', 'bold');

%% -----------------------------------------------------------------------
%  FIGURE 2: The Textbook Case  — cos(n)  zoomed in
%  Demonstrates there is NO repeat: check x[0] vs x[k] for all k
% -----------------------------------------------------------------------
fig2 = figure('Color', BG, 'Position', [100 100 1100 700]);

w0_fail = 1;   % w0 = 1 rad/sample
N_check = 200;
n_check = (0:N_check-1)';
x_fail  = cos(w0_fail .* n_check);

%% Subplot 1: Signal itself over many samples
ax2a = subplot(2,1,1);
styleAxes(ax2a, AX_BG, GRID_C);
hold(ax2a, 'on');

t_long = linspace(0, N_check-1, 5000)';
plot(ax2a, t_long, cos(w0_fail .* t_long), '-', ...
     'Color', [C_CONT 0.30], 'LineWidth', 1.0);

for k = 1:N_check
    plot(ax2a, [n_check(k) n_check(k)], [0, x_fail(k)], '-', ...
         'Color', [C_NPER 0.45], 'LineWidth', 0.7);
end
scatter(ax2a, n_check, x_fail, 8, C_NPER, 'filled');

% Mark n=0 reference value
yline(ax2a, x_fail(1), '--', 'Color', C_ENV, 'LineWidth', 1.1, 'Alpha', 0.8);
text(ax2a, N_check*0.75, x_fail(1)+0.08, 'x[0] = 1.0', ...
     'Color', C_ENV, 'FontSize', 9);

set(ax2a, 'XLim', [-1 N_check], 'YLim', [-1.35 1.35]);
title(ax2a, 'x_3[n] = cos(n)   (w0 = 1 rad/sample)', ...
      'Color', C_NPER, 'FontSize', 11, 'FontWeight', 'bold');
xlabel(ax2a, 'n', 'Color', TEXT_C);
ylabel(ax2a, 'x_3[n]', 'Color', TEXT_C);
hold(ax2a, 'off');

%% Subplot 2: Error |x[n] - x[0]| — if periodic, this hits exactly zero at period
ax2b = subplot(2,1,2);
styleAxes(ax2b, AX_BG, GRID_C);
hold(ax2b, 'on');

error_from_x0 = abs(x_fail - x_fail(1));   % |cos(n) - cos(0)| = |cos(n) - 1|
bar(ax2b, n_check, error_from_x0, 1.0, ...
    'FaceColor', C_NPER, 'EdgeColor', 'none', 'FaceAlpha', 0.75);

% For comparison: what a periodic signal would look like (pi/4 case)
w0_per  = pi/4;
x_per   = cos(w0_per .* n_check);
err_per = abs(x_per - x_per(1));
plot(ax2b, n_check, err_per, 'o', 'Color', C_PER, ...
     'MarkerFaceColor', C_PER, 'MarkerSize', 3.5);

% Zero line — where a periodic signal returns to
yline(ax2b, 0, '-', 'Color', C_ENV, 'LineWidth', 1.2);

legend(ax2b, {'|cos(n) - cos(0)|  (non-periodic)', ...
              '|cos(pi/4 * n) - cos(0)|  (periodic, N=8)'}, ...
       'Color', [0.15 0.15 0.20], 'TextColor', TEXT_C, ...
       'EdgeColor', GRID_C, 'FontSize', 8.5, 'Location', 'northeast');

title(ax2b, ...
      '|x[n] - x[0]|:   If periodic, this must return to EXACTLY 0 at period N', ...
      'Color', TEXT_C, 'FontSize', 10, 'FontWeight', 'bold');
xlabel(ax2b, 'n  (candidate period)', 'Color', TEXT_C);
ylabel(ax2b, '|x[n] - x[0]|', 'Color', TEXT_C);
set(ax2b, 'XLim', [-1 N_check], 'YLim', [-0.05 2.05]);
hold(ax2b, 'off');

sgtitle(fig2, ...
    'The Textbook Failure:  cos(n) has NO integer period N', ...
    'Color', TEXT_C, 'FontSize', 13, 'FontWeight', 'bold');

%% -----------------------------------------------------------------------
%  FIGURE 3: Rationality check table — printed to command window
% -----------------------------------------------------------------------
fprintf('\n');
fprintf('========================================================\n');
fprintf('  Discrete-Time Periodicity Check\n');
fprintf('  Condition: w0 / (2*pi) must be RATIONAL\n');
fprintf('========================================================\n');
fprintf('  %-28s  %-12s  %-14s  %s\n', ...
        'Signal', 'w0 / (2*pi)', 'Rational?', 'Period N');
fprintf('--------------------------------------------------------\n');

check_cases = {
    'cos(pi/4 * n)',         pi/4,       '1/8',           'YES', '8';
    'cos(pi/6 * n)',         pi/6,       '1/12',          'YES', '12';
    'cos(2*pi/7 * n)',       2*pi/7,     '1/7',           'YES', '7';
    'cos(n)  [x3 in text]',  1,          '0.15915... ',   'NO',  'NONE';
    'cos(sqrt(2)*n)',        sqrt(2),    '0.22508... ',   'NO',  'NONE';
    'cos(pi/sqrt(2)*n)',     pi/sqrt(2), '0.35355... ',   'NO',  'NONE';
};

for i = 1:size(check_cases,1)
    fprintf('  %-28s  %-12s  %-14s  %s\n', ...
            check_cases{i,1}, check_cases{i,3}, check_cases{i,4}, check_cases{i,5});
end
fprintf('========================================================\n\n');

fprintf('Key insight:\n');
fprintf('  Continuous cos(w0*t): ALWAYS periodic, T = 2*pi/w0\n');
fprintf('  Discrete  cos(w0*n): periodic ONLY when w0/(2*pi) = k/N (rational)\n');
fprintf('  cos(n) is periodic in continuous time (T=2*pi), but\n');
fprintf('  has NO integer period in discrete time.\n\n');

%% -----------------------------------------------------------------------
%  LOCAL HELPERS
% -----------------------------------------------------------------------
function styleAxes(ax, bg, grid_c)
    set(ax, 'Color', bg, ...
            'XColor', [0.6 0.6 0.6], 'YColor', [0.6 0.6 0.6], ...
            'GridColor', grid_c, 'GridAlpha', 0.5);
    grid(ax, 'on');
    box(ax, 'on');
end
