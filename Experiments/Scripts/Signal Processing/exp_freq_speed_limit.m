%% exp_freq_speed_limit.m
%
% Test 1: The Speed Limit
%
% Sweeps w0 from pi/4 to 2*pi and plots both the continuous analog curve
% and the discrete samples simultaneously. Demonstrates that once w0 > pi,
% the discrete stems decouple from the fast continuous wave and instead
% outline a slower aliased frequency.
%
% Aliasing identity:  cos((2*pi - w0)*n) = cos(w0*n) for all integer n
%
% Author : Ocean Library — Signal Processing Experiments
% Date   : 2026-05-19
% -----------------------------------------------------------------------

clc; clear; close all;

%% -----------------------------------------------------------------------
%  COLOR SCHEME
% -----------------------------------------------------------------------
BG       = [0.10 0.10 0.14];
AX_BG    = [0.07 0.07 0.11];
AX_ALIAS = [0.16 0.08 0.08];   % reddish background for alias zone
GRID_C   = [0.25 0.25 0.30];
TEXT_C   = [0.85 0.85 0.90];
C_CONT   = [0.30 0.85 0.50];   % continuous curve — green
C_NORM   = [0.25 0.65 0.95];   % discrete stems — blue (normal zone)
C_ALIAS  = [1.00 0.40 0.40];   % discrete stems — red  (alias zone)
C_GOLD   = [1.00 0.80 0.20];   % pi marker / annotations

%% -----------------------------------------------------------------------
%  FREQUENCY VALUES TO TEST
% -----------------------------------------------------------------------
%  Row 1: normal zone  (w0 <= pi)
%  Row 2: alias zone   (w0  > pi)
w0_values = [pi/4,   pi/2,   3*pi/4, pi, ...     % Row 1: normal zone
             7*pi/4, 3*pi/2, 5*pi/4, 2*pi];     % Row 2: alias zone (col-matched)

labels = {
    'pi/4',  'pi/2',  '3pi/4',  'pi', ...
    '7pi/4', '3pi/2', '5pi/4',  '2pi'
};

alias_of = {
    '',      '',      '',       '', ...
    'pi/4',  'pi/2',  '3pi/4',  '0 (DC)'
};

N_samp  = 28;                          % discrete samples shown
n       = (0:N_samp-1)';
t_cont  = linspace(0, N_samp-1, 2000)';

%% -----------------------------------------------------------------------
%  FIGURE 1: 2 x 4 SPEED LIMIT GRID
% -----------------------------------------------------------------------
fig1 = figure('Color', BG, 'Position', [40 40 1400 720]);

for i = 1:8
    w0       = w0_values(i);
    is_alias = (w0 > pi + 1e-9);

    x_disc = cos(w0 .* n);
    x_cont = cos(w0 .* t_cont);

    ax = subplot(2, 4, i);

    if is_alias
        set(ax, 'Color', AX_ALIAS);
    else
        set(ax, 'Color', AX_BG);
    end
    set(ax, 'XColor', GRID_C, 'YColor', GRID_C, ...
            'GridColor', GRID_C, 'GridAlpha', 0.5, ...
            'XLim', [-0.5 N_samp-0.5], 'YLim', [-1.55 1.70]);
    grid(ax, 'on');  box(ax, 'on');
    hold(ax, 'on');

    if is_alias
        w0_alias = 2*pi - w0;

        % Fast continuous wave: very faint — just shows "what the wave is doing"
        plot(ax, t_cont, x_cont, '-', 'Color', [C_CONT 0.12], 'LineWidth', 0.7);

        % Aliased slow wave: PROMINENT — this is what the dots trace
        if w0_alias > 1e-9
            x_alias_cont = cos(w0_alias .* t_cont);
            plot(ax, t_cont, x_alias_cont, '-', ...
                 'Color', [C_ALIAS 0.90], 'LineWidth', 2.5);
        else
            % DC alias (w0 = 2*pi): flat line at +1
            plot(ax, [0 N_samp], [1 1], '-', 'Color',[C_ALIAS 0.90],'LineWidth',2.5);
        end
        stem_color = C_ALIAS;

        % Discrete stems — thin lines, BIG dots so they clearly sit on the alias curve
        for k = 1:N_samp
            plot(ax, [n(k) n(k)], [0, x_disc(k)], '-', ...
                 'Color', [stem_color 0.45], 'LineWidth', 0.9);
        end
        scatter(ax, n, x_disc, 40, stem_color, 'filled', ...
                'MarkerEdgeColor','w','LineWidth',0.6);

        % Annotation labels
        alias_str = alias_of{i};
        title(ax, sprintf('w0 = %s  (%.4f rad)', labels{i}, w0), ...
              'Color', C_ALIAS, 'FontSize', 8.5, 'FontWeight', 'bold');
        text(ax, 0.3, 1.52, sprintf('Samples trace:  w0 = %s',alias_str), ...
             'Color', C_ALIAS, 'FontSize', 8, 'FontWeight', 'bold', ...
             'BackgroundColor',[0.22 0.06 0.06]);
        % Draw small ghost label for the faint fast wave
        text(ax, N_samp*0.55, -1.45, sprintf('(fast wave at %s is invisible to grid)',labels{i}), ...
             'Color',[C_CONT 0.55],'FontSize',6.5,'FontAngle','italic');

    else
        % Normal zone: continuous wave visible and aligned with stems
        plot(ax, t_cont, x_cont, '-', 'Color', [C_CONT 0.70], 'LineWidth', 1.5);
        stem_color = C_NORM;

        for k = 1:N_samp
            plot(ax, [n(k) n(k)], [0, x_disc(k)], '-', ...
                 'Color', [stem_color 0.65], 'LineWidth', 1.1);
        end
        scatter(ax, n, x_disc, 28, stem_color, 'filled', ...
                'MarkerEdgeColor','w','LineWidth',0.5);

        if abs(w0 - pi) < 1e-9
            title(ax, sprintf('w0 = %s  ---  SPEED LIMIT', labels{i}), ...
                  'Color', C_GOLD, 'FontSize', 8.5, 'FontWeight', 'bold');
            text(ax, 0.3, 1.52,'Max rate: alternates +1, -1, +1, -1 ...', ...
                 'Color',C_GOLD,'FontSize',7.5,'FontWeight','bold');
        else
            title(ax, sprintf('w0 = %s  (%.4f rad)', labels{i}, w0), ...
                  'Color', C_NORM, 'FontSize', 8.5, 'FontWeight', 'bold');
            text(ax, 0.3, 1.52,'Stems sit ON the continuous wave', ...
                 'Color',[C_CONT 0.80],'FontSize',7,'FontAngle','italic');
        end
    end

    xlabel(ax, 'n', 'Color', TEXT_C, 'FontSize', 8);
    ylabel(ax, 'cos(w0 n)', 'Color', TEXT_C, 'FontSize', 8);
    hold(ax, 'off');
end

% Row labels via annotations
annotation(fig1, 'textbox', [0.01 0.92 0.08 0.06], ...
    'String', 'NORMAL ZONE', 'Color', C_NORM, 'FontSize', 9, ...
    'FontWeight', 'bold', 'EdgeColor', 'none', 'BackgroundColor', 'none', ...
    'Rotation', 90, 'HorizontalAlignment', 'center');
annotation(fig1, 'textbox', [0.01 0.45 0.08 0.06], ...
    'String', 'ALIAS ZONE', 'Color', C_ALIAS, 'FontSize', 9, ...
    'FontWeight', 'bold', 'EdgeColor', 'none', 'BackgroundColor', 'none', ...
    'Rotation', 90, 'HorizontalAlignment', 'center');

sgtitle(fig1, ...
    'Test 1: The Speed Limit   |   Discrete samples cannot resolve w0 > pi', ...
    'Color', TEXT_C, 'FontSize', 13, 'FontWeight', 'bold');

%% -----------------------------------------------------------------------
%  FIGURE 2: FOLDING DIAGRAM
%  Shows the frequency axis from 0 to 2*pi with the fold at pi
% -----------------------------------------------------------------------
fig2 = figure('Color', BG, 'Position', [80 80 1000 380]);
ax2  = axes('Parent', fig2, 'Color', AX_BG, ...
            'XColor', TEXT_C, 'YColor', GRID_C, ...
            'GridColor', GRID_C, 'GridAlpha', 0.4, ...
            'XLim', [0 2*pi], 'YLim', [-0.5 2.5]);
grid(ax2, 'on');  box(ax2, 'on');  hold(ax2, 'on');

% Frequency axis line
plot(ax2, [0 2*pi], [1 1], '-', 'Color', [TEXT_C 0.3], 'LineWidth', 1.0);

% Unique zone (0 to pi) — filled region
fill(ax2, [0 pi pi 0], [0 0 2.2 2.2], C_NORM, 'FaceAlpha', 0.08, 'EdgeColor', 'none');
fill(ax2, [pi 2*pi 2*pi pi], [0 0 2.2 2.2], C_ALIAS, 'FaceAlpha', 0.08, 'EdgeColor', 'none');

% Pi wall
xline(ax2, pi, '-', 'Color', C_GOLD, 'LineWidth', 2.5, 'Alpha', 0.9);
text(ax2, pi, 2.30, 'w0 = pi   (SPEED LIMIT)', ...
     'Color', C_GOLD, 'FontSize', 10, 'FontWeight', 'bold', ...
     'HorizontalAlignment', 'center');

% Normal zone w0 values — vertical markers
w0_normal = [pi/4, pi/2, 3*pi/4, pi];
for w = w0_normal
    xline(ax2, w, ':', 'Color', C_NORM, 'LineWidth', 1.2, 'Alpha', 0.8);
end

% Alias zone w0 values — with curved arrows back to their aliases
w0_alias_pairs = [5*pi/4, 3*pi/4; 3*pi/2, pi/2; 7*pi/4, pi/4; 2*pi, 0];
for i = 1:size(w0_alias_pairs,1)
    w_high = w0_alias_pairs(i,1);
    w_low  = w0_alias_pairs(i,2);
    xline(ax2, w_high, ':', 'Color', C_ALIAS, 'LineWidth', 1.2, 'Alpha', 0.8);
    % Folding arrow
    mid_x = (w_high + w_low) / 2;
    mid_y = 1.6 + 0.3*(i-1)*0;
    annotation(fig2, 'arrow', ...
        [w_high/(2*pi)*0.84+0.08,  w_low/(2*pi)*0.84+0.08], ...
        [0.58 0.58], ...
        'Color', C_ALIAS, 'LineWidth', 1.2, 'HeadStyle', 'vback2', ...
        'HeadWidth', 7, 'HeadLength', 7);
end

text(ax2, pi/2, 0.25, 'UNIQUE ZONE  (0 to pi)', ...
     'Color', C_NORM, 'FontSize', 10, 'FontWeight', 'bold', ...
     'HorizontalAlignment', 'center');
text(ax2, 3*pi/2, 0.25, 'ALIAS ZONE  (pi to 2pi)', ...
     'Color', C_ALIAS, 'FontSize', 10, 'FontWeight', 'bold', ...
     'HorizontalAlignment', 'center');

set(ax2, 'XTick', [0, pi/4, pi/2, 3*pi/4, pi, 5*pi/4, 3*pi/2, 7*pi/4, 2*pi], ...
    'XTickLabel', {'0','pi/4','pi/2','3pi/4','pi','5pi/4','3pi/2','7pi/4','2pi'}, ...
    'YTick', [], 'FontSize', 8.5, 'TickLabelInterpreter', 'none');

title(ax2, ...
    'Frequency Folding Diagram:  cos(w0*n) = cos((2pi - w0)*n) for all integer n', ...
    'Color', TEXT_C, 'FontSize', 11, 'FontWeight', 'bold');
xlabel(ax2, 'w0  (rad / sample)', 'Color', TEXT_C, 'FontSize', 10);
hold(ax2, 'off');

fprintf('Test 1 figures generated.\n');
