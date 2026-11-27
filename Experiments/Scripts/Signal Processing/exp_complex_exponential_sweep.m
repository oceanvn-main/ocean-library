%% exp_complex_exponential_sweep.m
%
% Experiment: Parametric Sweep of the Complex Exponential Sequence
%
% The sequence is:  x[n] = A * alpha^n
%
% where alpha = |alpha| * exp(j*w0)
%
% This experiment sweeps three independent parameters:
%   A     : Amplitude scaling (real scalar)
%   |alpha|: Magnitude of alpha  -> controls envelope (grow / decay / stable)
%   w0    : Digital frequency (rad/sample) -> controls oscillation speed
%
% For each configuration we plot:
%   (1) Real part   of x[n]  (observable cosine-like signal)
%   (2) Imaginary part of x[n] (observable sine-like signal)
%   (3) Magnitude   |x[n]|   (the envelope)
%   (4) Complex plane trajectory (spiral / circle)
%
% Author : Ocean Library — Signal Processing Experiments
% Date   : 2026-05-19
% -----------------------------------------------------------------------

clc; clear; close all;

%% -----------------------------------------------------------------------
%  PARAMETER GRID
%  Modify any of these vectors to add / remove sweep values.
% -----------------------------------------------------------------------

A_values        = [1, 2, 0.5];          % Amplitude scaling
alpha_mag_values = [0.85, 1.0, 1.10];  % |alpha|: <1 decay, =1 stable, >1 grow
w0_values       = [pi/8, pi/4, pi/2];  % Digital frequency (rad/sample)

N = 40;          % Number of time steps  n = 0, 1, ..., N-1
n = (0:N-1)';    % Column vector of time indices

%% -----------------------------------------------------------------------
%  COLOR PALETTE  (one distinct color per |alpha| level)
% -----------------------------------------------------------------------
palette = [
    0.20  0.60  0.86;   % blue-ish    (decay)
    0.18  0.80  0.44;   % green-ish   (stable)
    0.95  0.35  0.35;   % red-ish     (grow)
];

%% -----------------------------------------------------------------------
%  SWEEP LOOP
% -----------------------------------------------------------------------
for iA = 1:numel(A_values)
    A = A_values(iA);

    %% --- Figure 1 : Real part for all |alpha| & w0 combinations --------
    fig1 = figure('Name', sprintf('Real part  |  A = %.2f', A), ...
                  'Color', [0.12 0.12 0.16], ...
                  'Position', [50 50 1400 800]);

    nRows = numel(alpha_mag_values);
    nCols = numel(w0_values);
    sp_idx = 0;

    for iMag = 1:nRows
        alpha_mag = alpha_mag_values(iMag);

        for iW = 1:nCols
            w0 = w0_values(iW);

            % Build alpha and compute the sequence
            alpha   = alpha_mag * exp(1j * w0);
            x       = A .* alpha.^n;          % x[n] = A * alpha^n

            sp_idx  = sp_idx + 1;
            ax = subplot(nRows, nCols, sp_idx);
            styleAxes(ax);

            stem(n, real(x), 'filled', ...
                 'Color',           palette(iMag,:), ...
                 'MarkerFaceColor', palette(iMag,:), ...
                 'MarkerSize', 4, 'LineWidth', 1.2);

            % Overlay continuous envelope  A * |alpha|^n * cos(w0*n)
            hold on;
            n_cont = linspace(0, N-1, 500);
            env_pos =  A .* alpha_mag.^n_cont .* cos(w0 .* n_cont);
            plot(n_cont, env_pos, '--', 'Color', [palette(iMag,:) 0.45], 'LineWidth', 1);
            hold off;

            title(ax, sprintf('|\\alpha|=%.2f  w_0=%.3f\\pi', ...
                  alpha_mag, w0/pi), ...
                  'Color','w','FontSize',8);

            xlabel(ax,'n (samples)','Color',[0.7 0.7 0.7],'FontSize',7);
            ylabel(ax,'Re\{x[n]\}',  'Color',[0.7 0.7 0.7],'FontSize',7);
        end
    end

    sgtitle(fig1, sprintf('Real Part  of  x[n] = A \\cdot \\alpha^n        A = %.2f', A), ...
            'Color','w','FontSize',13,'FontWeight','bold');

    %% --- Figure 2 : Complex-plane trajectory (spiral) ------------------
    fig2 = figure('Name', sprintf('Complex Plane  |  A = %.2f', A), ...
                  'Color', [0.12 0.12 0.16], ...
                  'Position', [100 100 1400 800]);

    sp_idx = 0;
    for iMag = 1:nRows
        alpha_mag = alpha_mag_values(iMag);
        for iW = 1:nCols
            w0 = w0_values(iW);

            alpha = alpha_mag * exp(1j * w0);
            x     = A .* alpha.^n;

            sp_idx = sp_idx + 1;
            ax = subplot(nRows, nCols, sp_idx);
            styleAxes(ax);

            % Gradient coloring: early samples darker, late samples brighter
            clr_map = makeGradient(palette(iMag,:), N);
            for k = 1:N-1
                plot(real(x(k:k+1)), imag(x(k:k+1)), '-', ...
                     'Color', clr_map(k,:), 'LineWidth', 1.4);
                hold on;
            end
            scatter(real(x), imag(x), 18, clr_map, 'filled');
            % Mark n=0
            scatter(real(x(1)), imag(x(1)), 60, 'w', 'filled', ...
                    'MarkerEdgeColor','none');
            hold off;

            axis equal;
            title(ax, sprintf('|\\alpha|=%.2f  w_0=%.3f\\pi', ...
                  alpha_mag, w0/pi), ...
                  'Color','w','FontSize',8);
            xlabel(ax,'Real','Color',[0.7 0.7 0.7],'FontSize',7);
            ylabel(ax,'Imag','Color',[0.7 0.7 0.7],'FontSize',7);
        end
    end

    sgtitle(fig2, sprintf('Complex Plane Trajectory        A = %.2f', A), ...
            'Color','w','FontSize',13,'FontWeight','bold');

    %% --- Figure 3 : Magnitude envelope comparison (all w0 on one plot) -
    fig3 = figure('Name', sprintf('Envelope  |  A = %.2f', A), ...
                  'Color', [0.12 0.12 0.16], ...
                  'Position', [150 150 900 500]);
    ax3  = axes('Parent', fig3, 'Color', [0.08 0.08 0.12], ...
                'XColor', [0.6 0.6 0.6], 'YColor', [0.6 0.6 0.6], ...
                'GridColor', [0.3 0.3 0.3], 'GridAlpha', 0.4);
    box(ax3,'on'); hold(ax3,'on'); grid(ax3,'on');

    leg_entries = {};
    for iMag = 1:nRows
        alpha_mag = alpha_mag_values(iMag);
        % Envelope |x[n]| = A * |alpha|^n  (independent of w0)
        env = A .* alpha_mag.^n;
        plot(ax3, n, env, 'o-', ...
             'Color', palette(iMag,:), ...
             'MarkerFaceColor', palette(iMag,:), ...
             'MarkerSize', 4, 'LineWidth', 1.5);
        leg_entries{end+1} = sprintf('|\\alpha| = %.2f', alpha_mag); %#ok<AGROW>
    end

    legend(ax3, leg_entries, 'Location','best', ...
           'TextColor','w','Color',[0.15 0.15 0.2],'EdgeColor',[0.4 0.4 0.4]);
    title(ax3, sprintf('Magnitude Envelope  |x[n]| = A \\cdot |\\alpha|^n        A = %.2f', A), ...
          'Color','w','FontSize',11,'FontWeight','bold');
    xlabel(ax3,'n (samples)','Color',[0.7 0.7 0.7]);
    ylabel(ax3,'|x[n]|','Color',[0.7 0.7 0.7]);

    hold(ax3,'off');
end

%% -----------------------------------------------------------------------
%  SUMMARY TABLE  (printed to Command Window)
% -----------------------------------------------------------------------
fprintf('\n');
fprintf('=======================================================\n');
fprintf('  Complex Exponential Sweep — Parameter Summary\n');
fprintf('=======================================================\n');
fprintf('  x[n] = A * alpha^n,   alpha = |alpha| * exp(j*w0)\n');
fprintf('-------------------------------------------------------\n');
fprintf('  %-8s  %-10s  %-14s  %-20s\n', ...
        'A', '|alpha|', 'w0 (rad/samp)', 'Behaviour');
fprintf('-------------------------------------------------------\n');

behaviours = {'Decaying oscillation', 'Pure oscillation (stable)', ...
              'Growing oscillation'};

for iA = 1:numel(A_values)
    for iMag = 1:numel(alpha_mag_values)
        for iW = 1:numel(w0_values)
            fprintf('  %-8.2f  %-10.2f  %-14.4f  %s\n', ...
                A_values(iA), ...
                alpha_mag_values(iMag), ...
                w0_values(iW), ...
                behaviours{iMag});
        end
    end
end
fprintf('=======================================================\n\n');

disp('Experiment complete. All figures generated.');

%% -----------------------------------------------------------------------
%  LOCAL HELPER FUNCTIONS
% -----------------------------------------------------------------------

function styleAxes(ax)
% Apply dark-theme style to an axes object.
    set(ax, 'Color',      [0.08 0.08 0.12], ...
            'XColor',     [0.6  0.6  0.6 ], ...
            'YColor',     [0.6  0.6  0.6 ], ...
            'GridColor',  [0.3  0.3  0.3 ], ...
            'GridAlpha',  0.4);
    grid(ax, 'on');
    box(ax,  'on');
end

function cmap = makeGradient(baseColor, N)
% Create an N-by-3 color matrix that fades from dark to baseColor.
    dark  = baseColor * 0.25;
    cmap  = zeros(N, 3);
    t     = linspace(0, 1, N)';
    for ch = 1:3
        cmap(:,ch) = dark(ch) + t .* (baseColor(ch) - dark(ch));
    end
end
