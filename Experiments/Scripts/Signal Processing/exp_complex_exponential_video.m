%% exp_complex_exponential_video.m
%
% Animated Video: Complex Exponential Sequence  x[n] = A * alpha^n
%
% Each video frame adds one new sample n, showing simultaneously:
%   TOP    : Complex plane — rotating vector + traced spiral/circle path
%   BOTTOM : Re{x[n]} stem plot building up, with envelope overlay
%
% One MP4 video is saved per parameter configuration.
%
% Author : Ocean Library — Signal Processing Experiments
% Date   : 2026-05-19
% -----------------------------------------------------------------------

clc; clear; close all;

%% -----------------------------------------------------------------------
%  CONFIGURATIONS  — each row is one video
%  [A,  alpha_mag,  w0]
% -----------------------------------------------------------------------
configs = [
    1.0,  0.95,  pi/16;   % Decaying — gentle inward spiral  (|alpha|<1)
    1.0,  1.00,  pi/16;   % Stable   — perfect circle         (|alpha|=1)
    1.0,  1.05,  pi/16;   % Growing  — gentle outward spiral  (|alpha|>1)
    ];
    

N           = 100;     % Total samples per video  n = 0,1,...,N-1
fps         = 12;     % Frames per second
pause_extra = 8;      % Extra frames held at final state
n_all       = (0:N-1)';

% Output folder (same directory as this script)
out_dir = fileparts(mfilename('fullpath'));
if isempty(out_dir), out_dir = pwd; end

%% -----------------------------------------------------------------------
%  COLOR SCHEME
% -----------------------------------------------------------------------
BG        = [0.10 0.10 0.14];   % figure background
AX_BG     = [0.07 0.07 0.11];   % axes background
GRID_C    = [0.25 0.25 0.30];   % grid lines
TRACE_C   = [0.25 0.60 0.95];   % spiral / stem color
ENV_C     = [0.30 0.85 0.50];   % envelope lines
VEC_C     = [1.00 0.80 0.20];   % current rotating vector (gold)
DOT_C     = [1.00 0.35 0.35];   % current sample dot (red)
TEXT_C    = [0.85 0.85 0.90];   % labels

%% -----------------------------------------------------------------------
%  BUILD A VIDEO FOR EACH CONFIGURATION
% -----------------------------------------------------------------------
for cfg = 1:size(configs,1)

    A         = configs(cfg,1);
    alpha_mag = configs(cfg,2);
    w0        = configs(cfg,3);

    alpha = alpha_mag * exp(1j * w0);
    x     = A .* alpha.^n_all;        % full sequence, column vector

    % Axis limits — give 20 % headroom
    lim_stem = max(abs(real(x))) * 1.30;
    lim_stem = max(lim_stem, 0.5);

    r_max = max(abs(x)) * 1.30;
    r_max = max(r_max, 0.5);

    % Descriptive filename tag
    if     alpha_mag < 1,   behaviour = 'decay';
    elseif alpha_mag == 1,  behaviour = 'stable';
    else,                   behaviour = 'grow';
    end
    fname = sprintf('complex_exp_A%.1f_mag%.2f_w0pi%.3f_%s.mp4', ...
                    A, alpha_mag, w0/pi, behaviour);
    fpath = fullfile(out_dir, fname);

    fprintf('Generating: %s\n', fname);

    %% ---- Setup figure -------------------------------------------------
    fig = figure('Color', BG, ...
                 'Position', [80 80 920 740], ...
                 'Visible', 'on');

    % ---- Figure-level title (top bar) -----------------------------------
    annotation(fig, 'textbox', [0 0.955 1 0.045], ...
        'String', sprintf('x[n] = A · alpha^n      A = %.2f    |alpha| = %.2f    w0 = %.4f rad/samp  (%.4f × pi)', ...
                          A, alpha_mag, w0, w0/pi), ...
        'Color', TEXT_C, 'FontSize', 10, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'EdgeColor', 'none', 'BackgroundColor', [0.15 0.15 0.20]);

    % --- Complex plane axes (top) ----------------------------------------
    ax_cp = axes('Parent', fig, ...
                 'Position', [0.10  0.44  0.72  0.50], ...
                 'Color',    AX_BG, ...
                 'XColor',   GRID_C, 'YColor', GRID_C, ...
                 'GridColor',GRID_C, 'GridAlpha', 0.5, ...
                 'XLim', [-r_max r_max], 'YLim', [-r_max r_max]);
    axis(ax_cp, 'equal');
    grid(ax_cp, 'on');  box(ax_cp, 'on');
    hold(ax_cp, 'on');

    % Static decorations: unit circle + crosshairs
    th = linspace(0, 2*pi, 300);
    plot(ax_cp, cos(th), sin(th), '--', 'Color', [0.40 0.40 0.55], 'LineWidth', 0.9);
    xline(ax_cp, 0, 'Color', [0.40 0.40 0.55], 'LineWidth', 0.8);
    yline(ax_cp, 0, 'Color', [0.40 0.40 0.55], 'LineWidth', 0.8);

    % Axis labels
    xlabel(ax_cp, 'Real Axis', 'Color', TEXT_C, 'FontSize', 9, 'FontWeight', 'bold');
    ylabel(ax_cp, 'Imaginary Axis', 'Color', TEXT_C, 'FontSize', 9, 'FontWeight', 'bold');
    title(ax_cp, 'Complex Plane  —  Trajectory of  alpha^n', ...
          'Color', TEXT_C, 'FontSize', 11, 'FontWeight', 'bold');

    % Animated objects — complex plane
    h_trace = plot(ax_cp, NaN, NaN, '-',  'Color', TRACE_C, 'LineWidth', 1.6);
    h_dots  = scatter(ax_cp, NaN, NaN, 24, TRACE_C, 'filled');
    h_vec   = plot(ax_cp, NaN, NaN, '-',  'Color', VEC_C,   'LineWidth', 2.8);
    h_tip   = scatter(ax_cp, NaN, NaN, 90, VEC_C,   'filled', ...
                      'MarkerEdgeColor', 'w', 'LineWidth', 1.0);

    % Legend inside complex plane
    legend(ax_cp, [h_trace, h_vec, h_tip], ...
           {'Traced path  (past samples)', ...
            'alpha^n  (rotating vector)', ...
            'Current tip  x[n]'}, ...
           'Location', 'southoutside', 'Orientation', 'horizontal', ...
           'TextColor', TEXT_C, 'Color', [0.10 0.10 0.18], ...
           'EdgeColor', GRID_C, 'FontSize', 8);

    % --- Live readout panel (right of complex plane) ---------------------
    ax_info = axes('Parent', fig, ...
                   'Position', [0.84  0.44  0.14  0.50], ...
                   'Color',    [0.10 0.10 0.18], ...
                   'XColor',   'none', 'YColor', 'none', ...
                   'XLim', [0 1], 'YLim', [0 1]);
    box(ax_info, 'on');
    set(ax_info, 'XTickLabel', [], 'YTickLabel', []);

    % Static readout labels
    text(ax_info, 0.5, 0.97, 'Live Values', ...
         'Color', TEXT_C, 'FontSize', 9, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
    text(ax_info, 0.10, 0.83, 'n', ...
         'Color', [0.65 0.65 0.70], 'FontSize', 8);
    text(ax_info, 0.10, 0.64, 'Angle', ...
         'Color', [0.65 0.65 0.70], 'FontSize', 8);
    text(ax_info, 0.10, 0.45, '|x[n]|', ...
         'Color', [0.65 0.65 0.70], 'FontSize', 8);
    text(ax_info, 0.10, 0.27, 'Re{x}', ...
         'Color', [0.65 0.65 0.70], 'FontSize', 8);
    text(ax_info, 0.10, 0.10, 'Im{x}', ...
         'Color', [0.65 0.65 0.70], 'FontSize', 8);

    % Animated readout values
    h_v_n   = text(ax_info, 0.5, 0.74, '', ...
                   'Color', VEC_C, 'FontSize', 14, 'FontWeight', 'bold', ...
                   'HorizontalAlignment', 'center');
    h_v_ang = text(ax_info, 0.5, 0.56, '', ...
                   'Color', TEXT_C, 'FontSize', 8.5, ...
                   'HorizontalAlignment', 'center');
    h_v_mag = text(ax_info, 0.5, 0.37, '', ...
                   'Color', TEXT_C, 'FontSize', 8.5, ...
                   'HorizontalAlignment', 'center');
    h_v_re  = text(ax_info, 0.5, 0.19, '', ...
                   'Color', TRACE_C, 'FontSize', 8.5, ...
                   'HorizontalAlignment', 'center');
    h_v_im  = text(ax_info, 0.5, 0.02, '', ...
                   'Color', [0.80 0.55 0.90], 'FontSize', 8.5, ...
                   'HorizontalAlignment', 'center');

    % --- Stem plot axes (bottom) -----------------------------------------
    ax_st = axes('Parent', fig, ...
                 'Position', [0.10  0.06  0.82  0.33], ...
                 'Color',    AX_BG, ...
                 'XColor',   GRID_C, 'YColor', GRID_C, ...
                 'GridColor',GRID_C, 'GridAlpha', 0.5, ...
                 'XLim', [-1 N], 'YLim', [-lim_stem lim_stem]);
    grid(ax_st, 'on');  box(ax_st, 'on');
    hold(ax_st, 'on');

    % Static envelope curves with labels
    n_cont  = linspace(0, N-1, 600);
    env_p   =  A .* alpha_mag.^n_cont;
    env_n   = -env_p;
    plot(ax_st, n_cont, env_p, '--', 'Color', ENV_C, 'LineWidth', 1.2);
    plot(ax_st, n_cont, env_n, '--', 'Color', ENV_C, 'LineWidth', 1.2);
    yline(ax_st, 0, 'Color', [0.45 0.45 0.50], 'LineWidth', 0.8);
    text(ax_st, N*0.75, A*alpha_mag^(N*0.75)*1.08, ...
         sprintf('Envelope: A·|alpha|^n'), ...
         'Color', ENV_C, 'FontSize', 7.5, 'FontAngle', 'italic');

    xlabel(ax_st, 'Sample index  n', 'Color', TEXT_C, 'FontSize', 9, 'FontWeight', 'bold');
    ylabel(ax_st, 'Re{x[n]}', 'Color', TEXT_C, 'FontSize', 9, 'FontWeight', 'bold');
    title(ax_st, 'Real Part  —  Re{x[n]} = A · |alpha|^n · cos(w0·n)', ...
          'Color', TEXT_C, 'FontSize', 10, 'FontWeight', 'bold');

    % Pre-allocate N stem lines and N head markers (initially invisible)
    h_stems = gobjects(N, 1);
    h_heads = gobjects(N, 1);
    for k = 1:N
        h_stems(k) = plot(ax_st, [NaN NaN], [0 0], '-',  ...
                          'Color', TRACE_C, 'LineWidth', 1.3, ...
                          'Visible', 'off');
        h_heads(k) = plot(ax_st, NaN, 0, 'o', ...
                          'Color', TRACE_C, ...
                          'MarkerFaceColor', TRACE_C, ...
                          'MarkerEdgeColor', 'none', ...
                          'MarkerSize', 6, 'Visible', 'off');
    end
    % Highlight dot for the current sample
    h_cur = plot(ax_st, NaN, NaN, 'o', ...
                 'Color', DOT_C, 'MarkerFaceColor', DOT_C, ...
                 'MarkerEdgeColor', 'w', 'LineWidth', 0.9, ...
                 'MarkerSize', 10);
    % Live Re{x[n]} value label on stem plot
    h_val_lbl = text(ax_st, 0, 0, '', ...
                     'Color', DOT_C, 'FontSize', 8, 'FontWeight', 'bold', ...
                     'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');

    %% ---- VideoWriter --------------------------------------------------
    vw = VideoWriter(fpath, 'MPEG-4');
    vw.FrameRate = fps;
    vw.Quality   = 95;
    open(vw);

    %% ---- Animation loop -----------------------------------------------
    for frame = 1 : N + pause_extra

        k   = min(frame, N);   % clamp so last pause_extra frames are static
        n_k = n_all(k);        % current time index (0-based)
        x_k = x(k);            % current complex value

        %% Complex plane updates
        set(h_trace, 'XData', real(x(1:k)), 'YData', imag(x(1:k)));
        set(h_dots,  'XData', real(x(1:k)), 'YData', imag(x(1:k)));
        set(h_vec,   'XData', [0, real(x_k)], 'YData', [0, imag(x_k)]);
        set(h_tip,   'XData', real(x_k),       'YData', imag(x_k));

        % Live readout panel values
        angle_deg = rad2deg(angle(x_k));   % angle in degrees
        set(h_v_n,   'String', sprintf('%d', n_k));
        set(h_v_ang, 'String', sprintf('%.1f deg\n(%.4f rad)', angle_deg, angle(x_k)));
        set(h_v_mag, 'String', sprintf('%.4f', abs(x_k)));
        set(h_v_re,  'String', sprintf('%.4f', real(x_k)));
        set(h_v_im,  'String', sprintf('%.4f', imag(x_k)));

        %% Stem plot updates — reveal sample k
        rv = real(x_k);
        if frame <= N
            set(h_stems(k), 'XData', [n_k n_k], 'YData', [0, rv], ...
                            'Visible', 'on');
            set(h_heads(k), 'XData', n_k, 'YData', rv, ...
                            'Visible', 'on');
        end

        % Move the highlight dot and live value label
        set(h_cur, 'XData', n_k, 'YData', rv);
        set(h_val_lbl, 'Position', [n_k + 0.5, rv + lim_stem*0.04, 0], ...
                       'String', sprintf('%.3f', rv));

        drawnow;

        fr = getframe(fig);
        writeVideo(vw, fr);
    end

    close(vw);
    close(fig);

    fprintf('  Saved -> %s\n', fpath);
end

fprintf('\nAll %d videos saved to:\n  %s\n', size(configs,1), out_dir);
