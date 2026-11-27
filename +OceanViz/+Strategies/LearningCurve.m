classdef LearningCurve < handle
    % OCEANVIZ.STRATEGIES.LEARNINGCURVE
    % Strategy for plotting convergence and learning curves.
    %
    % OVERVIEW:
    %   Designed for training/test error vs iterations plots — clean continuous
    %   lines with NO markers. Matches the aesthetic of textbook convergence
    %   figures (e.g., ESL Figure 10.13).
    %
    %   Unlike MultiComparison (which adds per-point markers), LearningCurve
    %   renders smooth lines optimized for dense x-axes (100–10,000+ points).
    %
    % DATA INTERFACE:
    %   data.X          — M×1 vector of x-axis values (e.g., iteration index)
    %   data.Lines      — cell array of structs, each with:
    %       .Y          — M×1 vector of y values
    %       .Name       — display name for legend
    %       .LineStyle  — (optional) '-', '--', ':', '-.'. Default: '-'
    %       .LineWidth  — (optional) scalar. Default: 1.8
    %   data.RefLines   — (optional) cell array of structs, each with:
    %       .Y          — scalar y value for a horizontal reference line
    %       .Name       — display name for legend ('' to hide)
    %       .LineStyle  — (optional) '--' or ':'. Default: ':'
    %
    % USAGE:
    %   viz = OceanViz.Core('Theme', 'Dark');
    %   data.X = (1:M)';
    %   data.Lines = { struct('Y', aae_train, 'Name', 'Train AAE'), ...
    %                  struct('Y', aae_test,  'Name', 'Test AAE') };
    %   data.RefLines = { struct('Y', baseline, 'Name', 'Baseline') };
    %   viz.plot(ax, 'LearningCurve', data, 'Title', '...', 'XLabel', '...', 'YLabel', '...');
    %
    % See also: OceanViz.Core, OceanViz.Strategies.MultiComparison

    properties
        Parent  % Injected OceanViz.Core instance
    end

    methods
        function plot(obj, ax, data, options)
            % PLOT  Render learning curves onto the given axes.
            arguments
                obj
                ax   (1,1) matlab.graphics.axis.Axes
                data (1,1) struct
                options struct
            end

            C       = obj.Parent.Colors;
            nLines  = numel(data.Lines);
            palette = obj.Parent.getColorSequence(nLines);

            hold(ax, 'on');

            % ── 1. Main learning curves (clean lines, NO markers) ──────────
            for i = 1:nLines
                L = data.Lines{i};

                % Resolve optional per-line style
                ls = '-';
                if isfield(L, 'LineStyle'), ls = L.LineStyle; end
                lw = 1.8;
                if isfield(L, 'LineWidth'), lw = L.LineWidth; end

                plot(ax, data.X, L.Y, ...
                    'Color',     palette(i, :), ...
                    'LineStyle', ls,            ...
                    'LineWidth', lw,            ...
                    'DisplayName', L.Name);
            end

            % ── 2. Optional reference lines (horizontal dashed) ────────────
            if isfield(data, 'RefLines') && ~isempty(data.RefLines)
                refPalette = obj.Parent.getColorSequence(nLines + numel(data.RefLines));
                for j = 1:numel(data.RefLines)
                    R  = data.RefLines{j};
                    ls = ':';
                    if isfield(R, 'LineStyle'), ls = R.LineStyle; end
                    col = refPalette(nLines + j, :);

                    % Dim the reference line relative to the main curves
                    yline(ax, R.Y, ls, ...
                        'Color',     col * 0.75, ...
                        'LineWidth', 1.0,         ...
                        'DisplayName', R.Name,    ...
                        'HandleVisibility', 'on');
                end
            end

            % ── 3. Decoration ──────────────────────────────────────────────
            title(ax,  char(options.Title),  'Color', C.Text, 'FontSize', 12);
            xlabel(ax, char(options.XLabel), 'Color', C.Text);
            ylabel(ax, char(options.YLabel), 'Color', C.Text);

            if isfield(options, 'ShowLegend') && options.ShowLegend
                leg = legend(ax, ...
                    'TextColor',  C.Text,       ...
                    'Color',      C.Background, ...
                    'EdgeColor',  C.Grid,       ...
                    'Location',   'northeast');
                leg.FontSize = 9;
            end

            hold(ax, 'off');
        end
    end
end
