classdef SpeechContourStrategy < OceanViz.Strategies.BaseStrategy
    % SPEECHCONTOURSTRATEGY  One or more named line series over a common x axis (time,
    %   frequency or training step), with optional shaded intervals and reference lines.
    %
    %   viz.plot(ax, "SpeechContour", data)
    %
    %   DATA CONTRACT:
    %     data.X          1 x T x values (e.g. frame times in s, bin frequencies in Hz, steps)
    %     data.Series     1 x S struct array: Name (string), Y (1 x T), optional LineStyle,
    %                     optional Marker; NaN values are gaps (not drawn)
    %     data.XLabel     x-axis label (e.g. "Time (s)")
    %     data.YLabel     y-axis label (with unit)
    %     data.Intervals  (optional) n x 2 [start end] in x units, shaded (e.g. detected
    %                     speech segments); data.IntervalName names them in the legend
    %     data.RefLines   (optional) struct array Name, Y: horizontal dashed lines
    %                     (e.g. a detection threshold)
    %     data.XLog       (optional) logical, logarithmic x axis
    %     data.Title      (optional)
    %   The strategy draws exactly the given values (no smoothing or rescaling).
    %   Legend: every series, the intervals and the reference lines.

    methods
        function plot(obj, ax, data, options)
            arguments
                obj
                ax
                data struct
                options struct = struct()
            end
            U = OceanViz.Strategies.Speech.SpeechVizUtils;
            U.requireFields(data, ["X", "Series", "XLabel", "YLabel"], "SpeechContour");
            c = U.colors(obj);
            x = reshape(double(data.X), 1, []);
            S = numel(data.Series);
            pal = obj.Parent.getColorSequence(max(S, 1));
            hold(ax, 'on');
            allY = [];
            for s = 1:S
                y = reshape(double(data.Series(s).Y), 1, []);
                if numel(y) ~= numel(x)
                    error('OceanViz:Strategies:Speech:SpeechContourStrategy:Length', ...
                        'Series "%s" has %d values for %d x values.', data.Series(s).Name, numel(y), numel(x));
                end
                allY = [allY, y(isfinite(y))]; %#ok<AGROW>
            end
            if isfield(data, 'Intervals') && ~isempty(data.Intervals)
                iv = double(data.Intervals);
                if isempty(allY), lo = 0; hi = 1; else, lo = min(allY); hi = max(allY); end
                if hi <= lo, hi = lo + 1; end
                name = string(U.valueOr(data, 'IntervalName', "Interval"));
                for i = 1:size(iv, 1)
                    p = patch(ax, iv(i, [1 2 2 1]), [lo lo hi hi], c.Boundary, 'FaceAlpha', 0.18, ...
                        'EdgeColor', 'none', 'DisplayName', name);
                    if i > 1, p.HandleVisibility = 'off'; end
                end
            end
            for s = 1:S
                d = data.Series(s);
                ls = '-'; mk = 'none';
                if isfield(d, 'LineStyle') && ~isempty(d.LineStyle), ls = char(d.LineStyle); end
                if isfield(d, 'Marker') && ~isempty(d.Marker), mk = char(d.Marker); end
                plot(ax, x, reshape(double(d.Y), 1, []), 'Color', pal(s, :), 'LineWidth', 1.3, ...
                    'LineStyle', ls, 'Marker', mk, 'MarkerSize', 4, 'DisplayName', string(d.Name));
            end
            if isfield(data, 'RefLines') && ~isempty(data.RefLines)
                for r = 1:numel(data.RefLines)
                    yline(ax, double(data.RefLines(r).Y), '--', 'Color', c.Text, 'LineWidth', 1, ...
                        'DisplayName', string(data.RefLines(r).Name));
                end
            end
            hold(ax, 'off');
            if isfield(data, 'XLog') && data.XLog, ax.XScale = 'log'; end
            if numel(x) > 1 && all(isfinite(x([1 end]))) && x(end) > x(1)
                xlim(ax, [x(1), x(end)]);
            end
            U.label(ax, c, U.pickTitle(options, data, "Contour"), string(data.XLabel), string(data.YLabel));
            if ~isfield(options, 'ShowLegend') || options.ShowLegend
                U.legend(ax, c, 'Location', 'best');
            end
        end
    end
end
