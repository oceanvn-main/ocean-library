classdef FunctionalDataStrategy < OceanViz.Strategies.BaseStrategy
    % FUNCTIONALDATASTRATEGY  Overlaid curves grouped by class/label.
    %
    % Generic visualization for functional data: spectra, time series,
    % waveforms, log-periodograms, etc.  Each group gets its own color
    % and (optionally) a bold group mean curve.
    %
    % DATA CONTRACT:
    %   data.X          - (N x P) matrix of curves (each row is one curve)
    %   data.Y          - (N x 1) group labels (numeric or categorical)
    %   data.xAxis      - (optional, 1 x P) domain axis (e.g. frequency)
    %   data.classNames - (optional, cell/string) human-readable label names
    %   data.ShowMean   - (optional, logical) overlay group mean (default true)
    %   data.BaseColor  - (optional, 1x3 RGB) force a single color for all curves
    %   data.NumPaletteColors - (optional, int) request N-color palette and use
    %                           the group index to pick (for multi-panel layouts)
    %   data.ShowQuantiles - (optional, logical) show 10th/90th percentile band
    %
    % See also: OceanViz.Core, OceanViz.Strategies.BaseStrategy

    methods
        function plot(obj, ax, data, options)
            arguments
                obj, ax
                data   struct
                options struct
            end

            hold(ax, 'on');
            colors = obj.Parent.Colors;

            X = data.X;
            Y = data.Y;
            [~, P] = size(X);

            % Domain axis
            if isfield(data, 'xAxis') && ~isempty(data.xAxis)
                xAx = data.xAxis(:)';
            else
                xAx = 1:P;
            end

            % Groups
            classes = unique(Y);
            K = numel(classes);

            % Color resolution: BaseColor > NumPaletteColors > auto
            if isfield(data, 'BaseColor') && ~isempty(data.BaseColor)
                palette = repmat(data.BaseColor, K, 1);
            elseif isfield(data, 'NumPaletteColors') && data.NumPaletteColors > 0
                fullPalette = obj.Parent.getColorSequence(data.NumPaletteColors);
                palette = zeros(K, 3);
                for ki = 1:K
                    idx = mod(classes(ki) - 1, data.NumPaletteColors) + 1;
                    palette(ki, :) = fullPalette(idx, :);
                end
            else
                palette = obj.Parent.getColorSequence(K);
            end

            % Show options
            showMean = true;
            if isfield(data, 'ShowMean'), showMean = data.ShowMean; end
            showQuantiles = false;
            if isfield(data, 'ShowQuantiles'), showQuantiles = data.ShowQuantiles; end

            % Class names
            if isfield(data, 'classNames') && ~isempty(data.classNames)
                names = string(data.classNames);
            else
                names = arrayfun(@(c) sprintf('Class %d', c), classes, ...
                    'UniformOutput', false);
                names = string(names);
            end

            % --- Plot each group ---
            for ki = 1:K
                mask = (Y == classes(ki));
                Xg = X(mask, :);
                nCurves = size(Xg, 1);
                col = palette(ki, :);

                % Individual curves (thin, very transparent)
                curveAlpha = min(0.25, max(0.02, 10 / nCurves));
                for ci = 1:nCurves
                    plot(ax, xAx, Xg(ci, :), ...
                        'Color', [col, curveAlpha], ...
                        'LineWidth', 0.3, ...
                        'HandleVisibility', 'off');
                end

                % Quantile band (10th–90th percentile shaded region)
                if showQuantiles && nCurves > 5
                    q10 = prctile(Xg, 10, 1);
                    q90 = prctile(Xg, 90, 1);
                    fill(ax, [xAx, fliplr(xAx)], [q10, fliplr(q90)], ...
                        col, 'FaceAlpha', 0.12, 'EdgeColor', 'none', ...
                        'HandleVisibility', 'off');
                end

                % Mean curve (bold, with dark outline for contrast)
                if showMean
                    meanCurve = mean(Xg, 1);
                    % Dark outline for readability
                    plot(ax, xAx, meanCurve, ...
                        'Color', [0 0 0], 'LineWidth', 3.2, ...
                        'HandleVisibility', 'off');
                    % Colored line on top
                    plot(ax, xAx, meanCurve, ...
                        'Color', col, 'LineWidth', 2.2, ...
                        'DisplayName', sprintf('%s (n=%d)', names(ki), nCurves));
                end
            end

            % --- Styling ---
            set(ax, 'Color', colors.Background);
            ax.XColor = colors.Text;
            ax.YColor = colors.Text;
            ax.FontSize = 9;
            box(ax, 'on');

            % Title / Labels
            if isfield(options, 'Title')
                title(ax, char(options.Title), ...
                    'Color', colors.Text, 'FontSize', 12, 'FontWeight', 'bold');
            end
            if isfield(options, 'XLabel')
                xlabel(ax, char(options.XLabel), 'Color', colors.Text);
            end
            if isfield(options, 'YLabel')
                ylabel(ax, char(options.YLabel), 'Color', colors.Text);
            end

            % Legend
            if isfield(options, 'ShowLegend') && options.ShowLegend
                legend(ax, 'Location', 'best', 'FontSize', 8, ...
                    'EdgeColor', [0.3 0.3 0.3], ...
                    'TextColor', colors.Text, ...
                    'Color', [colors.Background, 0.85]);
            end

            hold(ax, 'off');
        end
    end
end
