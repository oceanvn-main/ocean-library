classdef PrototypeClassifierStrategy < OceanViz.Strategies.BaseStrategy
    % PROTOTYPECLASSIFIERSTRATEGY  Visualizes prototype-based classifiers.
    %
    % Renders the classic ESL-style plot: stippled decision regions (tiny dots),
    % data points (open circles), prototypes (filled circles with dark edge),
    % decision boundary contours, and optional Bayes boundary overlay.
    %
    % DATA CONTRACT:
    %   data.X           - (N x 2) data points
    %   data.Y           - (N x 1) class labels (1..K)
    %   data.Prototypes  - (M x 2) prototype locations
    %   data.ProtoLabels - (M x 1) prototype class labels
    %   data.Grid        - struct with .xx, .yy (meshgrid), .zz (class on grid)
    %   data.BayesGrid   - (optional) struct with .xx, .yy, .zz for Bayes boundary
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
            classes = unique(data.Y);
            K = numel(classes);
            palette = obj.Parent.getColorSequence(K);

            xx = data.Grid.xx;
            yy = data.Grid.yy;
            zz = data.Grid.zz;

            % Dot sizes: read from data struct (options has a fixed allowlist in Core.m)
            ptSize    = 15;  if isfield(data, 'DataPointSize'),  ptSize    = data.DataPointSize;  end
            protoSize = 50;  if isfield(data, 'ProtoSize'),      protoSize = data.ProtoSize;      end
            stippleDensity = 60; if isfield(data, 'StippleDensity'), stippleDensity = data.StippleDensity; end

            step = max(1, floor(size(xx,1) / stippleDensity));
            xSub = xx(1:step:end, 1:step:end);
            ySub = yy(1:step:end, 1:step:end);
            zSub = zz(1:step:end, 1:step:end);

            for ki = 1:K
                mask = (zSub == classes(ki));
                scatter(ax, xSub(mask), ySub(mask), 3, ...
                    'MarkerFaceColor', palette(ki,:), ...
                    'MarkerEdgeColor', 'none', ...
                    'MarkerFaceAlpha', 0.45, ...
                    'HandleVisibility', 'off');
            end

            % --- 2. Decision boundary (solid, prominent) ---
            [~, hCont] = contour(ax, xx, yy, zz, K-1, ...
                'LineColor', colors.Text, ...
                'LineWidth', 2.0, ...
                'LineStyle', '-');
            set(hCont, 'HandleVisibility', 'off');

            % --- 3. Data points (open circles, colored by class) ---
            for ki = 1:K
                mask = (data.Y == classes(ki));
                scatter(ax, data.X(mask,1), data.X(mask,2), ptSize, ...
                    'MarkerEdgeColor', palette(ki,:), ...
                    'MarkerFaceColor', 'none', ...
                    'LineWidth', 0.9, ...
                    'HandleVisibility', 'off');
            end

            % --- 4. Prototypes (filled circles with dark ring) ---
            if protoSize > 0 && ~isempty(data.Prototypes)
                for ki = 1:K
                    mask = (data.ProtoLabels == classes(ki));
                    if any(mask)
                        scatter(ax, data.Prototypes(mask,1), data.Prototypes(mask,2), ...
                            protoSize, ...
                            'MarkerFaceColor', palette(ki,:), ...
                            'MarkerEdgeColor', [0 0 0], ...
                            'LineWidth', 1.8, ...
                            'DisplayName', sprintf('Class %d prototypes', classes(ki)));
                    end
                end
            end

            % --- 5. Bayes boundary (dashed purple) ---
            if isfield(data, 'BayesGrid') && ~isempty(data.BayesGrid)
                bxx = data.BayesGrid.xx;
                byy = data.BayesGrid.yy;
                bzz = data.BayesGrid.zz;

                bayesColor = [0.58, 0.22, 0.82];
                [~, hBayes] = contour(ax, bxx, byy, bzz, K-1, ...
                    'LineColor', bayesColor, ...
                    'LineWidth', 2.0, ...
                    'LineStyle', '-');
                set(hBayes, 'HandleVisibility', 'off');

                % Clean legend proxy
                plot(ax, NaN, NaN, '-', 'Color', bayesColor, ...
                    'LineWidth', 2.0, ...
                    'DisplayName', 'Bayes boundary');
            end

            % --- Axis styling ---
            set(ax, 'Color', colors.Background);
            ax.XColor = colors.Text;
            ax.YColor = colors.Text;
            ax.FontSize = 9;
            grid(ax, 'off');
            box(ax, 'on');
            axis(ax, 'equal');

            % --- Title ---
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

            % --- Legend ---
            if isfield(options, 'ShowLegend') && options.ShowLegend
                lg = legend(ax, 'Location', 'best', ...
                    'FontSize', 8, ...
                    'EdgeColor', [0.3 0.3 0.3], ...
                    'TextColor', colors.Text, ...
                    'Color', colors.Background);
            end

            hold(ax, 'off');
        end
    end
end
