classdef ACFStrategy < OceanViz.Strategies.BaseStrategy
    % ACFSTRATEGY Premium OceanViz strategy for Autocorrelation (ACF) & PACF plots.
    %
    % Renders a publication-quality ACF stem plot with a semi-transparent 95%
    % confidence ribbon, zero-reference baseline, distinct stem markers, and 
    % highlighted statistically significant lags.
    %
    % Data Contract:
    %   data.ACF       - (1 x maxLag+1) double, ACF values
    %   data.Lags      - (1 x maxLag+1) double, lag indices (0..maxLag)
    %   data.ConfBound - (1,1) double, 95% confidence bound (1.96 / sqrt(N))
    %   data.Title     - (string) optional custom plot title
    %   data.YLabel    - (string) optional custom Y-axis label
    %
    % See also: OceanViz.Core, OceanViz.Strategies.TimeSeriesVizStrategy
    
    methods
        function plot(obj, ax, data, options)
            arguments
                obj
                ax
                data    struct
                options struct
            end
            
            colors = obj.Parent.Colors;
            
            lags = data.Lags;
            acfVals = data.ACF;
            maxLag = max(lags);
            minLag = min(lags);
            
            if isfield(data, 'ConfBound')
                confBound = data.ConfBound;
            elseif isfield(data, 'N')
                confBound = 1.96 / sqrt(data.N);
            else
                confBound = 0.1;
            end
            
            hold(ax, 'on');
            
            % 1. Render translucent 95% confidence ribbon
            fill(ax, [minLag - 0.5, maxLag + 0.5, maxLag + 0.5, minLag - 0.5], ...
                 [confBound, confBound, -confBound, -confBound], ...
                 colors.Secondary, 'FaceAlpha', 0.15, 'EdgeColor', 'none', ...
                 'DisplayName', '95% Conf. Band');
            
            % 2. Render subtle upper/lower confidence boundary lines
            yline(ax, confBound, '--', 'Color', colors.Secondary * 0.85, 'LineWidth', 1.0, 'HandleVisibility', 'off');
            yline(ax, -confBound, '--', 'Color', colors.Secondary * 0.85, 'LineWidth', 1.0, 'HandleVisibility', 'off');
            
            % 3. Zero-reference baseline
            yline(ax, 0, 'Color', colors.Grid, 'LineWidth', 1.0, 'HandleVisibility', 'off');
            
            % 4. Identify significant vs non-significant lags
            sigIdx = abs(acfVals) > confBound & lags > 0;
            insigIdx = ~sigIdx;
            
            % 5. Render primary ACF stems
            stem(ax, lags(insigIdx), acfVals(insigIdx), 'Filled', ...
                 'Color', colors.Primary, 'MarkerFaceColor', colors.Primary, ...
                 'MarkerSize', 4.5, 'LineWidth', 1.2, 'DisplayName', 'ACF \rho_k');
            
            % 6. Highlight statistically significant lags with accent markers
            if any(sigIdx)
                stem(ax, lags(sigIdx), acfVals(sigIdx), 'Filled', ...
                     'Color', colors.Boundary, 'MarkerFaceColor', colors.Boundary, ...
                     'MarkerSize', 5.5, 'LineWidth', 1.4, 'DisplayName', 'Significant Lag (p < 0.05)');
            end
            
            % 7. Apply layout formatting & TeX text rendering
            titleStr = "Autocorrelation Function (ACF)";
            if isfield(data, 'Title') && data.Title ~= ""
                titleStr = data.Title;
            end
            
            yLabelStr = "ACF \rho_k";
            if isfield(data, 'YLabel') && data.YLabel ~= ""
                yLabelStr = data.YLabel;
            end
            
            title(ax, titleStr, 'Color', colors.Text, 'FontSize', 11, 'FontWeight', 'bold', 'Interpreter', 'tex');
            xlabel(ax, "Lag (k)", 'Color', colors.Text, 'FontSize', 10, 'Interpreter', 'tex');
            ylabel(ax, yLabelStr, 'Color', colors.Text, 'FontSize', 10, 'Interpreter', 'tex');
            
            xlim(ax, [minLag - 0.5, maxLag + 0.5]);
            ylim(ax, [-0.25, 1.08]);
            
            if options.ShowLegend
                legend(ax, 'Location', 'northeast', 'TextColor', colors.Text, ...
                       'EdgeColor', 'none', 'Color', 'none', 'FontSize', 9);
            end
            
            hold(ax, 'off');
        end
    end
end
