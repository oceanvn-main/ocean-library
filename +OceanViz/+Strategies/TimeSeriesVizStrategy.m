classdef TimeSeriesVizStrategy < OceanViz.Strategies.BaseStrategy
    % TIMESERIESVIZSTRATEGY Renders a 3-panel time-series analysis dashboard.
    %
    % Layout:
    %   - Panel 1 (Top, full width): Time series realization y(t) vs t.
    %   - Panel 2 (Bottom Left): Empirical and theoretical ACF with +/- 1.96/sqrt(N) confidence bounds.
    %   - Panel 3 (Bottom Right): Empirical and theoretical PACF with +/- 1.96/sqrt(N) confidence bounds.
    %
    % Data Contract:
    %   data.Y              - (N x 1) double, the time series sequence
    %   data.X              - (N x 1) double, time steps (optional, defaults to 1:N)
    %   data.ACF            - (1 x maxLag+1) double, empirical ACF values
    %   data.PACF           - (1 x maxLag) double, empirical PACF values
    %   data.TheoreticalACF - (1 x maxLag+1) double, theoretical ACF values (optional)
    %   data.TheoreticalPACF- (1 x maxLag) double, theoretical PACF values (optional)
    %   data.Lags           - (1, 1) double, maximum lag plotted
    %   data.ModelName      - (string) model/process identifier (optional)
    %
    % See also: OceanViz.Core, OceanMath.TimeSeries.Models.ARMA
    
    methods
        function plot(obj, ax, data, options)
            arguments
                obj
                ax
                data    struct
                options struct
            end
            
            colors = obj.Parent.Colors;
            N = length(data.Y);
            maxLag = data.Lags;
            
            % Setup Time indices
            if isfield(data, 'X')
                t_idx = data.X;
            else
                t_idx = (1:N)';
            end
            
            % Locate parent container (figure or panel) and replace ax with tiledlayout
            parent = ax.Parent;
            delete(ax);
            
            % Create a 2x2 grid layout
            layout = tiledlayout(parent, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
            
            % 1. Time Series Plot (Spanning Top Row)
            axTime = nexttile(layout, [1, 2]);
            obj.Parent.styleAxes(axTime);
            hold(axTime, 'on');
            
            plot(axTime, t_idx, data.Y, 'Color', colors.Primary, 'LineWidth', 1.2, 'DisplayName', 'Realization');
            
            titleStr = "Time Series Sequence";
            if isfield(data, 'ModelName') && data.ModelName ~= ""
                titleStr = titleStr + " — " + data.ModelName;
            end
            title(axTime, titleStr, 'Color', colors.Text, 'FontSize', 11, 'FontWeight', 'bold');
            xlabel(axTime, "Time Index (t)", 'Color', colors.Text);
            ylabel(axTime, "Y(t)", 'Color', colors.Text);
            hold(axTime, 'off');
            
            % Confidence bound for White Noise: 1.96 / sqrt(N)
            confBound = 1.96 / sqrt(N);
            
            % 2. ACF Plot (Bottom-Left)
            axACF = nexttile(layout);
            obj.Parent.styleAxes(axACF);
            hold(axACF, 'on');
            
            % Draw confidence bounds as shaded area
            fill(axACF, [0, maxLag, maxLag, 0], [confBound, confBound, -confBound, -confBound], ...
                colors.Secondary, 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            
            % Draw empirical ACF stems
            stem(axACF, 0:maxLag, data.ACF, 'Filled', 'Color', colors.Primary, ...
                'MarkerFaceColor', colors.Primary, 'MarkerSize', 4, 'LineWidth', 1.2, 'DisplayName', 'Empirical');
            
            % Draw theoretical ACF if provided
            if isfield(data, 'TheoreticalACF') && ~isempty(data.TheoreticalACF)
                plot(axACF, 0:maxLag, data.TheoreticalACF, 'o--', 'Color', colors.Boundary, ...
                    'LineWidth', 1.2, 'MarkerSize', 4, 'DisplayName', 'Theoretical');
            end
            
            % Draw zero reference line
            yline(axACF, 0, 'Color', colors.Text * 0.5, 'LineWidth', 0.8, 'HandleVisibility', 'off');
            
            title(axACF, "Autocorrelation Function (ACF)", 'Color', colors.Text, 'FontSize', 10, 'FontWeight', 'bold');
            xlabel(axACF, "Lag (k)", 'Color', colors.Text);
            ylabel(axACF, "ACF (\rho_k)", 'Color', colors.Text);
            xlim(axACF, [-0.5, maxLag + 0.5]);
            ylim(axACF, [-1.05, 1.05]);
            
            if isfield(data, 'TheoreticalACF') && ~isempty(data.TheoreticalACF) && options.ShowLegend
                legend(axACF, 'Location', 'northeast', 'TextColor', colors.Text, 'EdgeColor', 'none', 'Color', 'none');
            end
            hold(axACF, 'off');
            
            % 3. PACF Plot (Bottom-Right)
            axPACF = nexttile(layout);
            obj.Parent.styleAxes(axPACF);
            hold(axPACF, 'on');
            
            % Draw confidence bounds as shaded area
            fill(axPACF, [1, maxLag, maxLag, 1], [confBound, confBound, -confBound, -confBound], ...
                colors.Secondary, 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            
            % Draw empirical PACF stems (starts at lag 1)
            stem(axPACF, 1:maxLag, data.PACF, 'Filled', 'Color', colors.Primary, ...
                'MarkerFaceColor', colors.Primary, 'MarkerSize', 4, 'LineWidth', 1.2, 'DisplayName', 'Empirical');
            
            % Draw theoretical PACF if provided
            if isfield(data, 'TheoreticalPACF') && ~isempty(data.TheoreticalPACF)
                plot(axPACF, 1:maxLag, data.TheoreticalPACF, 'o--', 'Color', colors.Boundary, ...
                    'LineWidth', 1.2, 'MarkerSize', 4, 'DisplayName', 'Theoretical');
            end
            
            % Draw zero reference line
            yline(axPACF, 0, 'Color', colors.Text * 0.5, 'LineWidth', 0.8, 'HandleVisibility', 'off');
            
            title(axPACF, "Partial Autocorrelation Function (PACF)", 'Color', colors.Text, 'FontSize', 10, 'FontWeight', 'bold');
            xlabel(axPACF, "Lag (k)", 'Color', colors.Text);
            ylabel(axPACF, "PACF (\phi_{kk})", 'Color', colors.Text);
            xlim(axPACF, [0.5, maxLag + 0.5]);
            ylim(axPACF, [-1.05, 1.05]);
            
            if isfield(data, 'TheoreticalPACF') && ~isempty(data.TheoreticalPACF) && options.ShowLegend
                legend(axPACF, 'Location', 'northeast', 'TextColor', colors.Text, 'EdgeColor', 'none', 'Color', 'none');
            end
            hold(axPACF, 'off');
        end
    end
end
