classdef Calibration < OceanViz.Strategies.BaseStrategy
    % CALIBRATION Strategy for Reliability Diagrams.
    %
    % This strategy visualizes the relationship between a model's predicted 
    % confidence and its observed frequency (the reality). Perfect calibration 
    % results in a series following the 45-degree diagonal.
    
    methods
        function plot(obj, ax, data, options)
            % PLOT Renders a reliability diagram.
            %
            % data struct:
            %   .bins        : Center values of probability bins (1 x N)
            %   .frequencies : Observed class frequencies (1 x N)
            %   .modelName   : Legend label (e.g., 'PolyMARS')
            %   .color       : RGB vector defining the line/marker color
            
            arguments
                obj, ax, data, options
            end
            
            colors = obj.Parent.Colors;
            hold(ax, 'on');
            
            % 1. Reference Line (Perfect Calibration)
            % STATE TRACKING: We only draw the dashed diagonal once per axis.
            % We store a flag in UserData to prevent stacking lines in multi-model plots.
            if ~isfield(ax.UserData, 'IsRefDrawn') || ~ax.UserData.IsRefDrawn
                plot(ax, [0 1], [0 1], '--', 'Color', colors.Grid, 'LineWidth', 1.5, ...
                    'HandleVisibility', 'off'); % Keeps the legend clean
                ax.UserData.IsRefDrawn = true;
            end
            
            % 2. Reliability Curve
            % We filter out 'NaN' frequencies (bins with no data points)
            valid = ~isnan(data.frequencies);
            
            plot(ax, data.bins(valid), data.frequencies(valid), '-o', ...
                'Color', data.color, 'LineWidth', 2.5, ...
                'MarkerFaceColor', data.color, 'MarkerSize', 8, ...
                'DisplayName', data.modelName);
            
            % 3. Analytical Decoration
            xlabel(ax, 'Predicted Probability', 'Interpreter', 'latex');
            ylabel(ax, 'Actual Frequency', 'Interpreter', 'latex');
            title(ax, options.Title, 'Interpreter', 'latex', 'Color', colors.Text);
            
            grid(ax, 'on');
            axis(ax, [0 1 0 1]); % Force standard probability scale
            
            hold(ax, 'off');
        end
    end
end
