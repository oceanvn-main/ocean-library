classdef Regression < OceanViz.Strategies.BaseStrategy
    methods
        function plot(obj, ax, data, options)
            arguments
                obj, ax, data
                options struct
            end
            
            colors = obj.Parent.Colors;
            
            scatter(ax, data.X, data.Y, 40, 'o', ...
                'MarkerEdgeColor', 'none', ...
                'MarkerFaceColor', colors.Scatter, ...
                'MarkerFaceAlpha', 0.6);
            hold(ax, 'on');
            
            if isfield(data, 'Prediction') && ~isempty(data.Prediction)
                [X_sort, idx] = sort(data.X);
                plot(ax, X_sort, data.Prediction(idx), '-', ...
                    'Color', colors.Secondary, 'LineWidth', 2.5);
            end
            
            xlabel(ax, options.XLabel, 'Interpreter', 'latex');
            ylabel(ax, options.YLabel, 'Interpreter', 'latex');
            title(ax, options.Title, 'Interpreter', 'latex', 'Color', colors.Text);
            hold(ax, 'off');
        end
    end
end