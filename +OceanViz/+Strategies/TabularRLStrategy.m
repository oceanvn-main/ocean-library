classdef TabularRLStrategy < OceanViz.Interfaces.IVizStrategy
    % TABULARRLSTRATEGY Concrete OceanViz strategy for Tabular RL visualization.
    %
    % NAMESPACE: OceanViz.Strategies
    % FILE LOCATION: +OceanViz/+Strategies/TabularRLStrategy.m
    %
    % Renders high-contrast 3D surface topographies of state value functions V(s)
    % and 2D heatmaps of optimal policy decision boundaries pi(s).
    
    methods
        function draw(obj, ax, data, options, theme)
            % DRAW Implementation of IVizStrategy contract.
            
            mode = "Combined";
            if isfield(options, 'Mode') && ~isempty(options.Mode)
                mode = string(options.Mode);
            end
            
            switch mode
                case "ValueSurface"
                    obj.render_value_surface_(ax, data, options, theme);
                case "PolicyHeatmap"
                    obj.render_policy_heatmap_(ax, data, options, theme);
                otherwise % "Combined" or multi-panel fallback
                    obj.render_value_surface_(ax, data, options, theme);
            end
        end
    end
    
    methods (Access = private)
        function render_value_surface_(~, ax, data, options, theme)
            V = data.ValueMatrix;
            grid_size = size(V, 1);
            cars_range = 0:(grid_size - 1);
            
            [X, Y] = meshgrid(cars_range, cars_range);
            
            hold(ax, 'on');
            % V is indexed V(n1+1, n2+1) where n1 is row (y-axis) and n2 is col (x-axis)
            s = surf(ax, X, Y, V', 'EdgeColor', [0.2 0.2 0.2], 'EdgeAlpha', 0.5, 'FaceAlpha', 0.95);
            s.DisplayName = 'State Value V(s)';
            
            colormap(ax, turbo);
            c = colorbar(ax);
            c.Label.String = 'Expected Return V(s)';
            c.Label.Color = theme.Text;
            c.Color = theme.Text;
            
            view(ax, [-35, 30]);
            grid(ax, 'on');
            set(ax, 'Color', theme.Background, 'XColor', theme.Text, 'YColor', theme.Text, 'ZColor', theme.Text);
            
            xlabel(ax, '# Cars at second location', 'Color', theme.Text);
            ylabel(ax, '# Cars at first location', 'Color', theme.Text);
            zlabel(ax, 'Optimal Value V(s)', 'Color', theme.Text);
            
            title_str = 'Optimal State-Value Function V(s)';
            if isfield(options, 'Title') && ~isempty(options.Title)
                title_str = options.Title;
            end
            title(ax, title_str, 'Color', theme.Text, 'FontSize', 12, 'FontWeight', 'bold');
            
            legend(ax, 'Location', 'northeast', 'TextColor', theme.Text, 'Color', theme.Background, 'EdgeColor', 'none');
            hold(ax, 'off');
        end
        
        function render_policy_heatmap_(~, ax, data, options, theme)
            pi_mat = data.PolicyMatrix;
            grid_size = size(pi_mat, 1);
            cars_range = 0:(grid_size - 1);
            
            hold(ax, 'on');
            % Display policy heatmap where x-axis is location 2, y-axis is location 1
            imagesc(ax, cars_range, cars_range, pi_mat');
            set(ax, 'YDir', 'normal');
            
            % Overlay contour lines to highlight decision boundaries
            [C, hc] = contour(ax, cars_range, cars_range, pi_mat', -5:5, 'k-', 'LineWidth', 1.2);
            if ~isempty(C) && size(C, 1) == 2
                try
                    clabel(ax, C, hc, 'Color', [1 1 1], 'FontSize', 9, 'FontWeight', 'bold');
                catch
                end
            end
            
            colormap(ax, jet);
            c = colorbar(ax);
            c.Label.String = 'Cars Moved (1 \rightarrow 2)';
            c.Label.Color = theme.Text;
            c.Color = theme.Text;
            c.Ticks = min(pi_mat(:)):max(pi_mat(:));
            
            grid(ax, 'on');
            set(ax, 'Color', theme.Background, 'XColor', theme.Text, 'YColor', theme.Text);
            
            xlabel(ax, '# Cars at second location', 'Color', theme.Text);
            ylabel(ax, '# Cars at first location', 'Color', theme.Text);
            
            title_str = 'Optimal Decision Policy \pi(s)';
            if isfield(options, 'Title') && ~isempty(options.Title)
                title_str = options.Title;
            end
            title(ax, title_str, 'Color', theme.Text, 'FontSize', 12, 'FontWeight', 'bold');
            
            hold(ax, 'off');
        end
    end
end
