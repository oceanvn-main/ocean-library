classdef SOMGridStrategy < OceanViz.Strategies.BaseStrategy
    % SOMGRIDSTRATEGY  Renders a 2D SOM grid with data assignments.
    %
    % Replicates the classic Kohonen/ESL Figure 14.16 layout:
    %   - Each prototype is drawn as a circle at its integer grid position.
    %   - Data points assigned to each prototype are plotted as colored dots
    %     jittered randomly inside the corresponding circle.
    %   - Multiple panels (e.g. initial vs final) are supported.
    %
    % DATA CONTRACT:
    %   data.GridSize       - [Q1, Q2] grid dimensions
    %   data.Labels         - (N x 1) BMU assignment (1..M) for each sample
    %   data.ClassLabels    - (N x 1) true class labels for coloring
    %   data.NumClasses     - (1 x 1) number of distinct classes (optional, auto-detected)
    %   data.CircleRadius   - (1 x 1) radius of prototype circles (optional, default 0.4)
    %   data.JitterScale    - (1 x 1) jitter radius inside circle (optional, default 0.3)
    %   data.PanelTitle     - (string)  title for this panel (optional)
    %
    % USAGE (via OceanViz.Core):
    %   viz = OceanViz.Core('Theme', 'Dark');
    %   viz.plot(ax, 'SOMGrid', data, 'Title', 'SOM Assignments');
    %
    % See also: OceanViz.Core, OceanStats.Unsupervised.SOM
    
    methods
        function plot(obj, ax, data, options)
            arguments
                obj, ax
                data    struct
                options struct
            end
            
            hold(ax, 'on');
            
            colors = obj.Parent.Colors;
            
            Q1 = data.GridSize(1);
            Q2 = data.GridSize(2);
            M  = Q1 * Q2;
            
            % Grid coordinates (same convention as SOM.getGridCoords)
            [gy, gx] = ndgrid(1:Q1, 1:Q2);
            gx = gx(:);
            gy = gy(:);
            
            % Circle / jitter params
            circR = 0.4;
            if isfield(data, 'CircleRadius'), circR = data.CircleRadius; end
            jitterR = 0.3;
            if isfield(data, 'JitterScale'), jitterR = data.JitterScale; end
            
            % Class info
            classLabels = data.ClassLabels;
            classes = unique(classLabels);
            K = numel(classes);
            if isfield(data, 'NumClasses'), K = data.NumClasses; end
            palette = obj.Parent.getColorSequence(K);
            
            % --- Draw prototype circles ---
            theta = linspace(0, 2*pi, 64);
            for m = 1:M
                cx = gx(m) + circR * cos(theta);
                cy = gy(m) + circR * sin(theta);
                plot(ax, cx, cy, '-', ...
                    'Color', colors.Text * 0.6, ...
                    'LineWidth', 1.0, ...
                    'HandleVisibility', 'off');
            end
            
            % --- Plot jittered data points inside circles ---
            labels = data.Labels;
            legendDrawn = false(K, 1);
            
            for i = 1:numel(labels)
                bmu = labels(i);
                cls = classLabels(i);
                clsIdx = find(classes == cls, 1);
                
                % Random jitter inside circle (polar coordinates)
                r_jit = jitterR * sqrt(rand());
                a_jit = 2 * pi * rand();
                px = gx(bmu) + r_jit * cos(a_jit);
                py = gy(bmu) + r_jit * sin(a_jit);
                
                if ~legendDrawn(clsIdx)
                    scatter(ax, px, py, 18, ...
                        'MarkerFaceColor', palette(clsIdx, :), ...
                        'MarkerEdgeColor', 'none', ...
                        'MarkerFaceAlpha', 0.9, ...
                        'DisplayName', sprintf('Class %d', cls));
                    legendDrawn(clsIdx) = true;
                else
                    scatter(ax, px, py, 18, ...
                        'MarkerFaceColor', palette(clsIdx, :), ...
                        'MarkerEdgeColor', 'none', ...
                        'MarkerFaceAlpha', 0.9, ...
                        'HandleVisibility', 'off');
                end
            end
            
            % --- Axis styling ---
            set(ax, 'Color', colors.Background);
            ax.XColor = colors.Text;
            ax.YColor = colors.Text;
            ax.FontSize = 9;
            grid(ax, 'off');
            box(ax, 'on');
            axis(ax, 'equal');
            xlim(ax, [0.5 - circR, Q2 + 0.5 + circR]);
            ylim(ax, [0.5 - circR, Q1 + 0.5 + circR]);
            set(ax, 'XTick', 1:Q2, 'YTick', 1:Q1);
            
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
