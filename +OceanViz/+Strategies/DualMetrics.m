classdef DualMetrics < handle
    % OCEANVIZ.STRATEGIES.DUALMETRICS
    % Strategy for comparative analysis of two metrics using dual Y-axes.
    %
    % OVERVIEW:
    %   This strategy is optimized for visualizing the "Harmony" or "Conflict" 
    %   between different model selection criteria (e.g., Mallows' Cp vs. AIC).
    %   It utilizes dual axes to handle metrics with different units and scales.
    %
    % DESIGN PRINCIPLES:
    %   - Dependency Injection: Uses a 'Parent' property to pull theme-specific 
    %     configuration from the central Core engine.
    %   - Visual Rigor: Enforces LaTeX interpretation for all labels to maintain 
    %     academic/professional standards.
    %   - Scalability: Handles automatic color mapping for primary and secondary 
    %     metrics based on the active JSON theme.
    
    properties
        Parent % Reference to the OceanViz.Core engine (Injected at runtime)
    end
    
    methods
        function plot(obj, ax, data, options)
            % PLOT Executes the dual-axis rendering pipeline.
            %
            % Inputs:
            %   obj     : Strategy instance.
            %   ax      : Target Axes handle.
            %   data    : Struct containing:
            %             - .X  : Vector for the independent variable (e.g., Degree).
            %             - .Y1 : Vector for the primary metric.
            %             - .Y2 : Vector for the secondary metric.
            %   options : Struct containing labels (Title, YLabel, YLabelRight).
            
            arguments
                obj
                ax (1,1) matlab.graphics.axis.Axes
                data (1,1) struct
                options struct 
            end
            
            % 1. Understand the Problem: Context Retrieval
            % Extract theme colors from the Parent engine. This ensures the 
            % strategy is decoupled from specific color values.
            themeColors = obj.Parent.Colors; 
            
            % 2. Establish Primary Manifold (Left Axis)
            % Typically used for AIC or the dominant criterion.
            yyaxis(ax, 'left');
            plot(ax, data.X, data.Y1, '-o', ...
                'LineWidth', 2, ...
                'Color', themeColors.Primary, ...
                'MarkerFaceColor', themeColors.Primary);
            
            ylabel(ax, options.YLabel, 'Interpreter', 'latex', 'Color', themeColors.Primary);
            ax.YColor = themeColors.Primary;
            
            % 3. Establish Secondary Manifold (Right Axis)
            % Typically used for Cp or RSS to check for asymptotic equivalence.
            yyaxis(ax, 'right');
            plot(ax, data.X, data.Y2, '--s', ...
                'LineWidth', 2, ...
                'Color', themeColors.Secondary, ...
                'MarkerFaceColor', themeColors.Secondary);
            
            ylabel(ax, options.YLabelRight, 'Interpreter', 'latex', 'Color', themeColors.Secondary);
            ax.YColor = themeColors.Secondary;
            
            % 4. Decoration and Rigor
            % Title and X-axis labels are unified under the theme's text color.
            title(ax, options.Title, 'Interpreter', 'latex', 'Color', themeColors.Text);
            xlabel(ax, options.XLabel, 'Interpreter', 'latex', 'Color', themeColors.Text);
            
            % Grid configuration
            grid(ax, 'on');
            ax.GridColor = themeColors.Grid;
            ax.GridAlpha = 0.4; % Subtle grid for scannability
            
            % Final visual cleanup
            set(ax, 'Color', 'none'); % Keep background transparent for Core control
            hold(ax, 'off');
        end
    end
end