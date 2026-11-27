classdef BarGroup < handle
    % OCEANVIZ.STRATEGIES.BARGROUP
    % Strategy for plotting Scalable Grouped Bar Charts.
    %
    % OVERVIEW:
    %   This class handles the visualization of grouped data (e.g., comparing 
    %   changes in AIC vs Cp). It is designed to be "Scalable," meaning it 
    %   automatically requests the correct number of colors from the Core 
    %   engine based on how many groups (columns) are in the data.
    %
    % DESIGN PATTERN:
    %   - Strategy Pattern: Implements the plotting logic for 'BarGroup'.
    %   - Dependency Injection: The 'Core' injects itself into 'obj.Parent',
    %     allowing this strategy to access the central Theme and Color Engine.
    
    properties 
        % Reference to the central OceanViz.Core object.
        % This is set automatically by Core.plot() before calling plot().
        % Allows access to: obj.Parent.Colors, obj.Parent.getColorSequence()
        Parent 
    end
    
    methods
        function plot(obj, ax, data, options)
            % PLOT Executes the bar chart visualization.
            %
            % Inputs:
            %   obj     : The strategy instance itself.
            %   ax      : The target Axes handle to draw on.
            %   data    : Struct containing the plot data:
            %             - data.X : Vector of X-axis categories/indices.
            %             - data.Y : Matrix [Rows x Cols]. Each column is a group.
            %             - data.Names : (Optional) String array for Legend.
            %   options : Struct containing decoration settings (Title, XLabel, etc.)
            
            % 1. Input Validation
            arguments
                obj
                ax
                data struct
                options struct
            end
            
            hold(ax, 'on');
            
            % --- 2. Create Base Plot ---
            % We create a standard grouped bar chart. MATLAB returns an array
            % of Bar objects 'b', one for each column in data.Y.
            b = bar(ax, data.X, data.Y, 'grouped');
            
            % --- 3. Determine Scale & Color ---
            % How many distinct groups (columns) did we plot?
            numGroups = length(b);
            
            % Ask the Core for a robust color palette of exactly this size.
            % If numGroups=2, it gives Primary + Secondary.
            % If numGroups=5, it generates 5 distinct high-contrast colors.
            palette = obj.Parent.getColorSequence(numGroups);
            
            % --- 4. Apply Styles Loop ---
            for i = 1:numGroups
                % Apply color from our generated palette
                b(i).FaceColor = palette(i, :);
                b(i).EdgeColor = 'none';
                
                % Set Legend Label (if provided in data.Names)
                % We check 'isfield' to prevent errors if names are missing.
                if isfield(data, 'Names') && i <= length(data.Names)
                    b(i).DisplayName = data.Names(i);
                end
            end
            
            % --- 5. Zero Threshold Line ---
            % Adds a subtle horizontal line at Y=0 to distinguish positive/negative.
            % Uses the Theme's text color for visibility in Dark/Light modes.
            yline(ax, 0, 'Color', obj.Parent.Colors.Text, ...
                'LineWidth', 0.5, 'HandleVisibility', 'off'); % 'off' hides it from Legend
            
            % --- 6. Standard Decorations ---
            % Apply Title and Labels from the 'options' struct passed by Core.
            title(ax, options.Title);
            xlabel(ax, options.XLabel);
            ylabel(ax, options.YLabel);
            
            % Configure Legend
            % --- Block 7: Updated BarGroup.m Legend Location ---
            % Decoration
            title(ax, options.Title);
            xlabel(ax, options.XLabel);
            ylabel(ax, options.YLabel);
            
            if isfield(options, 'ShowLegend') && options.ShowLegend && isfield(data, 'Names')
                legend(ax, data.Names, ...
                    'Location', 'best', ...
                    'TextColor', obj.Parent.Colors.Text, ...
                    'EdgeColor', 'none', ...
                    'Color', 'none');
            end
            % Style axes (Grid/Background)
            grid(ax, 'on');
            ax.GridColor = obj.Parent.Colors.Grid;
        end
    end
end