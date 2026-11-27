classdef MultiComparison < handle
    % OCEANVIZ.STRATEGIES.MULTICOMPARISON
    % Plot N normalized model selection curves on shared axes.
    
    properties
        Parent % Injected Core (for Colors)
    end
    
    methods
        function plot(obj, ax, data, options)
            % VALIDATE INPUTS (Fixes 'Not enough input arguments' ambiguity)
            arguments
                obj
                ax
                data struct
                options struct
            end
            
            hold(ax, 'on');
            
            % 1. Get Colors (Scalable)
            numLines = length(data.Lines);
            palette = obj.Parent.getColorSequence(numLines);
            markers = {'o', 's', 'd', '^', 'v', 'p', 'h'};
            
            % 2. Plot Lines
            for i = 1:numLines
                L = data.Lines{i};
                col = palette(i, :);
                mark = markers{mod(i-1, length(markers)) + 1};
                
                plot(ax, data.X, L.Y, ...
                    'Color', col, ...
                    'LineWidth', 2, ...
                    'LineStyle', '-', ...
                    'Marker', mark, ...
                    'MarkerFaceColor', col, ...
                    'MarkerSize', 5, ...
                    'DisplayName', L.Name);
                    
                % Plot Pick Marker
                if isfield(L, 'PickX') && ~isempty(L.PickX)
                    xline(ax, L.PickX, '--', ...
                        'Color', col, ...
                        'LineWidth', 1.5, ...
                        'HandleVisibility', 'off');
                end
            end
            
            % 3. True Model Reference
            if isfield(data, 'TrueModelK')
                xline(ax, data.TrueModelK, '--w', ...
                    'LineWidth', 1, ...
                    'Label', "True (k="+data.TrueModelK+")", ...
                    'LabelVerticalAlignment', 'top', ...
                    'HandleVisibility', 'off');
            end
            
            % 4. Decoration (Robust Title)
            % Convert string to char just to be safe (supports older MATLAB versions too)
            title(ax, char(options.Title));
            xlabel(ax, char(options.XLabel));
            ylabel(ax, char(options.YLabel));
            
            if isfield(options, 'ShowLegend') && options.ShowLegend
                legend(ax, ...
                    'TextColor', obj.Parent.Colors.Text, ...
                    'Color',     obj.Parent.Colors.Background, ...
                    'EdgeColor', obj.Parent.Colors.Grid);
            end
        end
    end
end