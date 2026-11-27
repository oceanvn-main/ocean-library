classdef Distribution < handle
    % OCEANVIZ.STRATEGIES.DISTRIBUTION
    % Plot probability density functions (PDFs) with standardized styling.
    
    properties
        Parent % Injected Core (for Colors)
    end
    
    properties (Constant)
        ColorMLE   = [0.8500 0.3250 0.0980]; % Red-ish
        ColorBayes = [0.0000 0.4470 0.7410]; % Blue-ish
        ColorBoot  = [0.4660 0.6740 0.1880]; % Green-ish
        LineWidthRef = 2.0;
        LineWidth    = 1.5;
    end
    
    methods
        function plot(obj, ax, data, options)
            arguments
                obj, ax
                data struct % Requires .Distributions (Struct Array) with fields: X, Y, Type, Name
                options struct
            end
            
            hold(ax, 'on');
            
            dists = data.Distributions;
            for i = 1:length(dists)
                d = dists(i);
                
                % Determine Style based on Type
                switch d.Type
                    case 'MLE'
                        col = obj.ColorMLE;
                        style = '-';
                        width = obj.LineWidthRef;
                    case 'Bayes'
                        col = obj.ColorBayes;
                        style = '--';
                        width = obj.LineWidth;
                    case 'Boot'
                        col = obj.ColorBoot;
                        style = ':';
                        width = obj.LineWidth;
                    otherwise
                        % Fallback / Cycle
                        col = [0.5 0.5 0.5];
                        style = '-';
                        width = 1.0;
                end
                
                plot(ax, d.X, d.Y, ...
                    'Color', col, ...
                    'LineStyle', style, ...
                    'LineWidth', width, ...
                    'DisplayName', d.Name);
            end
            
            % Decoration
            if isfield(options, 'Title'), title(ax, char(options.Title)); end
            if isfield(options, 'XLabel'), xlabel(ax, char(options.XLabel)); end
            if isfield(options, 'YLabel'), ylabel(ax, char(options.YLabel)); end
            
            if isfield(options, 'ShowLegend') && options.ShowLegend
                 legend(ax, 'Location', 'best', 'EdgeColor', 'none', ...
                        'TextColor', obj.Parent.Colors.Text);
            end
        end
    end
end
