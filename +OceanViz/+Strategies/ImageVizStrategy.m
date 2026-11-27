classdef ImageVizStrategy < OceanViz.Strategies.BaseStrategy
    % IMAGEVIZSTRATEGY  OceanViz strategy for 2D images, heatmaps, and RGB tensors.
    %
    %   Data Contract:
    %     data.Image  or data.Data — (H x W) 2D array or (H x W x 3/4) 3D array
    %     data.Title               — string title (optional)
    %     data.Colormap            — colormap name (optional, defaults to 'viridis' or 'gray')
    %
    %   Author: Ocean Library  |  2026-08-05
    
    methods
        function plot(obj, ax, data, options)
            arguments
                obj
                ax
                data    struct
                options struct
            end
            
            colors = obj.Parent.Colors;
            
            % Extract image matrix
            if isfield(data, 'Image')
                I = data.Image;
            elseif isfield(data, 'Data')
                I = data.Data;
            else
                error('ImageVizStrategy:MissingData', 'Data struct must contain .Image or .Data.');
            end
            
            hold(ax, 'off');
            
            % Normalize datatype range for double/single if in [0, 1] or arbitrary
            nd = ndims(I);
            sz = size(I);
            
            if nd == 2
                % Grayscale / 2D Heatmap
                I_double = double(I);
                imagesc(ax, I_double);
                axis(ax, 'image');
                axis(ax, 'off');
                
                % Colormap configuration
                if isfield(data, 'Colormap')
                    colormap(ax, data.Colormap);
                else
                    colormap(ax, 'gray');
                end
                
                if isfield(options, 'ShowColorbar') && options.ShowColorbar
                    cb = colorbar(ax);
                    cb.Color = colors.Text;
                end
                
            elseif nd == 3 && (sz(3) == 3 || sz(3) == 4)
                % RGB / RGBA Image
                I_rgb = double(I);
                if max(I_rgb(:)) > 1.0 && max(I_rgb(:)) <= 255.0
                    I_rgb = I_rgb / 255.0;
                end
                I_rgb = max(0.0, min(1.0, I_rgb));
                
                imshow(I_rgb, 'Parent', ax);
                axis(ax, 'image');
                axis(ax, 'off');
            else
                % Generic 2D matrix
                imagesc(ax, double(I));
                axis(ax, 'image');
                colormap(ax, 'parula');
            end
            
            % Styling title
            titleStr = "";
            if isfield(data, 'Title') && data.Title ~= ""
                titleStr = data.Title;
            elseif isfield(options, 'Title') && options.Title ~= "" && options.Title ~= "Analysis"
                titleStr = options.Title;
            end
            
            if titleStr ~= ""
                title(ax, titleStr, 'Color', colors.Text, 'FontSize', 10, 'FontWeight', 'bold');
            end
        end
    end
end
