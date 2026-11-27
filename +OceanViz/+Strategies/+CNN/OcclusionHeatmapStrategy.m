classdef OcclusionHeatmapStrategy < OceanViz.Strategies.BaseStrategy
    % OCCLUSIONHEATMAPSTRATEGY  Score-drop heatmap over the original image.
    %   "What happens if this region is removed?"
    %
    %   r = OcclusionSensitivity.run(engine, X, targetClass);
    %   strat.plot(ax, r, struct('Image', X));
    %
    % See also: OceanDL.Interpretability.OcclusionSensitivity

    methods
        function plot(obj, ax, data, options) %#ok<INUSL>
            if nargin < 4, options = struct(); end
            import OceanViz.Strategies.CNN.CNNVizUtils

            cla(ax, 'reset');
            heat = CNNVizUtils.normalize01(data.Heatmap);
            if isfield(options, 'Image') && ~isempty(options.Image)
                baseImg = options.Image;
                baseImg = reshape(baseImg, size(baseImg,1), size(baseImg,2), size(baseImg,3));
                if size(baseImg, 3) ~= 1 && size(baseImg, 3) ~= 3
                    baseImg = mean(baseImg, 3);
                end
                rgb = CNNVizUtils.blendOverlay(CNNVizUtils.normalize01(baseImg), heat, 0.55, hot(256));
            else
                rgb = CNNVizUtils.applyColormap(heat, hot(256));
            end
            image(ax, rgb);
            axis(ax, 'image', 'off');
            ax.Color = CNNVizUtils.COLOR_BG;
            title(ax, sprintf('Occlusion Sensitivity (base score = %.3f)', data.BaseScore), ...
                'Color', CNNVizUtils.COLOR_TEXT, 'FontSize', 10, 'FontWeight', 'bold');
        end
    end
end
