classdef SaliencyOverlayStrategy < OceanViz.Strategies.BaseStrategy
    % SALIENCYOVERLAYSTRATEGY  Saliency heatmap over the original image.
    %   "Which pixels locally affect the class score most?"
    %
    %   r = SaliencyMap.compute(engine, X, targetClass, 'OutputLayer', 'dense1');
    %   strat.plot(ax, r, struct('Image', X));
    %
    % See also: OceanDL.Interpretability.SaliencyMap

    methods
        function plot(obj, ax, data, options) %#ok<INUSL>
            if nargin < 4, options = struct(); end
            import OceanViz.Strategies.CNN.CNNVizUtils

            cla(ax, 'reset');
            sal = data.Saliency(:, :, 1);
            heat = CNNVizUtils.normalize01(sal);

            if isfield(options, 'Image') && ~isempty(options.Image)
                baseImg = options.Image;
                baseImg = reshape(baseImg, size(baseImg,1), size(baseImg,2), size(baseImg,3));
                if size(baseImg, 3) ~= 1 && size(baseImg, 3) ~= 3
                    baseImg = mean(baseImg, 3);
                end
                rgb = CNNVizUtils.blendOverlay(CNNVizUtils.normalize01(baseImg), heat, 0.6, hot(256));
            else
                rgb = CNNVizUtils.applyColormap(heat, hot(256));
            end
            image(ax, rgb);
            axis(ax, 'image', 'off');
            ax.Color = CNNVizUtils.COLOR_BG;
            title(ax, sprintf('Saliency (score = %.3f)', data.Score(1)), ...
                'Color', CNNVizUtils.COLOR_TEXT, 'FontSize', 10, 'FontWeight', 'bold');
        end
    end
end
