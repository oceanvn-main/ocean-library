classdef GradCAMOverlayStrategy < OceanViz.Strategies.BaseStrategy
    % GRADCAMOVERLAYSTRATEGY  Grad-CAM heatmap over the original image,
    %   upsampled from the (coarser) conv layer's spatial resolution.
    %   "Which high-level spatial features support this class?"
    %
    %   r = GradCAM.compute(engine, X, targetClass, 'conv2', 'OutputLayer', 'dense1');
    %   strat.plot(ax, r, struct('Image', X));
    %
    % See also: OceanDL.Interpretability.GradCAM

    methods
        function plot(obj, ax, data, options) %#ok<INUSL>
            if nargin < 4, options = struct(); end
            import OceanViz.Strategies.CNN.CNNVizUtils

            cla(ax, 'reset');
            cam = data.CAMNormalized(:, :, 1);

            if isfield(options, 'Image') && ~isempty(options.Image)
                baseImg = options.Image;
                baseImg = reshape(baseImg, size(baseImg,1), size(baseImg,2), size(baseImg,3));
                if size(baseImg, 3) ~= 1 && size(baseImg, 3) ~= 3
                    baseImg = mean(baseImg, 3);
                end
                baseImg = CNNVizUtils.normalize01(baseImg);
                camUp = CNNVizUtils.resizeNearest(cam, size(baseImg, 1), size(baseImg, 2));
                rgb = CNNVizUtils.blendOverlay(baseImg, camUp, 0.5, jet(256));
            else
                rgb = CNNVizUtils.applyColormap(cam, jet(256));
            end
            image(ax, rgb);
            axis(ax, 'image', 'off');
            ax.Color = CNNVizUtils.COLOR_BG;
            title(ax, 'Grad-CAM', 'Color', CNNVizUtils.COLOR_TEXT, 'FontSize', 10, 'FontWeight', 'bold');
        end
    end
end
