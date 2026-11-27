classdef PatchGridStrategy < OceanViz.Strategies.BaseStrategy
    % PATCHGRIDSTRATEGY  Renders TopActivatingPatches.find()'s top-K crops
    %   as a labeled tile grid. "What kinds of inputs excite this feature?"
    %
    %   r = TopActivatingPatches.find(engine, images, 'conv2', 5);
    %   strat.plot(ax, r, struct());
    %
    % See also: OceanDL.Interpretability.TopActivatingPatches

    methods
        function plot(obj, ax, data, options) %#ok<INUSL>
            if nargin < 4, options = struct(); end
            import OceanViz.Strategies.CNN.CNNVizUtils

            cla(ax, 'reset');
            imgs = cellfun(@CNNVizUtils.normalize01, data.Patches, 'UniformOutput', false);
            labels = arrayfun(@(v) sprintf('%.2f', v), data.Values, 'UniformOutput', false);

            CNNVizUtils.drawTileGrid(ax, imgs, labels);
            ax.Color = CNNVizUtils.COLOR_BG;
            axis(ax, 'off');
            title(ax, sprintf('Top Activating Patches (%d)', numel(imgs)), ...
                'Color', CNNVizUtils.COLOR_TEXT, 'FontSize', 10, 'FontWeight', 'bold');
        end
    end
end
