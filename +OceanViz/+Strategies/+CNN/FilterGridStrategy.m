classdef FilterGridStrategy < OceanViz.Strategies.BaseStrategy
    % FILTERGRIDSTRATEGY  Renders a Conv2D layer's filters as a tile grid.
    %   "What patterns have the filters learned?"
    %
    %   data = struct('Tiles', {WeightVisualizer.extractTiles(convLayer)})
    %   strat.plot(ax, data, struct());
    %
    % See also: OceanDL.Interpretability.WeightVisualizer

    methods
        function plot(obj, ax, data, options) %#ok<INUSL>
            if nargin < 4, options = struct(); end
            import OceanViz.Strategies.CNN.CNNVizUtils
            import OceanDL.Interpretability.WeightVisualizer

            cla(ax, 'reset');
            tiles = data.Tiles;
            imgs = cellfun(@WeightVisualizer.toDisplayImage, tiles, 'UniformOutput', false);
            labels = arrayfun(@(k) sprintf('f%d', k), 1:numel(imgs), 'UniformOutput', false);

            CNNVizUtils.drawTileGrid(ax, imgs, labels);
            ax.Color = CNNVizUtils.COLOR_BG;
            axis(ax, 'off');
            title(ax, sprintf('Conv Filters (%d)', numel(imgs)), ...
                'Color', CNNVizUtils.COLOR_TEXT, 'FontSize', 10, 'FontWeight', 'bold');
        end
    end
end
