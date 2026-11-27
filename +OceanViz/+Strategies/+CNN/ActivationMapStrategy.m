classdef ActivationMapStrategy < OceanViz.Strategies.BaseStrategy
    % ACTIVATIONMAPSTRATEGY  Single feature-map channel as a heatmap.
    %   "Where does a feature respond?"
    %
    %   data = ActivationMapExtractor.channelToImage(maps, channelIdx, sampleIdx);  % (H,W) in [0,1]
    %   strat.plot(ax, data, struct('Title', 'conv1 / channel 3'));
    %
    % See also: OceanDL.Interpretability.ActivationMapExtractor

    methods
        function plot(obj, ax, data, options) %#ok<INUSL>
            if nargin < 4, options = struct(); end
            import OceanViz.Strategies.CNN.CNNVizUtils

            cla(ax, 'reset');
            rgb = CNNVizUtils.applyColormap(CNNVizUtils.normalize01(data), parula(256));
            image(ax, rgb);
            axis(ax, 'image', 'off');
            ax.Color = CNNVizUtils.COLOR_BG;
            ttl = 'Activation Map';
            if isfield(options, 'Title'), ttl = options.Title; end
            title(ax, ttl, 'Color', CNNVizUtils.COLOR_TEXT, 'FontSize', 10, 'FontWeight', 'bold');
        end
    end
end
