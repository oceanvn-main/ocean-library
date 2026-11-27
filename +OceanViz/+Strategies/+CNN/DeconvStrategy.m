classdef DeconvStrategy < OceanViz.Strategies.BaseStrategy
    % DECONVSTRATEGY  Renders a DeconvVisualizer.reconstruct() result as a
    %   normalized input-space image. "What input-space structure
    %   corresponds to this activation?"
    %
    %   r = DeconvVisualizer.reconstruct(engine, X, 'conv2', 5);
    %   strat.plot(ax, r, struct());
    %
    % See also: OceanDL.Interpretability.DeconvVisualizer

    methods
        function plot(obj, ax, data, options) %#ok<INUSL>
            if nargin < 4, options = struct(); end
            import OceanViz.Strategies.CNN.CNNVizUtils

            cla(ax, 'reset');
            recon = data.Reconstruction;               % (H x W x Cin x 1)
            recon = reshape(recon, size(recon,1), size(recon,2), size(recon,3));
            Cin = size(recon, 3);
            if Cin == 1 || Cin == 3
                img = recon;
            else
                img = mean(recon, 3);
            end
            img = CNNVizUtils.normalize01(img);
            image(ax, CNNVizUtils.toRGB(img));
            axis(ax, 'image', 'off');
            ax.Color = CNNVizUtils.COLOR_BG;
            title(ax, sprintf('Deconv Reconstruction (channel %d @ [%d,%d])', ...
                data.SeedChannel, data.SeedRow, data.SeedCol), ...
                'Color', CNNVizUtils.COLOR_TEXT, 'FontSize', 10, 'FontWeight', 'bold');
        end
    end
end
