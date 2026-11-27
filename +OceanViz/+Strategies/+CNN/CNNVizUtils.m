classdef CNNVizUtils
    % CNNVIZUTILS  Shared rendering helpers for the +OceanViz/+Strategies/+CNN
    %   family (one Strategy per interpretability method -- see
    %   OceanDL.Interpretability for the math they render).
    %   Kept dependency-free (no Image Processing Toolbox) so every
    %   strategy stays usable wherever the rest of this library is.

    properties (Constant)
        COLOR_BG   = [0.08 0.08 0.10]
        COLOR_TEXT = [0.85 0.85 0.90]
        COLOR_DIM  = [0.50 0.53 0.58]
    end

    methods (Static)
        function rgb = toRGB(img)
            % TORGB  (H x W) or (H x W x 3) -> (H x W x 3), replicating a
            %   single channel across R/G/B.
            if size(img, 3) == 3
                rgb = img;
            elseif size(img, 3) == 1
                rgb = repmat(img, 1, 1, 3);
            else
                error('OceanViz:Strategies:CNN:CNNVizUtils:BadChannelCount', ...
                    'Expected 1 or 3 channels, got %d.', size(img, 3));
            end
        end

        function img = normalize01(img)
            lo = min(img(:)); hi = max(img(:));
            if hi > lo
                img = (img - lo) / (hi - lo);
            else
                img = zeros(size(img));
            end
        end

        function out = resizeNearest(img, targetH, targetW)
            % RESIZENEAREST  Nearest-neighbor resize, no toolbox dependency.
            [H, W, C] = size(img);
            rIdx = min(H, max(1, round(((1:targetH) - 0.5) * H / targetH + 0.5)));
            cIdx = min(W, max(1, round(((1:targetW) - 0.5) * W / targetW + 0.5)));
            out = img(rIdx, cIdx, 1:C);
        end

        function rgb = applyColormap(heatmap01, cmap)
            % APPLYCOLORMAP  (H x W) in [0,1] -> (H x W x 3) via a Nx3 colormap.
            arguments
                heatmap01
                cmap (:,3) double = hot(256)
            end
            n = size(cmap, 1);
            idx = min(n, max(1, round(heatmap01 * (n - 1)) + 1));
            rgb = reshape(cmap(idx(:), :), size(heatmap01, 1), size(heatmap01, 2), 3);
        end

        function rgb = blendOverlay(baseRGB, heatmap01, alpha, cmap)
            arguments
                baseRGB
                heatmap01
                alpha (1,1) double = 0.45
                cmap (:,3) double = hot(256)
            end
            import OceanViz.Strategies.CNN.CNNVizUtils
            baseRGB = CNNVizUtils.toRGB(baseRGB);
            overlayRGB = CNNVizUtils.applyColormap(heatmap01, cmap);
            if ~isequal(size(baseRGB, [1 2]), size(overlayRGB, [1 2]))
                overlayRGB = CNNVizUtils.resizeNearest(overlayRGB, size(baseRGB, 1), size(baseRGB, 2));
                heatmap01 = CNNVizUtils.resizeNearest(heatmap01, size(baseRGB, 1), size(baseRGB, 2));
            end
            w = alpha * heatmap01; % stronger response -> more overlay, weak response -> mostly base image
            rgb = baseRGB .* (1 - w) + overlayRGB .* w;
        end

        function drawTileGrid(ax, images, labels, thm)
            % DRAWTILEGRID  Lay `images` (cell of (h x w) or (h x w x 3),
            %   already normalized to [0,1]) out on a single axes as an
            %   auto-sized grid, one image() call per tile.
            arguments
                ax
                images cell
                labels cell = {}
                thm = struct('Bg', OceanViz.Strategies.CNN.CNNVizUtils.COLOR_BG, ...
                              'Text', OceanViz.Strategies.CNN.CNNVizUtils.COLOR_TEXT, ...
                              'Dim', OceanViz.Strategies.CNN.CNNVizUtils.COLOR_DIM)
            end
            import OceanViz.Strategies.CNN.CNNVizUtils
            n = numel(images);
            if n == 0
                text(ax, 0.5, 0.5, 'Nothing to display.', 'Color', thm.Dim, ...
                    'HorizontalAlignment', 'center', 'Units', 'normalized');
                return
            end
            cols = ceil(sqrt(n));
            rows = ceil(n / cols);
            pad = 0.08;
            hold(ax, 'on');
            for k = 1:n
                r = floor((k - 1) / cols);
                c = mod(k - 1, cols);
                x0 = c + pad; x1 = c + 1 - pad;
                y0 = (rows - 1 - r) + pad; y1 = (rows - 1 - r) + 1 - pad;
                img = CNNVizUtils.toRGB(images{k});
                image(ax, 'CData', img, 'XData', [x0 x1], 'YData', [y0 y1]);
                if numel(labels) >= k && ~isempty(labels{k})
                    text(ax, (x0+x1)/2, y0 - 0.06, labels{k}, 'Color', thm.Text, ...
                        'FontSize', 7.5, 'HorizontalAlignment', 'center');
                end
            end
            ax.XLim = [0 cols];
            ax.YLim = [-0.15 rows];
            ax.YDir = 'normal';
            hold(ax, 'off');
        end
    end
end
