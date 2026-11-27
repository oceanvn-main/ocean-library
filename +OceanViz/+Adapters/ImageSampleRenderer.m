classdef ImageSampleRenderer < OceanViz.Adapters.ISampleRenderer
    % IMAGESAMPLERENDERER  Renders a 2D pixel grid (grayscale or color) as
    %   a training sample panel inside the NNReplayEngine dashboard.
    %
    %   CONFIG fields (passed to setup):
    %     IMG_RES      - integer side-length (image is IMG_RES x IMG_RES)
    %     Colormap     - (optional) MATLAB colormap name, default 'gray'
    %     InputRange   - (optional) [lo hi] normalization range, default [-1 1]
    %
    % See also: OceanViz.Adapters.ISampleRenderer

    properties (Access = private)
        Config_  struct
        Colors_  struct
    end

    methods
        function setup(obj, ax, config, colors)
            assert(isfield(config, 'IMG_RES'), ...
                'OceanViz:ImageSampleRenderer:MissingConfig', ...
                'config.IMG_RES is required.');
            obj.Config_ = config;
            if ~isfield(obj.Config_, 'Colormap'),    obj.Config_.Colormap    = 'gray';    end
            if ~isfield(obj.Config_, 'InputRange'),  obj.Config_.InputRange  = [-1 1];    end
            obj.Colors_ = colors;
            set(ax, 'Color', colors.Background, 'XColor', 'none', 'YColor', 'none');
            title(ax, 'Sample', 'Color', colors.Text, 'FontSize', 9);
        end

        function render(obj, ax, frame, ep)
            r   = obj.Config_.InputRange;
            res = obj.Config_.IMG_RES;
            img = (reshape(frame.X, [res, res])' - r(1)) / (r(2) - r(1));
            img = max(0, min(1, img));
            imagesc(ax, img);
            colormap(ax, obj.Config_.Colormap);
            clim(ax, [0 1]);
            set(ax, 'XTick', [], 'YTick', []);
            title(ax, obj.getTitle(frame, ep), 'Color', obj.Colors_.Text, 'FontSize', 9);
        end

        function t = getTitle(~, frame, ep)
            t = sprintf('Input: %d  (epoch %d)', frame.Y, ep);
        end
    end
end
