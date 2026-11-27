classdef ImageView < handle
    % IMAGEVIEW  Fast image display on a uiaxes: the image object is created
    %   once and later frames only replace its pixel data (CData) -- sliders
    %   and animations stay smooth. Values are shown in [0, 1] (grey or RGB);
    %   'Mode' "noise" maps 0 to mid-grey and +-2 to white / black (for eps).

    properties (SetAccess = private)
        Axes
        Image = []
    end

    methods
        function obj = ImageView(ax, titleText, ctx)
            obj.Axes = ax;
            ax.Color = ctx.Theme.Background;
            ax.XColor = 'none'; ax.YColor = 'none';
            ax.Title.Color = ctx.Theme.Text;
            ax.Title.FontWeight = 'normal';
            title(ax, titleText);
            colormap(ax, gray(256));
            ax.CLim = [0 1];
            axis(ax, 'image'); ax.YDir = 'reverse';
            ax.Toolbar.Visible = 'off';
            disableDefaultInteractivity(ax);
        end

        function set(obj, img, mode)
            if nargin < 3, mode = "image"; end
            img = double(gather(img));
            if mode == "noise", img = img / 4 + 0.5; end
            img = min(1, max(0, img));
            if size(img, 3) == 1, img = img(:, :, 1); end
            if isempty(obj.Image) || ~isvalid(obj.Image) || ~isequal(size(obj.Image.CData), size(img))
                delete(obj.Image);
                obj.Image = image(obj.Axes, img, 'CDataMapping', 'scaled');
                axis(obj.Axes, 'image');
                obj.Axes.XLim = [0.5 size(img, 2) + 0.5]; obj.Axes.YLim = [0.5 size(img, 1) + 0.5];
            else
                obj.Image.CData = img;
            end
        end

        function setTitle(obj, text)
            title(obj.Axes, text);
        end
    end

    methods (Static)
        function G = tile(imgs, cols)
            % Grid of equally sized images (H x W x C x N, [0, 1]) with 1-px separators.
            [H, W, C, N] = size(imgs, 1:4);
            rows = ceil(N / cols);
            G = 0.2 * ones(rows * (H + 1) + 1, cols * (W + 1) + 1, C);
            for k = 1:N
                r = floor((k - 1) / cols); c = mod(k - 1, cols);
                G(r * (H + 1) + 1 + (1:H), c * (W + 1) + 1 + (1:W), :) = min(1, max(0, imgs(:, :, :, k)));
            end
        end
    end
end
