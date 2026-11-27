classdef InteractiveImageCanvas < handle
    % INTERACTIVEIMAGECANVAS  Interactive scribble brush tool for image colorization.
    %
    %   Part of Szeliski §4.2.4 "Interactive Colorization".
    %   Allows users to load a grayscale image, paint color scribbles with the mouse,
    %   choose palette colors and brush radii, run Levin optimization directly inside
    %   the canvas GUI, and toggle between Scribble view and Colorized output view.
    %
    %   Usage:
    %       canvas = OceanViz.App.InteractiveImageCanvas();
    %       canvas = OceanViz.App.InteractiveImageCanvas(I_gray);
    %
    %   Author: Ocean Library  |  2026-08-05

    properties
        UIFigure          matlab.ui.Figure
        MainLayout        matlab.ui.container.GridLayout
        HeaderPanel       matlab.ui.container.Panel
        SidebarPanel      matlab.ui.container.Panel
        CanvasPanel       matlab.ui.container.Panel
        
        CanvasAxes        matlab.ui.control.UIAxes
        ImageHandle
        
        BrushRadiusSlider matlab.ui.control.Slider
        LambdaSlider      matlab.ui.control.Slider
        SigmaSlider       matlab.ui.control.Slider
        ViewModeDropDown  matlab.ui.control.DropDown
        
        StatusLabel       matlab.ui.control.Label
        
        CurrentGrayImage  double
        ScribbleImageRGB  double
        ColorizedImageRGB double = []
        ScribbleMask      logical
        
        ActiveColor       (1,3) double = [1.0, 0.0, 0.0]  % Default Red
        BrushRadius       (1,1) double = 6.0
        IsPainting        (1,1) logical = false
        
        VizEngine         OceanViz.Core
        LastResult        OceanData.ExperimentResult
    end

    methods (Static)
        function canvas = launch(I_gray)
            % LAUNCH Static helper to launch interactive canvas on a grayscale image.
            arguments
                I_gray double = []
            end
            canvas = OceanViz.App.InteractiveImageCanvas(I_gray);
        end
    end

    methods
        function obj = InteractiveImageCanvas(I_gray)
            % INTERACTIVEIMAGECANVAS Constructor.
            arguments
                I_gray double = []
            end

            obj.VizEngine = OceanViz.Core('Theme', 'Dark');
            
            % Load default real image if not provided
            if isempty(I_gray)
                defaultPath = fullfile('Data', 'images', 'corgi.jpg');
                if ~isfile(defaultPath)
                    defaultPath = fullfile('Data', 'images', 'apple.jpg');
                end
                if isfile(defaultPath)
                    img_raw = imread(defaultPath);
                    if ndims(img_raw) == 3
                        img_raw = rgb2gray(img_raw);
                    end
                    I_gray = double(img_raw) / 255.0;
                else
                    [X, Y] = meshgrid(linspace(-2,2,128), linspace(-2,2,128));
                    I_gray = 0.5 + 0.3 * sin(X.^2 + Y.^2);
                end
            end
            
            % Downsample high-res images so solver runs in ~0.2s and brush strokes are clear
            I_gray = double(I_gray);
            if max(I_gray(:)) > 1.0
                I_gray = I_gray / 255.0;
            end
            
            maxDim = 256;
            [H_raw, W_raw] = size(I_gray);
            if max(H_raw, W_raw) > maxDim
                scale = maxDim / max(H_raw, W_raw);
                I_gray = imresize(I_gray, scale);
            end
            
            obj.CurrentGrayImage = I_gray;
            [H, W] = size(obj.CurrentGrayImage);
            obj.ScribbleImageRGB = repmat(obj.CurrentGrayImage, [1, 1, 3]);
            obj.ScribbleMask     = false(H, W);
            
            obj.buildUI();
            obj.redrawCanvas();
        end

        function addScribbleDot(obj, row, col)
            % ADDSCRIBBLEDOT Draw a filled circular scribble at (row, col).
            % Automatically switches view back to Scribbles mode when painting
            if strcmp(obj.ViewModeDropDown.Value, 'Colorized Output')
                obj.ViewModeDropDown.Value = 'Scribbles';
            end

            [H, W] = size(obj.CurrentGrayImage);
            r = ceil(obj.BrushRadius);
            
            r_min = max(1, row - r);  r_max = min(H, row + r);
            c_min = max(1, col - r);  c_max = min(W, col + r);
            
            [r_grid, c_grid] = ndgrid(r_min:r_max, c_min:c_max);
            dist_sq = (r_grid - row).^2 + (c_grid - col).^2;
            mask_circ = (dist_sq <= obj.BrushRadius^2);
            
            flat_indices = r_grid(mask_circ) + (c_grid(mask_circ) - 1) * H;
            
            if isempty(flat_indices)
                return;
            end
            
            obj.ScribbleMask(flat_indices) = true;
            
            for ch = 1:3
                ch_mat = obj.ScribbleImageRGB(:, :, ch);
                ch_mat(flat_indices) = obj.ActiveColor(ch);
                obj.ScribbleImageRGB(:, :, ch) = ch_mat;
            end
            
            obj.redrawCanvas();
        end

        function clearScribbles(obj)
            % CLEARSCRIBBLES Reset scribble layer.
            [H, W] = size(obj.CurrentGrayImage);
            obj.ScribbleImageRGB  = repmat(obj.CurrentGrayImage, [1, 1, 3]);
            obj.ColorizedImageRGB = [];
            obj.ScribbleMask      = false(H, W);
            obj.ViewModeDropDown.Value = 'Scribbles';
            obj.redrawCanvas();
        end

        function runColorization(obj)
            % RUNCOLORIZATION Execute Levin optimization directly inside this canvas GUI.
            if ~any(obj.ScribbleMask(:))
                uialert(obj.UIFigure, 'Please draw at least one color scribble on the canvas first!', 'No Scribbles');
                return;
            end
            
            obj.StatusLabel.Text = 'Status: Solving Levin colorization linear system...';
            drawnow;
            
            sigma_val  = obj.SigmaSlider.Value;
            lambda_val = obj.LambdaSlider.Value;
            
            colorizer = OceanMath.Image.Variational.LevinColorizer( ...
                'Sigma', sigma_val, ...
                'Lambda', lambda_val);
            
            tic;
            obj.ColorizedImageRGB = colorizer.colorize(obj.CurrentGrayImage, obj.ScribbleImageRGB, obj.ScribbleMask);
            time_sec = toc;
            
            obj.StatusLabel.Text = sprintf('Status: Done (%.2f sec). Displaying Colorized Output directly.', time_sec);
            
            % Save LastResult container
            res = OceanData.ExperimentResult('Interactive Colorization Result', ...
                'Description', 'Levin, Lischinski & Weiss (2004) Szeliski §4.2.4');
            res.addParam('Sigma', sigma_val);
            res.addParam('Lambda', lambda_val);
            res.addMetric('SolveTime_sec', time_sec);
            res.addMetric('ScribblePixels', sum(obj.ScribbleMask(:)));
            res.add('GrayscaleInput',  obj.CurrentGrayImage,  'Description', 'Input grayscale image');
            res.add('UserScribbles',   obj.ScribbleImageRGB,  'Description', 'Grayscale image with user color scribbles');
            res.add('ScribbleMask',    double(obj.ScribbleMask), 'Description', 'Binary indicator of scribble locations');
            res.add('ColorizedOutput', obj.ColorizedImageRGB, 'Description', 'Reconstructed RGB colorized image');
            obj.LastResult = res;

            % Automatically switch view mode to display Colorized Output directly on canvas!
            obj.ViewModeDropDown.Value = 'Colorized Output';
            obj.redrawCanvas();
        end

        function openInStudio(obj)
            % OPENINSTUDIO Explicitly open current result in ExperimentStudio.
            if isempty(obj.LastResult)
                if any(obj.ScribbleMask(:))
                    obj.runColorization();
                else
                    uialert(obj.UIFigure, 'Draw scribbles and run colorization first.', 'No Result');
                    return;
                end
            end
            OceanViz.App.ExperimentStudio.inspect(obj.LastResult);
        end
    end

    methods (Access = private)
        function buildUI(obj)
            colors = obj.VizEngine.Colors;
            
            obj.UIFigure = figure('Name', 'Szeliski §4.2.4 — Interactive Colorization Canvas', ...
                'NumberTitle', 'off', ...
                'Color', colors.Background, ...
                'Position', [120 100 1150 720], ...
                'WindowButtonDownFcn', @(~,~) obj.onMouseDown(), ...
                'WindowButtonMotionFcn', @(~,~) obj.onMouseMove(), ...
                'WindowButtonUpFcn', @(~,~) obj.onMouseUp());
            
            obj.MainLayout = uigridlayout(obj.UIFigure, [2, 2]);
            obj.MainLayout.RowHeight = {50, '1x'};
            obj.MainLayout.ColumnWidth = {280, '1x'};
            obj.MainLayout.BackgroundColor = colors.Background;
            
            % ── Header Panel ───────────────────────────────────────────────
            obj.HeaderPanel = uipanel(obj.MainLayout, 'BackgroundColor', colors.Background, 'BorderType', 'none');
            obj.HeaderPanel.Layout.Row = 1;
            obj.HeaderPanel.Layout.Column = [1, 2];
            
            hGrid = uigridlayout(obj.HeaderPanel, [1, 5]);
            hGrid.ColumnWidth = {'1x', 110, 110, 130, 130};
            hGrid.Padding = [10 5 10 5];
            
            uilabel(hGrid, 'Text', 'Szeliski §4.2.4 — Interactive Colorization Canvas', ...
                'FontWeight', 'bold', 'FontSize', 15, 'FontColor', colors.Text);
            
            uibutton(hGrid, 'push', 'Text', 'Load Image...', ...
                'BackgroundColor', colors.Background * 1.5, 'FontColor', colors.Text, ...
                'ButtonPushedFcn', @(~,~) obj.onLoadImage());
            
            uibutton(hGrid, 'push', 'Text', 'Clear Scribbles', ...
                'BackgroundColor', colors.Secondary * 0.8, 'FontColor', [1 1 1], ...
                'ButtonPushedFcn', @(~,~) obj.clearScribbles());
            
            uibutton(hGrid, 'push', 'Text', 'Run Colorize', ...
                'BackgroundColor', colors.Primary * 0.9, 'FontColor', [1 1 1], 'FontWeight', 'bold', ...
                'ButtonPushedFcn', @(~,~) obj.runColorization());
            
            uibutton(hGrid, 'push', 'Text', 'Open Studio...', ...
                'BackgroundColor', colors.Background * 1.8, 'FontColor', colors.Text, ...
                'ButtonPushedFcn', @(~,~) obj.openInStudio());
            
            % ── Sidebar Panel ──────────────────────────────────────────────
            obj.SidebarPanel = uipanel(obj.MainLayout, 'Title', ' Brush & Solver Options', ...
                'BackgroundColor', colors.Background, 'ForegroundColor', colors.Text, 'FontWeight', 'bold');
            obj.SidebarPanel.Layout.Row = 2;
            obj.SidebarPanel.Layout.Column = 1;
            
            sideGrid = uigridlayout(obj.SidebarPanel, [11, 1]);
            sideGrid.RowHeight = {22, 30, 22, 100, 22, 35, 22, 35, 22, 35, 25};
            sideGrid.Padding = [10 10 10 10];
            
            uilabel(sideGrid, 'Text', 'Canvas View Mode:', 'FontColor', colors.Text, 'FontWeight', 'bold');
            obj.ViewModeDropDown = uidropdown(sideGrid, ...
                'Items', {'Scribbles', 'Colorized Output'}, ...
                'Value', 'Scribbles', ...
                'BackgroundColor', colors.Background * 1.3, 'FontColor', colors.Text, ...
                'ValueChangedFcn', @(~,~) obj.redrawCanvas());
            
            uilabel(sideGrid, 'Text', 'Palette Preset:', 'FontColor', colors.Text, 'FontWeight', 'bold');
            
            % Color palette swatches
            paletteGrid = uigridlayout(sideGrid, [2, 4]);
            paletteGrid.Padding = [0 0 0 0];
            paletteColors = [ ...
                1.0 0.0 0.0; ...  % Red
                0.0 0.8 0.0; ...  % Green
                0.0 0.4 1.0; ...  % Blue
                1.0 0.8 0.0; ...  % Yellow
                0.0 0.9 0.9; ...  % Cyan
                0.9 0.0 0.9; ...  % Magenta
                1.0 0.5 0.0; ...  % Orange
                0.6 0.2 0.8       % Purple
            ];
            
            for p = 1:8
                c = paletteColors(p, :);
                uibutton(paletteGrid, 'push', 'Text', '', ...
                    'BackgroundColor', c, ...
                    'ButtonPushedFcn', @(~,~) obj.setActiveColor(c));
            end
            
            uilabel(sideGrid, 'Text', 'Brush Radius (px):', 'FontColor', colors.Text, 'FontWeight', 'bold');
            obj.BrushRadiusSlider = uislider(sideGrid, 'Limits', [1, 30], 'Value', 6, ...
                'FontColor', colors.Text, 'ValueChangedFcn', @(src,~) obj.setBrushRadius(src.Value));
            
            uilabel(sideGrid, 'Text', 'Luminance Affinity (\sigma):', 'FontColor', colors.Text, 'FontWeight', 'bold');
            obj.SigmaSlider = uislider(sideGrid, 'Limits', [0.01, 0.2], 'Value', 0.05, ...
                'FontColor', colors.Text);
            
            uilabel(sideGrid, 'Text', 'Smoothness (\lambda):', 'FontColor', colors.Text, 'FontWeight', 'bold');
            obj.LambdaSlider = uislider(sideGrid, 'Limits', [0.1, 10.0], 'Value', 1.0, ...
                'FontColor', colors.Text);
            
            obj.StatusLabel = uilabel(sideGrid, 'Text', 'Status: Ready. Draw scribbles on canvas.', ...
                'FontColor', colors.Text * 0.8, 'FontSize', 10);
            
            % ── Canvas Panel ───────────────────────────────────────────────
            obj.CanvasPanel = uipanel(obj.MainLayout, 'Title', ' Interactive Painting Canvas', ...
                'BackgroundColor', colors.Background, 'ForegroundColor', colors.Text, 'FontWeight', 'bold');
            obj.CanvasPanel.Layout.Row = 2;
            obj.CanvasPanel.Layout.Column = 2;
            
            cGrid = uigridlayout(obj.CanvasPanel, [1, 1]);
            cGrid.Padding = [10 10 10 10];
            
            obj.CanvasAxes = uiaxes(cGrid);
            set(obj.CanvasAxes, 'Color', colors.Background, 'XColor', colors.Text, 'YColor', colors.Text);
            axis(obj.CanvasAxes, 'image');
        end

        function redrawCanvas(obj)
            mode = obj.ViewModeDropDown.Value;
            
            if strcmp(mode, 'Colorized Output') && ~isempty(obj.ColorizedImageRGB)
                displayImg = obj.ColorizedImageRGB;
                titleText = 'Colorized Output Result';
            else
                displayImg = obj.ScribbleImageRGB;
                titleText = 'Click and drag mouse to paint scribbles';
            end
            
            if isempty(obj.ImageHandle) || ~isvalid(obj.ImageHandle)
                obj.ImageHandle = imshow(displayImg, 'Parent', obj.CanvasAxes);
                axis(obj.CanvasAxes, 'image');
                title(obj.CanvasAxes, titleText, 'Color', obj.VizEngine.Colors.Text);
            else
                obj.ImageHandle.CData = displayImg;
                title(obj.CanvasAxes, titleText, 'Color', obj.VizEngine.Colors.Text);
            end
            drawnow limitrate;
        end

        function setActiveColor(obj, c)
            obj.ActiveColor = c;
            obj.StatusLabel.Text = sprintf('Color selected: RGB [%.1f, %.1f, %.1f]', c(1), c(2), c(3));
        end

        function setBrushRadius(obj, r)
            obj.BrushRadius = r;
        end

        function onLoadImage(obj)
            [filename, path] = uigetfile({'*.png;*.jpg;*.jpeg;*.bmp;*.tif', 'Image Files'}, 'Select Image');
            if isequal(filename, 0)
                return;
            end
            img_raw = imread(fullfile(path, filename));
            if ndims(img_raw) == 3
                img_raw = rgb2gray(img_raw);
            end
            img = double(img_raw);
            if max(img(:)) > 1.0
                img = img / 255.0;
            end
            
            % Downsample so max dimension <= 256 for fast WLS solving & visible brush strokes
            maxDim = 256;
            [H_raw, W_raw] = size(img);
            if max(H_raw, W_raw) > maxDim
                scale = maxDim / max(H_raw, W_raw);
                img = imresize(img, scale);
            end
            
            obj.CurrentGrayImage  = img;
            obj.ColorizedImageRGB = [];
            obj.ImageHandle = [];  % Reset handle for new axes dimensions
            obj.clearScribbles();
        end

        function onMouseDown(obj)
            cp = obj.CanvasAxes.CurrentPoint;
            col = round(cp(1,1));
            row = round(cp(1,2));
            [H, W] = size(obj.CurrentGrayImage);
            if col >= 1 && col <= W && row >= 1 && row <= H
                obj.IsPainting = true;
                obj.addScribbleDot(row, col);
            end
        end

        function onMouseMove(obj)
            if ~obj.IsPainting
                return;
            end
            cp = obj.CanvasAxes.CurrentPoint;
            col = round(cp(1,1));
            row = round(cp(1,2));
            [H, W] = size(obj.CurrentGrayImage);
            if col >= 1 && col <= W && row >= 1 && row <= H
                obj.addScribbleDot(row, col);
            end
        end

        function onMouseUp(obj)
            obj.IsPainting = false;
        end
    end
end
