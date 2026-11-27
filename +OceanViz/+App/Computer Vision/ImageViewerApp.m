classdef ImageViewerApp < handle
    % IMAGEVIEWERAPP  Interactive multi-type image viewer with annotation tools.
    %
    % Explorer-archetype UIFigure app using viewer2d + imageshow.
    %
    % Supports image types:
    %   uint8 (RGB, grayscale), uint16, double/single (HDR/float),
    %   logical (binary mask), int8/int16/int32, multi-channel (>3).
    %
    % Display Modes:
    %   Normal       - Shows image as-is
    %   Grayscale    - Converts to luminance grayscale
    %   False Color  - Applies a colormap to single-channel images
    %   Channel Split - 3 linked viewer2d panes (R | G | B)
    %
    % Annotations: Rectangle, Circle, Line (via uidraw) + Clear All
    % Export:      Save current view to PNG / JPG / TIFF via uiputfile
    %
    % Usage:
    %   addpath('c:\Ocean Library\Matlab code practice\+OceanViz\+App\Computer Vision')
    %   app = ImageViewerApp();
    %
    % Author: Ocean Library
    % Date:   2026-09-16

    % ======================================================================
    % Properties
    % ======================================================================
    properties (Access = private)
        % Figure & root layout
        UIFigure_           matlab.ui.Figure
        RootGrid_           matlab.ui.container.GridLayout

        % Sidebar
        SidePanel_          matlab.ui.container.Panel
        SideGrid_           matlab.ui.container.GridLayout

        % Sidebar controls
        DropdownPreset_     matlab.ui.control.DropDown
        BtnLoad_            matlab.ui.control.Button
        LblFilename_        matlab.ui.control.Label
        LblDimensions_      matlab.ui.control.Label
        LblClass_           matlab.ui.control.Label
        LblChannels_        matlab.ui.control.Label
        LblIntensity_       matlab.ui.control.Label
        LblCursor_          matlab.ui.control.Label
        DropdownMode_       matlab.ui.control.DropDown
        DropdownColormap_   matlab.ui.control.DropDown
        BtnAnnotRect_       matlab.ui.control.Button
        BtnAnnotCircle_     matlab.ui.control.Button
        BtnAnnotLine_       matlab.ui.control.Button
        BtnClearAnnot_      matlab.ui.control.Button
        BtnExport_          matlab.ui.control.Button

        % Viewer area
        ViewerPanel_        matlab.ui.container.Panel

        % Main viewer (Normal / Grayscale / False Color modes)
        Viewer_
        Image_

        % Channel Split viewers (linked)
        ViewerR_
        ViewerG_
        ViewerB_
        ImageR_
        ImageG_
        ImageB_

        % State
        AnnotationHandles_  = {}
        CurrentImage_
        CurrentImagePath_   = ''
        CurrentMode_        = 'Normal'
        SplitModeActive_    = false
    end

    properties (Access = private, Constant)
        FIG_WIDTH    = 1180
        FIG_HEIGHT   = 740
        SIDEBAR_W    = 260

        PRESET_IMAGES = {'ocean_img.jpg', 'ocean_img_gray.jpg', 'elon_musk.jpg'}
        DISPLAY_MODES = {'Normal', 'Grayscale', 'False Color', 'Channel Split'}
        COLORMAPS     = {'parula', 'turbo', 'hot', 'cool', 'jet', 'hsv', 'bone', 'copper', 'gray'}
    end

    % ======================================================================
    % Public Methods
    % ======================================================================
    methods (Access = public)

        function app = ImageViewerApp()
            % IMAGEVIEWERAPP  Build and open the viewer window.
            app.buildUI_();
            app.loadPreset_(app.PRESET_IMAGES{1});
        end

        function show(app)
            % SHOW  Bring the window to the foreground.
            if ~isempty(app.UIFigure_) && isvalid(app.UIFigure_)
                figure(app.UIFigure_);
            end
        end

        function delete(app)
            % DELETE  Close the window and release resources.
            if ~isempty(app.UIFigure_) && isvalid(app.UIFigure_)
                delete(app.UIFigure_);
            end
        end
    end

    % ======================================================================
    % Private — UI Construction
    % ======================================================================
    methods (Access = private)

        function buildUI_(app)
            scr = get(0, 'ScreenSize');
            x0  = max(1, (scr(3) - app.FIG_WIDTH)  / 2);
            y0  = max(1, (scr(4) - app.FIG_HEIGHT) / 2);

            app.UIFigure_ = uifigure( ...
                'Name',            'Image Viewer — Ocean Library', ...
                'Position',        [x0, y0, app.FIG_WIDTH, app.FIG_HEIGHT], ...
                'CloseRequestFcn', @(~,~) app.onClose_());

            % Root: [sidebar | viewer] — Explorer archetype skeleton
            app.RootGrid_ = uigridlayout(app.UIFigure_, [1, 2]);
            app.RootGrid_.ColumnWidth   = {app.SIDEBAR_W, '1x'};
            app.RootGrid_.RowHeight     = {'1x'};
            app.RootGrid_.Padding       = [0, 0, 0, 0];
            app.RootGrid_.ColumnSpacing = 0;

            app.buildSidebar_();
            app.buildViewerPanel_();
        end

        % ------------------------------------------------------------------
        function buildSidebar_(app)
            % Sidebar panel in column 1
            app.SidePanel_ = uipanel(app.RootGrid_, 'Title', '');
            app.SidePanel_.Layout.Row    = 1;
            app.SidePanel_.Layout.Column = 1;

            % Inner grid — 'fit' rows pack tightly; last row '1x' is filler
            % 27 content rows + 1 filler row
            nRows = 28;
            rowH  = repmat({'fit'}, 1, nRows);
            rowH{nRows} = '1x'; % filler at bottom

            app.SideGrid_ = uigridlayout(app.SidePanel_, [nRows, 1]);
            app.SideGrid_.RowHeight    = rowH;
            app.SideGrid_.ColumnWidth  = {'1x'};
            app.SideGrid_.Padding      = [12, 12, 12, 12];
            app.SideGrid_.RowSpacing   = 5;

            r = 0;

            % ---- IMAGE SOURCE ------------------------------------------
            r = r+1; app.sectionLabel_(r, 'IMAGE SOURCE');
            r = r+1;
            app.DropdownPreset_ = uidropdown(app.SideGrid_, ...
                'Items', app.PRESET_IMAGES, ...
                'Value', app.PRESET_IMAGES{1}, ...
                'Tooltip', 'Select a preset image', ...
                'ValueChangedFcn', @(~,~) app.onPresetChanged_());
            app.DropdownPreset_.Layout.Row    = r;
            app.DropdownPreset_.Layout.Column = 1;

            r = r+1;
            app.BtnLoad_ = uibutton(app.SideGrid_, ...
                'Text', 'Load Image File...', ...
                'ButtonPushedFcn', @(~,~) app.onLoadCustom_());
            app.BtnLoad_.Layout.Row    = r;
            app.BtnLoad_.Layout.Column = 1;

            % ---- IMAGE INFO --------------------------------------------
            r = r+1; app.sectionLabel_(r, 'IMAGE INFO');
            r = r+1; app.LblFilename_   = app.infoLabel_(r, 'File: —');
            r = r+1; app.LblDimensions_ = app.infoLabel_(r, 'Size: —');
            r = r+1; app.LblClass_      = app.infoLabel_(r, 'Class: —');
            r = r+1; app.LblChannels_   = app.infoLabel_(r, 'Channels: —');
            r = r+1; app.LblIntensity_  = app.infoLabel_(r, 'Range: —');

            % ---- DISPLAY MODE -----------------------------------------
            r = r+1; app.sectionLabel_(r, 'DISPLAY MODE');
            r = r+1;
            app.DropdownMode_ = uidropdown(app.SideGrid_, ...
                'Items', app.DISPLAY_MODES, ...
                'Value', 'Normal', ...
                'Tooltip', 'Select display rendering mode', ...
                'ValueChangedFcn', @(~,~) app.onModeChanged_());
            app.DropdownMode_.Layout.Row    = r;
            app.DropdownMode_.Layout.Column = 1;

            r = r+1;
            app.DropdownColormap_ = uidropdown(app.SideGrid_, ...
                'Items',   app.COLORMAPS, ...
                'Value',   'parula', ...
                'Enable',  'off', ...
                'Tooltip', 'Colormap — active in False Color mode only', ...
                'ValueChangedFcn', @(~,~) app.renderDisplay_());
            app.DropdownColormap_.Layout.Row    = r;
            app.DropdownColormap_.Layout.Column = 1;

            % ---- ANNOTATIONS ------------------------------------------
            r = r+1; app.sectionLabel_(r, 'ANNOTATIONS');
            r = r+1;
            app.BtnAnnotRect_ = uibutton(app.SideGrid_, ...
                'Text', 'Draw Rectangle', ...
                'ButtonPushedFcn', @(~,~) app.drawAnnotation_('rectangle'));
            app.BtnAnnotRect_.Layout.Row = r; app.BtnAnnotRect_.Layout.Column = 1;

            r = r+1;
            app.BtnAnnotCircle_ = uibutton(app.SideGrid_, ...
                'Text', 'Draw Circle', ...
                'ButtonPushedFcn', @(~,~) app.drawAnnotation_('circle'));
            app.BtnAnnotCircle_.Layout.Row = r; app.BtnAnnotCircle_.Layout.Column = 1;

            r = r+1;
            app.BtnAnnotLine_ = uibutton(app.SideGrid_, ...
                'Text', 'Draw Line', ...
                'ButtonPushedFcn', @(~,~) app.drawAnnotation_('line'));
            app.BtnAnnotLine_.Layout.Row = r; app.BtnAnnotLine_.Layout.Column = 1;

            r = r+1;
            app.BtnClearAnnot_ = uibutton(app.SideGrid_, ...
                'Text', 'Clear All Annotations', ...
                'ButtonPushedFcn', @(~,~) app.clearAnnotations_());
            app.BtnClearAnnot_.Layout.Row = r; app.BtnClearAnnot_.Layout.Column = 1;

            % ---- CURSOR -----------------------------------------------
            r = r+1; app.sectionLabel_(r, 'CURSOR');
            r = r+1; app.LblCursor_ = app.infoLabel_(r, 'Click image to read pixel');

            % ---- EXPORT -----------------------------------------------
            r = r+1; app.sectionLabel_(r, 'EXPORT');
            r = r+1;
            app.BtnExport_ = uibutton(app.SideGrid_, ...
                'Text', 'Export View...', ...
                'ButtonPushedFcn', @(~,~) app.exportImage_());
            app.BtnExport_.Layout.Row = r; app.BtnExport_.Layout.Column = 1;

            % filler row (row nRows) — intentionally empty
        end

        % ------------------------------------------------------------------
        function buildViewerPanel_(app)
            app.ViewerPanel_ = uipanel(app.RootGrid_, 'Title', 'Image Viewer');
            app.ViewerPanel_.Layout.Row    = 1;
            app.ViewerPanel_.Layout.Column = 2;
            app.buildSingleViewer_();
        end

        function buildSingleViewer_(app)
            g = uigridlayout(app.ViewerPanel_, [1, 1]);
            g.Padding     = [6, 6, 6, 6];
            g.RowHeight   = {'1x'};
            g.ColumnWidth = {'1x'};

            app.Viewer_ = viewer2d(g);
            app.Viewer_.Layout.Row    = 1;
            app.Viewer_.Layout.Column = 1;

            % imageshow with empty data — Data set later when image loads
            app.Image_ = imageshow([], Parent=app.Viewer_);

            % Respond to clicks for pixel readout
            app.Viewer_.ObjectClickedFcn = @(~, evt) app.onViewerClick_(evt);
        end

        function buildSplitViewers_(app)
            % 3 linked viewer2d objects for Channel Split mode
            g = uigridlayout(app.ViewerPanel_, [1, 3]);
            g.Padding       = [6, 6, 6, 6];
            g.ColumnWidth   = {'1x', '1x', '1x'};
            g.RowHeight     = {'1x'};
            g.ColumnSpacing = 6;

            app.ViewerR_ = viewer2d(g);
            app.ViewerR_.Layout.Row = 1; app.ViewerR_.Layout.Column = 1;

            app.ViewerG_ = viewer2d(g);
            app.ViewerG_.Layout.Row = 1; app.ViewerG_.Layout.Column = 2;

            app.ViewerB_ = viewer2d(g);
            app.ViewerB_.Layout.Row = 1; app.ViewerB_.Layout.Column = 3;

            % Titles — pass viewer object directly (not gca)
            title(app.ViewerR_, 'Red Channel');
            title(app.ViewerG_, 'Green Channel');
            title(app.ViewerB_, 'Blue Channel');

            app.ImageR_ = imageshow([], Parent=app.ViewerR_);
            app.ImageG_ = imageshow([], Parent=app.ViewerG_);
            app.ImageB_ = imageshow([], Parent=app.ViewerB_);

            % Synchronise pan / zoom across all 3 viewers
            linkviewers([app.ViewerR_, app.ViewerG_, app.ViewerB_]);
        end

    end

    % ======================================================================
    % Private — Image Loading
    % ======================================================================
    methods (Access = private)

        function loadPreset_(app, filename)
            % Resolve: <this_file>/../../../Data/images/<filename>
            scriptDir   = fileparts(mfilename('fullpath'));
            projectRoot = fullfile(scriptDir, '..', '..', '..');
            imgPath     = fullfile(projectRoot, 'Data', 'images', filename);

            if isfile(imgPath)
                app.loadImageFromFile_(imgPath);
            else
                % Fall back to a synthetic colour-wheel demo
                app.loadSyntheticDemo_();
                [~, fn, ext] = fileparts(filename);
                app.LblFilename_.Text = sprintf('File: %s%s (not found — synthetic)', fn, ext);
            end
        end

        function loadImageFromFile_(app, path)
            try
                img = imread(path);
                app.CurrentImage_     = img;
                app.CurrentImagePath_ = path;
                app.updateInfoPanel_();
                app.renderDisplay_();
            catch ME
                uialert(app.UIFigure_, ...
                    sprintf('Could not load image:\n%s', ME.message), ...
                    'Load Error');
            end
        end

        function loadSyntheticDemo_(app)
            % Synthetic colour-wheel — always produces a valid RGB uint8
            sz     = 256;
            [X, Y] = meshgrid(linspace(-1,1,sz), linspace(-1,1,sz));
            ang    = atan2(Y, X);
            rad    = sqrt(X.^2 + Y.^2);
            R = uint8(128 + 127*cos(ang));
            G = uint8(128 + 127*cos(ang + 2*pi/3));
            B = uint8(128 + 127*cos(ang + 4*pi/3));
            mask = rad > 1;
            R(mask) = 20; G(mask) = 20; B(mask) = 20;
            app.CurrentImage_     = cat(3, R, G, B);
            app.CurrentImagePath_ = '';
            app.updateInfoPanel_();
            app.renderDisplay_();
        end

    end

    % ======================================================================
    % Private — Display Rendering
    % ======================================================================
    methods (Access = private)

        function renderDisplay_(app)
            if isempty(app.CurrentImage_), return; end

            mode = app.DropdownMode_.Value;
            app.CurrentMode_ = mode;

            if strcmp(mode, 'Channel Split')
                app.switchToSplitMode_();
                app.renderChannelSplit_();
            else
                app.switchToSingleMode_();
                switch mode
                    case 'Normal'
                        app.Image_.Data = app.toDisplay_(app.CurrentImage_);
                    case 'Grayscale'
                        app.Image_.Data = app.toDisplay_(app.toGray_(app.CurrentImage_));
                    case 'False Color'
                        app.Image_.Data = app.applyFalseColor_();
                end
            end
        end

        function out = applyFalseColor_(app)
            % Convert to normalised [0,1] grayscale then map through colormap
            g = im2double(app.toGray_(app.CurrentImage_));
            if size(g, 3) == 3, g = rgb2gray(g); end
            cmap = feval(app.DropdownColormap_.Value, 256);
            idx  = max(1, min(256, round(g * 255) + 1));
            [h, w] = size(idx);
            out  = reshape(cmap(idx(:), :), [h, w, 3]);
        end

        function renderChannelSplit_(app)
            img = app.toDisplay_(app.CurrentImage_);
            if size(img, 3) >= 3
                app.ImageR_.Data = img(:,:,1);
                app.ImageG_.Data = img(:,:,2);
                app.ImageB_.Data = img(:,:,3);
            else
                % Grayscale — show the same channel in all three panes
                app.ImageR_.Data = img;
                app.ImageG_.Data = img;
                app.ImageB_.Data = img;
                title(app.ViewerR_, 'Channel (gray)');
                title(app.ViewerG_, 'Channel (gray)');
                title(app.ViewerB_, 'Channel (gray)');
            end
        end

        function switchToSingleMode_(app)
            if app.SplitModeActive_
                app.deleteViewerPanelChildren_();
                app.SplitModeActive_ = false;
                app.buildSingleViewer_();
            end
        end

        function switchToSplitMode_(app)
            if ~app.SplitModeActive_
                app.deleteViewerPanelChildren_();
                app.Viewer_            = [];
                app.Image_             = [];
                app.AnnotationHandles_ = {};
                app.buildSplitViewers_();
                app.SplitModeActive_   = true;
            end
        end

        function deleteViewerPanelChildren_(app)
            try
                kids = app.ViewerPanel_.Children;
                for k = numel(kids):-1:1
                    delete(kids(k));
                end
            catch
            end
        end

        % ------------------------------------------------------------------
        % Image-type normalisation — any type → uint8 RGB/gray for display
        % ------------------------------------------------------------------
        function out = toDisplay_(~, img)
            % Convert arbitrary class to uint8 displayable image
            switch class(img)
                case 'uint8'
                    out = img;
                case 'uint16'
                    out = uint8(double(img) / 65535 * 255);
                case {'double', 'single'}
                    v  = double(img);
                    mn = min(v(:)); mx = max(v(:));
                    if mx > mn, v = (v - mn) / (mx - mn); end
                    out = uint8(v * 255);
                case 'logical'
                    out = uint8(img) * 255;
                case {'int8', 'int16', 'int32', 'int64', 'uint32', 'uint64'}
                    v  = double(img);
                    mn = min(v(:)); mx = max(v(:));
                    if mx > mn, v = (v - mn) / (mx - mn); end
                    out = uint8(v * 255);
                otherwise
                    out = uint8(img);
            end
            % Multi-channel (>3): show first 3 channels as RGB
            if size(out, 3) > 3
                out = out(:,:,1:3);
            end
        end

        function out = toGray_(~, img)
            % Convert any image to a single-channel grayscale
            if size(img, 3) == 3
                if isa(img, 'double') || isa(img, 'single')
                    out = 0.2126*img(:,:,1) + 0.7152*img(:,:,2) + 0.0722*img(:,:,3);
                else
                    out = rgb2gray(img);
                end
            elseif size(img, 3) > 3
                out = rgb2gray(img(:,:,1:3));
            else
                out = img;
            end
        end

    end

    % ======================================================================
    % Private — Info Panel
    % ======================================================================
    methods (Access = private)

        function updateInfoPanel_(app)
            img = app.CurrentImage_;
            if isempty(img)
                app.LblFilename_.Text   = 'File: —';
                app.LblDimensions_.Text = 'Size: —';
                app.LblClass_.Text      = 'Class: —';
                app.LblChannels_.Text   = 'Channels: —';
                app.LblIntensity_.Text  = 'Range: —';
                return;
            end

            [~, fn, ext] = fileparts(app.CurrentImagePath_);
            if isempty(fn)
                app.LblFilename_.Text = 'File: (synthetic demo)';
            else
                app.LblFilename_.Text = ['File: ', fn, ext];
            end

            [h, w, c] = size(img);
            app.LblDimensions_.Text = sprintf('Size: %d \xD7 %d px', h, w);
            app.LblClass_.Text      = sprintf('Class: %s', class(img));

            switch c
                case 1,  cs = '1 (grayscale)';
                case 3,  cs = '3 (RGB)';
                otherwise, cs = sprintf('%d (multi-channel)', c);
            end
            app.LblChannels_.Text = ['Channels: ', cs];

            try
                mn = double(min(img(:)));
                mx = double(max(img(:)));
                app.LblIntensity_.Text = sprintf('Range: %.4g \x2013 %.4g', mn, mx);
            catch
                app.LblIntensity_.Text = 'Range: —';
            end
        end

    end

    % ======================================================================
    % Private — Annotations
    % ======================================================================
    methods (Access = private)

        function drawAnnotation_(app, shape)
            if app.SplitModeActive_ || isempty(app.Image_) || isempty(app.Image_.Data)
                uialert(app.UIFigure_, ...
                    'Load an image and use Normal, Grayscale, or False Color mode first.', ...
                    'Annotations Unavailable');
                return;
            end
            try
                roi = uidraw(app.Image_, shape, Color=[0, 0.45, 0.9]);
                app.AnnotationHandles_{end+1} = roi;
            catch ME
                uialert(app.UIFigure_, ...
                    sprintf('Annotation failed:\n%s', ME.message), 'Error');
            end
        end

        function clearAnnotations_(app)
            for i = 1:numel(app.AnnotationHandles_)
                try
                    if ~isempty(app.AnnotationHandles_{i}) && ...
                            isvalid(app.AnnotationHandles_{i})
                        delete(app.AnnotationHandles_{i});
                    end
                catch
                end
            end
            app.AnnotationHandles_ = {};
        end

    end

    % ======================================================================
    % Private — Export
    % ======================================================================
    methods (Access = private)

        function exportImage_(app)
            if isempty(app.CurrentImage_)
                uialert(app.UIFigure_, 'Load an image first.', 'Export');
                return;
            end
            [file, path] = uiputfile( ...
                {'*.png','PNG Image (*.png)'; ...
                 '*.jpg','JPEG Image (*.jpg)'; ...
                 '*.tiff','TIFF Image (*.tiff)'}, ...
                'Export Image As', 'exported_image.png');
            if isequal(file, 0), return; end
            try
                imwrite(app.buildExportImage_(), fullfile(path, file));
                uialert(app.UIFigure_, ['Saved: ', file], ...
                    'Export Complete', 'Icon', 'success');
            catch ME
                uialert(app.UIFigure_, ...
                    sprintf('Export failed:\n%s', ME.message), 'Error');
            end
        end

        function img = buildExportImage_(app)
            switch app.CurrentMode_
                case 'Normal'
                    img = app.toDisplay_(app.CurrentImage_);
                case 'Grayscale'
                    img = app.toDisplay_(app.toGray_(app.CurrentImage_));
                case 'False Color'
                    img = uint8(app.applyFalseColor_() * 255);
                case 'Channel Split'
                    n = app.toDisplay_(app.CurrentImage_);
                    if size(n, 3) >= 3
                        z = zeros(size(n,1), size(n,2), 'uint8');
                        r = cat(3, n(:,:,1), z, z);
                        g = cat(3, z, n(:,:,2), z);
                        b = cat(3, z, z, n(:,:,3));
                        img = [r, g, b];   % tiled side-by-side
                    else
                        img = n;
                    end
                otherwise
                    img = app.toDisplay_(app.CurrentImage_);
            end
        end

    end

    % ======================================================================
    % Private — Callbacks
    % ======================================================================
    methods (Access = private)

        function onPresetChanged_(app)
            app.loadPreset_(app.DropdownPreset_.Value);
        end

        function onLoadCustom_(app)
            [file, path] = uigetfile( ...
                {'*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.tif;*.gif', ...
                 'Image Files (*.jpg,*.png,*.bmp,*.tiff,*.gif)'; ...
                 '*.*', 'All Files (*.*)'}, ...
                'Load Image');
            if isequal(file, 0), return; end
            app.loadImageFromFile_(fullfile(path, file));
        end

        function onModeChanged_(app)
            isFalseColor = strcmp(app.DropdownMode_.Value, 'False Color');
            if isFalseColor
                app.DropdownColormap_.Enable = 'on';
            else
                app.DropdownColormap_.Enable = 'off';
            end
            app.renderDisplay_();
        end

        function onViewerClick_(app, evt)
            % Update cursor readout with pixel coordinates + value
            try
                x = round(evt.Position(1));
                y = round(evt.Position(2));
                img = app.CurrentImage_;
                if ~isempty(img) && x >= 1 && y >= 1 && ...
                        x <= size(img,2) && y <= size(img,1)
                    if size(img, 3) == 1
                        val = sprintf('%.4g', double(img(y, x)));
                    else
                        v   = double(squeeze(img(y, x, :)))';
                        val = ['[', strtrim(sprintf('%g ', v)), ']'];
                    end
                    app.LblCursor_.Text = sprintf('x:%d  y:%d  |  val: %s', x, y, val);
                end
            catch
                % Pixel readout is best-effort — ignore errors silently
            end
        end

        function onClose_(app)
            app.clearAnnotations_();
            delete(app);
        end

    end

    % ======================================================================
    % Private — Sidebar UI Helpers
    % ======================================================================
    methods (Access = private)

        function sectionLabel_(app, row, text)
            lbl = uilabel(app.SideGrid_, ...
                'Text',              text, ...
                'FontWeight',        'bold', ...
                'FontSize',          10, ...
                'FontColor',         [0.3, 0.3, 0.3], ...
                'HorizontalAlignment','left');
            lbl.Layout.Row    = row;
            lbl.Layout.Column = 1;
        end

        function lbl = infoLabel_(app, row, text)
            lbl = uilabel(app.SideGrid_, ...
                'Text',      text, ...
                'FontSize',  10, ...
                'Interpreter','none', ...
                'WordWrap',  'on');
            lbl.Layout.Row    = row;
            lbl.Layout.Column = 1;
        end

    end
end
