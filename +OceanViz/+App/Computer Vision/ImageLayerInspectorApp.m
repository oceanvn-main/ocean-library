classdef ImageLayerInspectorApp < handle
    % IMAGELAYERINSPECTORAPP  GUI for inspecting image internal representation.
    %
    % Displays 5 organized layers (from highest abstraction to disk storage):
    %   Layer 0 — Rendering (Final screen display & physical output metrics)
    %   Layer 1 — Decoded Pixels (Spatial pixel matrices & center 8×8 block zoom)
    %   Layer 2 — DCT & Quantization (Frequency domain & quantization table heatmaps)
    %   Layer 3 — Binary Encoding (Bitstream representation & memory byte buffer)
    %   Layer 4 — Raw File Bytes (Raw file bytes & format markers on disk)
    %
    % Usage:
    %   app = ImageLayerInspectorApp();
    %
    % Author: Ocean Library
    % Date:   2026-09-16

    % ======================================================================
    properties (Access = private)
        % Figure & root layout
        UIFigure_    matlab.ui.Figure
        RootGrid_    matlab.ui.container.GridLayout

        % Sidebar
        SidePanel_   matlab.ui.container.Panel
        SideGrid_    matlab.ui.container.GridLayout
        BtnLoad_     matlab.ui.control.Button
        LblFile_     matlab.ui.control.Label
        LblRes_      matlab.ui.control.Label
        LblClass_    matlab.ui.control.Label
        LblFormat_   matlab.ui.control.Label
        LblFileSize_ matlab.ui.control.Label
        LblStatus_   matlab.ui.control.Label

        % Tab group
        TabGroup_    matlab.ui.container.TabGroup

        % Layer 0 — Rendering
        Tab0_
        Viewer0_
        Image0_
        TextRender_  matlab.ui.control.TextArea

        % Layer 1 — Decoded Pixels
        Tab1_
        Viewer1_
        Image1_
        AxBlock_     matlab.ui.control.UIAxes
        TextL1_      matlab.ui.control.TextArea

        % Layer 2 — DCT & Quantization
        Tab2_
        AxQuant_     matlab.ui.control.UIAxes
        AxDCTRaw_    matlab.ui.control.UIAxes
        AxDCTQuant_  matlab.ui.control.UIAxes
        TextL2_      matlab.ui.control.TextArea

        % Layer 3 — Binary Encoding
        Tab3_
        TextL3_      matlab.ui.control.TextArea

        % Layer 4 — Raw File Bytes
        Tab4_
        TextL4_      matlab.ui.control.TextArea

        % Analysis data
        CurrentFilePath_  = ''
        Info_             struct
        IsJPEG_           = false
    end

    % ======================================================================
    methods (Access = public)

        function app = ImageLayerInspectorApp()
            app.buildUI_();
        end

        function delete(app)
            if ~isempty(app.UIFigure_) && isvalid(app.UIFigure_)
                delete(app.UIFigure_);
            end
        end
    end

    % ======================================================================
    % UI Construction
    % ======================================================================
    methods (Access = private)

        function buildUI_(app)
            scr = get(0, 'ScreenSize');
            W = 1280; H = 780;
            app.UIFigure_ = uifigure( ...
                'Name',     'Image Layer Inspector — Ocean Library', ...
                'Position', [(scr(3)-W)/2, (scr(4)-H)/2, W, H], ...
                'CloseRequestFcn', @(~,~) delete(app));

            app.RootGrid_ = uigridlayout(app.UIFigure_, [1, 2]);
            app.RootGrid_.ColumnWidth   = {240, '1x'};
            app.RootGrid_.RowHeight     = {'1x'};
            app.RootGrid_.Padding       = [0, 0, 0, 0];
            app.RootGrid_.ColumnSpacing = 0;

            app.buildSidebar_();
            app.buildTabArea_();
        end

        % ------------------------------------------------------------------
        function buildSidebar_(app)
            app.SidePanel_ = uipanel(app.RootGrid_, 'Title', '');
            app.SidePanel_.Layout.Row    = 1;
            app.SidePanel_.Layout.Column = 1;

            nRows = 16;
            rh    = repmat({'fit'}, 1, nRows);
            rh{nRows} = '1x';

            app.SideGrid_ = uigridlayout(app.SidePanel_, [nRows, 1]);
            app.SideGrid_.RowHeight    = rh;
            app.SideGrid_.ColumnWidth  = {'1x'};
            app.SideGrid_.Padding      = [12, 12, 12, 12];
            app.SideGrid_.RowSpacing   = 6;

            r = 0;

            r=r+1; app.sideSection_(r, 'IMAGE SOURCE');
            r=r+1;
            app.BtnLoad_ = uibutton(app.SideGrid_, ...
                'Text',            'Load Image...', ...
                'ButtonPushedFcn', @(~,~) app.onLoadImage_());
            app.BtnLoad_.Layout.Row = r; app.BtnLoad_.Layout.Column = 1;

            r=r+1; app.sideSection_(r, 'FILE INFO');
            r=r+1; app.LblFile_     = app.sideInfo_(r, 'File: —');
            r=r+1; app.LblFormat_   = app.sideInfo_(r, 'Format: —');
            r=r+1; app.LblFileSize_ = app.sideInfo_(r, 'Disk size: —');
            r=r+1; app.LblRes_      = app.sideInfo_(r, 'Resolution: —');
            r=r+1; app.LblClass_    = app.sideInfo_(r, 'Class: —');

            r=r+1; app.sideSection_(r, 'STATUS');
            r=r+1;
            app.LblStatus_ = uilabel(app.SideGrid_, ...
                'Text',      'Load an image to begin.', ...
                'FontSize',  10, ...
                'FontColor', [0.4, 0.4, 0.4], ...
                'WordWrap',  'on');
            app.LblStatus_.Layout.Row = r; app.LblStatus_.Layout.Column = 1;

            r=r+1; app.sideSection_(r, 'LAYER 2 NOTE');
            r=r+1;
            note = uilabel(app.SideGrid_, ...
                'Text',     ['DCT quant table parsed from JPEG DQT marker. ' ...
                             'For PNG/BMP/TIFF the standard baseline table is used.'], ...
                'FontSize', 9, 'FontColor', [0.5 0.5 0.5], 'WordWrap', 'on');
            note.Layout.Row = r; note.Layout.Column = 1;
        end

        % ------------------------------------------------------------------
        function buildTabArea_(app)
            mainPanel = uipanel(app.RootGrid_, 'Title', '');
            mainPanel.Layout.Row    = 1;
            mainPanel.Layout.Column = 2;

            outerG = uigridlayout(mainPanel, [1,1]);
            outerG.Padding = [6, 6, 6, 6];
            outerG.RowHeight   = {'1x'};
            outerG.ColumnWidth = {'1x'};

            app.TabGroup_ = uitabgroup(outerG);
            app.TabGroup_.Layout.Row    = 1;
            app.TabGroup_.Layout.Column = 1;

            app.buildTab0_();   % Layer 0: Rendering (Display)
            app.buildTab1_();   % Layer 1: Decoded Pixels
            app.buildTab2_();   % Layer 2: DCT & Quantization
            app.buildTab3_();   % Layer 3: Binary Encoding
            app.buildTab4_();   % Layer 4: Raw File Bytes
        end

        % --- Tab 0: Layer 0 — Rendering (Display) ---
        function buildTab0_(app)
            app.Tab0_ = uitab(app.TabGroup_, 'Title', 'Layer 0 — Rendering (Display)');
            g = uigridlayout(app.Tab0_, [2, 1]);
            g.RowHeight   = {'1x', 80};
            g.ColumnWidth = {'1x'};
            g.Padding     = [8, 8, 8, 8];
            g.RowSpacing  = 6;

            % Full-size interactive viewer
            vGrid = uigridlayout(g, [1, 1]);
            vGrid.Layout.Row    = 1;
            vGrid.Layout.Column = 1;
            vGrid.Padding       = [0, 0, 0, 0];
            vGrid.RowHeight     = {'1x'};
            vGrid.ColumnWidth   = {'1x'};

            app.Viewer0_ = viewer2d(vGrid);
            app.Viewer0_.Layout.Row    = 1;
            app.Viewer0_.Layout.Column = 1;
            app.Image0_  = imageshow([], Parent=app.Viewer0_);

            % Display metadata bar
            app.TextRender_ = uitextarea(g, ...
                'Value',    {'Load an image to see the rendered display.'}, ...
                'FontName', 'Courier New', ...
                'FontSize', 10, ...
                'Editable', 'off');
            app.TextRender_.Layout.Row    = 2;
            app.TextRender_.Layout.Column = 1;
        end

        % --- Tab 1: Layer 1 — Decoded Pixels ---
        function buildTab1_(app)
            app.Tab1_ = uitab(app.TabGroup_, 'Title', 'Layer 1 — Decoded Pixels');
            g = uigridlayout(app.Tab1_, [2, 2]);
            g.RowHeight   = {'1x', '1x'};
            g.ColumnWidth = {'1x', '1x'};
            g.Padding     = [8, 8, 8, 8];
            g.RowSpacing  = 6;
            g.ColumnSpacing = 8;

            % Full image viewer spanning both columns
            vPanel = uipanel(g, 'Title', 'Full Decoded Image');
            vPanel.Layout.Row    = 1;
            vPanel.Layout.Column = [1, 2];

            vGrid = uigridlayout(vPanel, [1,1]);
            vGrid.Padding = [4,4,4,4];
            vGrid.RowHeight   = {'1x'};
            vGrid.ColumnWidth = {'1x'};

            app.Viewer1_ = viewer2d(vGrid);
            app.Viewer1_.Layout.Row    = 1;
            app.Viewer1_.Layout.Column = 1;
            app.Image1_  = imageshow([], Parent=app.Viewer1_);

            % 8x8 block zoom (pixelated)
            bPanel = uipanel(g, 'Title', 'Center 8×8 Block (nearest-neighbour zoom)');
            bPanel.Layout.Row    = 2;
            bPanel.Layout.Column = 1;
            bGrid = uigridlayout(bPanel, [1,1]);
            bGrid.Padding = [4,4,4,4];
            bGrid.RowHeight   = {'1x'};
            bGrid.ColumnWidth = {'1x'};
            app.AxBlock_ = uiaxes(bGrid);
            app.AxBlock_.Layout.Row = 1; app.AxBlock_.Layout.Column = 1;
            app.AxBlock_.XTick = []; app.AxBlock_.YTick = [];
            axis(app.AxBlock_, 'image');

            % Pixel matrix text
            app.TextL1_ = uitextarea(g, ...
                'Value',    {'Load an image.'}, ...
                'FontName', 'Courier New', ...
                'FontSize', 10, ...
                'Editable', 'off');
            app.TextL1_.Layout.Row    = 2;
            app.TextL1_.Layout.Column = 2;
        end

        % --- Tab 2: Layer 2 — DCT & Quantization ---
        function buildTab2_(app)
            app.Tab2_ = uitab(app.TabGroup_, 'Title', 'Layer 2 — DCT & Quantization');
            g = uigridlayout(app.Tab2_, [3, 3]);
            g.RowHeight   = {'1x', 'fit', '1x'};
            g.ColumnWidth = {'1x', '1x', '1x'};
            g.Padding     = [8, 8, 8, 8];
            g.RowSpacing  = 4;
            g.ColumnSpacing = 8;

            % Three heatmap axes
            app.AxQuant_ = uiaxes(g);
            app.AxQuant_.Layout.Row = 1; app.AxQuant_.Layout.Column = 1;
            title(app.AxQuant_, 'Quant Table — Luma');

            app.AxDCTRaw_ = uiaxes(g);
            app.AxDCTRaw_.Layout.Row = 1; app.AxDCTRaw_.Layout.Column = 2;
            title(app.AxDCTRaw_, 'DCT Coefficients (raw)');

            app.AxDCTQuant_ = uiaxes(g);
            app.AxDCTQuant_.Layout.Row = 1; app.AxDCTQuant_.Layout.Column = 3;
            title(app.AxDCTQuant_, 'Quantized DCT');

            % Note row
            lbl = uilabel(g, ...
                'Text',     'Heatmaps above show the 8×8 luma block. Matrix values below.', ...
                'FontSize', 10, 'FontColor', [0.4 0.4 0.4]);
            lbl.Layout.Row = 2; lbl.Layout.Column = [1, 3];

            % Text area spanning bottom row
            app.TextL2_ = uitextarea(g, ...
                'Value',    {'Load an image.'}, ...
                'FontName', 'Courier New', ...
                'FontSize', 10, ...
                'Editable', 'off');
            app.TextL2_.Layout.Row = 3; app.TextL2_.Layout.Column = [1, 3];
        end

        % --- Tab 3: Layer 3 — Binary Encoding ---
        function buildTab3_(app)
            app.Tab3_ = uitab(app.TabGroup_, 'Title', 'Layer 3 — Binary Encoding');
            g = uigridlayout(app.Tab3_, [2, 1]);
            g.RowHeight   = {'fit', '1x'};
            g.ColumnWidth = {'1x'};
            g.Padding     = [8, 8, 8, 8];
            g.RowSpacing  = 6;

            hdr = uilabel(g, ...
                'Text',       'Layer 3 — Binary Encoding & Pixel Memory Layout', ...
                'FontWeight', 'bold', 'FontSize', 11);
            hdr.Layout.Row = 1; hdr.Layout.Column = 1;

            app.TextL3_ = uitextarea(g, ...
                'Value',    {'Load an image.'}, ...
                'FontName', 'Courier New', ...
                'FontSize', 11, ...
                'Editable', 'off');
            app.TextL3_.Layout.Row = 2; app.TextL3_.Layout.Column = 1;
        end

        % --- Tab 4: Layer 4 — Raw File Bytes ---
        function buildTab4_(app)
            app.Tab4_ = uitab(app.TabGroup_, 'Title', 'Layer 4 — Raw File Bytes');
            g = uigridlayout(app.Tab4_, [2, 1]);
            g.RowHeight   = {'fit', '1x'};
            g.ColumnWidth = {'1x'};
            g.Padding     = [8, 8, 8, 8];
            g.RowSpacing  = 6;

            hdr = uilabel(g, ...
                'Text',       'Layer 4 — Raw File Bytes & Format Signatures (Disk Storage)', ...
                'FontWeight', 'bold', 'FontSize', 11);
            hdr.Layout.Row = 1; hdr.Layout.Column = 1;

            app.TextL4_ = uitextarea(g, ...
                'Value',       {'Load an image to inspect its raw file bytes.'}, ...
                'FontName',    'Courier New', ...
                'FontSize',    11, ...
                'Editable',    'off');
            app.TextL4_.Layout.Row = 2; app.TextL4_.Layout.Column = 1;
        end
    end

    % ======================================================================
    % Callbacks & Analysis
    % ======================================================================
    methods (Access = private)

        function onLoadImage_(app)
            [file, path] = uigetfile( ...
                {'*.jpg;*.jpeg', 'JPEG Images'; ...
                 '*.png',        'PNG Images'; ...
                 '*.bmp',        'BMP Images'; ...
                 '*.tiff;*.tif', 'TIFF Images'; ...
                 '*.gif',        'GIF Images'; ...
                 '*.*',          'All Image Files'}, ...
                'Load Image File');
            if isequal(file, 0), return; end

            filepath = fullfile(path, file);
            app.CurrentFilePath_ = filepath;
            app.setStatus_('Analysing…');
            drawnow;

            try
                app.runAnalysis_(filepath);
                app.setStatus_('Ready.');
            catch ME
                app.setStatus_(['Error: ', ME.message]);
                uialert(app.UIFigure_, ME.message, 'Analysis Error');
            end
        end

        % ------------------------------------------------------------------
        function runAnalysis_(app, filepath)
            % ---- Layer 1: raw bytes
            fid      = fopen(filepath, 'rb');
            rawBytes = fread(fid, '*uint8');
            fclose(fid);

            % ---- Detect format
            [~, fn, ext] = fileparts(filepath);
            app.IsJPEG_ = ismember(lower(ext), {'.jpg', '.jpeg'});
            fmtStr      = app.detectFormat_(rawBytes);

            % ---- Sidebar info
            img = imread(filepath);
            if size(img, 3) == 1, img = repmat(img, [1,1,3]); end
            [H, W, C] = size(img);
            app.LblFile_.Text     = ['File: ', fn, ext];
            app.LblFormat_.Text   = ['Format: ', fmtStr];
            app.LblFileSize_.Text = sprintf('Disk size: %s', app.humanBytes_(numel(rawBytes)));
            app.LblRes_.Text      = sprintf('Resolution: %d × %d', W, H);
            app.LblClass_.Text    = sprintf('Class: %s  Ch: %d', class(img), C);

            % ---- Build info struct
            info.rawBytes  = rawBytes;
            info.fullImage = img;

            % --- Quant tables
            [qL, qC] = app.parseQuantTables_(rawBytes);
            info.quantLuma   = qL;
            info.quantChroma = qC;

            % --- Centre 8×8 block coordinates
            r0 = max(1, floor((H - 8) / 2) + 1);
            c0 = max(1, floor((W - 8) / 2) + 1);
            r1 = min(H, r0 + 7);
            c1 = min(W, c0 + 7);
            info.blockOrigin = [r0, c0];

            % --- Centre 8×8 pixel block
            block8 = img(r0:r1, c0:c1, :);
            if size(block8,1) < 8 || size(block8,2) < 8
                block8 = padarray(block8, [8-size(block8,1), 8-size(block8,2)], 0, 'post');
            end
            info.block8 = block8;

            % --- DCT of centre 8×8 luma block
            if H >= 8 && W >= 8
                ycbcr = rgb2ycbcr(uint8(img));
                luma  = double(ycbcr(r0:r1, c0:c1, 1)) - 128;
            else
                luma  = double(block8(:,:,1)) - 128;
            end
            if size(luma,1) < 8 || size(luma,2) < 8
                luma = padarray(luma, [8-size(luma,1), 8-size(luma,2)], 0, 'post');
            end
            dctRaw   = dct2(luma);
            dctQuant = round(dctRaw ./ info.quantLuma);
            info.dctRaw   = dctRaw;
            info.dctQuant = dctQuant;

            app.Info_ = info;

            % ---- Update all tabs in top-down hierarchy order
            app.updateLayer0_Rendering_(img);   % auto-selects Layer 0 tab
            app.updateLayer1_Pixels_(img, info);
            app.updateLayer2_DCT_(info);
            app.updateLayer3_Binary_(info.block8);
            app.updateLayer4_RawBytes_(rawBytes, fmtStr);
        end

        % ------------------------------------------------------------------
        function updateLayer0_Rendering_(app, img)
            % Show the decoded image exactly as the display renders it
            app.Image0_.Data = img;

            % Auto-switch to Rendering tab on load
            app.TabGroup_.SelectedTab = app.Tab0_;

            % Gather display metadata
            dpi      = get(0, 'ScreenPixelsPerInch');
            screenSz = get(0, 'ScreenSize');
            [H, W, C] = size(img);

            switch class(img)
                case 'uint8',  bpc = 8;
                case 'uint16', bpc = 16;
                case 'double', bpc = 64;
                case 'single', bpc = 32;
                otherwise,     bpc = 8;
            end

            physW_cm = (W / dpi) * 2.54;
            physH_cm = (H / dpi) * 2.54;

            lines = {};
            lines{end+1} = sprintf( ...
                '  Image    : %d x %d px   %d ch   %s   %d bits/ch', ...
                W, H, C, class(img), bpc);
            lines{end+1} = sprintf( ...
                '  Screen   : %d x %d px   @ %.0f PPI', ...
                screenSz(3), screenSz(4), dpi);
            lines{end+1} = sprintf( ...
                '  Physical : %.1f x %.1f cm at 100%% zoom  (%.1f x %.1f in)', ...
                physW_cm, physH_cm, physW_cm/2.54, physH_cm/2.54);
            lines{end+1} = ...
                '  Pipeline : File bytes -> Huffman decode -> IDCT -> YCbCr -> sRGB -> GPU raster -> Display';

            app.TextRender_.Value = lines;
        end

        % ------------------------------------------------------------------
        function updateLayer1_Pixels_(app, img, info)
            block8 = info.block8;

            % Full image in viewer2d
            app.Image1_.Data = img;

            % Centre 8×8 block — upscale 24× with nearest for pixelated view
            scale  = 24;
            scaled = imresize(block8, scale, 'nearest');
            imshow(uint8(scaled), 'Parent', app.AxBlock_);
            app.AxBlock_.XTick = []; app.AxBlock_.YTick = [];

            % Pixel matrix text
            [H, W, ~] = size(img);
            lines = {};
            lines{end+1} = sprintf('Block shape   : 8 × 8 × 3  (uint8 RGB)');
            lines{end+1} = sprintf('Sample origin : row %d, col %d  (centre of %d × %d image)', ...
                info.blockOrigin(1), info.blockOrigin(2), H, W);
            lines{end+1} = sprintf('Pixel [1,1]   : R=%3d  G=%3d  B=%3d', ...
                block8(1,1,1), block8(1,1,2), block8(1,1,3));
            lines{end+1} = '';

            chNames = {'Red', 'Green', 'Blue'};
            for ch = 1:3
                lines{end+1} = sprintf('%s channel (8×8):', chNames{ch}); %#ok<AGROW>
                lines = [lines, app.matToLines_(double(block8(:,:,ch)), '%4d')]; %#ok<AGROW>
                lines{end+1} = ''; %#ok<AGROW>
            end

            app.TextL1_.Value = lines;
        end

        % ------------------------------------------------------------------
        function updateLayer2_DCT_(app, info)
            % --- Heatmaps
            app.drawHeatmap_(app.AxQuant_,    info.quantLuma, 'Quant Table — Luma',          'hot');
            app.drawHeatmap_(app.AxDCTRaw_,   info.dctRaw,    'DCT Coefficients (raw)',       'rdbu');
            app.drawHeatmap_(app.AxDCTQuant_, info.dctQuant,  'Quantized DCT (÷ quant table)','rdbu');

            % --- Text
            lines = {};
            if app.IsJPEG_
                lines{end+1} = 'Source: DQT marker parsed from JPEG bitstream (FF DB segment)';
            else
                lines{end+1} = 'Source: JPEG baseline standard table (quant table not in file for this format)';
            end
            lines{end+1} = '';
            lines{end+1} = 'Quant Table — Luma (8×8):';
            lines = [lines, app.matToLines_(info.quantLuma, '%4d')];
            lines{end+1} = '';
            lines{end+1} = 'DCT Coefficients, center 8×8 luma block (level-shifted by -128 before DCT):';
            lines = [lines, app.matToLines_(info.dctRaw, '%8.2f')];
            lines{end+1} = '';
            lines{end+1} = 'Quantized DCT = round(DCT_raw ./ quant_table):';
            lines = [lines, app.matToLines_(info.dctQuant, '%5d')];

            app.TextL2_.Value = lines;
        end

        % ------------------------------------------------------------------
        function updateLayer3_Binary_(app, block8)
            px = squeeze(block8(1, 1, :));
            R  = px(1); G = px(2); B = px(3);

            % Raw memory bytes from the center 8x8x3 block
            rawMem = block8(:);          % column-major flattening
            maxHex = min(64, numel(rawMem));

            lines = {};
            lines{end+1} = 'Binary encoding of pixel values (dtype: uint8)';
            lines{end+1} = repmat('─', 1, 52);
            lines{end+1} = sprintf('Center Pixel [1,1] : [R=%d  G=%d  B=%d]', R, G, B);
            lines{end+1} = '';
            lines{end+1} = sprintf('R = %-3d   →   %s', R, dec2bin(R, 8));
            lines{end+1} = sprintf('G = %-3d   →   %s', G, dec2bin(G, 8));
            lines{end+1} = sprintf('B = %-3d   →   %s', B, dec2bin(B, 8));
            lines{end+1} = '';
            lines{end+1} = repmat('─', 1, 52);
            lines{end+1} = sprintf('Raw memory bytes — first %d of 8×8×3=%d bytes:', ...
                maxHex, numel(rawMem));
            lines{end+1} = '(Column-major layout: R[1,1] G[1,1] B[1,1] R[2,1] G[2,1] B[2,1] ...)';
            lines{end+1} = '';

            % Format as 16 hex pairs per row
            chunk = rawMem(1:maxHex);
            for i = 1:16:maxHex
                seg = chunk(i : min(i+15, maxHex));
                lines{end+1} = strtrim(sprintf('%02X ', seg)); %#ok<AGROW>
            end

            lines{end+1} = '';
            lines{end+1} = repmat('─', 1, 52);
            lines{end+1} = 'Center 8×8 block pixel values — binary per channel:';
            lines{end+1} = '';

            chNames = {'R', 'G', 'B'};
            for ch = 1:3
                slice = block8(:,:,ch);
                row   = slice(1,:);   % first row of 8 pixels
                binStr = '';
                for k = 1:numel(row)
                    binStr = [binStr, sprintf('%s ', dec2bin(row(k), 8))]; %#ok<AGROW>
                end
                lines{end+1} = sprintf('%s row 1: %s', chNames{ch}, strtrim(binStr)); %#ok<AGROW>
            end

            app.TextL3_.Value = lines;
        end

        % ------------------------------------------------------------------
        function updateLayer4_RawBytes_(app, rawBytes, fmtStr)
            n       = numel(rawBytes);
            maxShow = min(n, 512);    % show up to 512 bytes
            bytes   = rawBytes(1:maxShow);

            lines = {};
            lines{end+1} = sprintf('Format detected : %s', fmtStr);
            lines{end+1} = sprintf('File size       : %d bytes  (%s)', n, app.humanBytes_(n));
            lines{end+1} = '';
            lines{end+1} = sprintf('Raw hex bytes on disk — first %d of %d bytes:', maxShow, n);
            lines{end+1} = '';

            % 16 bytes per row with offset
            for i = 1:16:maxShow
                chunk  = bytes(i : min(i+15, maxShow));
                hexStr = sprintf('%02X ', chunk);
                lines{end+1} = sprintf('%06X  %s', i-1, hexStr); %#ok<AGROW>
            end

            if n > maxShow
                lines{end+1} = '';
                lines{end+1} = sprintf('... (%d bytes not shown)', n - maxShow);
            end

            app.TextL4_.Value = lines;
        end

    end

    % ======================================================================
    % Analysis Helpers
    % ======================================================================
    methods (Access = private)

        function [quantLuma, quantChroma] = parseQuantTables_(app, rawBytes)
            quantLuma   = [];
            quantChroma = [];
            zz = app.zigzagOrder_();
            i  = 1; n = numel(rawBytes);

            while i <= n - 1
                if rawBytes(i) == 0xFF && rawBytes(i+1) == 0xDB
                    if i + 3 > n, break; end
                    segLen = double(rawBytes(i+2))*256 + double(rawBytes(i+3));
                    segEnd = i + 1 + segLen;
                    ptr    = i + 4;
                    while ptr < segEnd && ptr <= n
                        pqtq = rawBytes(ptr);
                        pq   = bitshift(pqtq, -4);
                        tq   = bitand(pqtq, 0x0F);
                        ptr  = ptr + 1;
                        if pq == 0
                            if ptr + 63 > n, break; end
                            vals = double(rawBytes(ptr:ptr+63));
                            ptr  = ptr + 64;
                        else
                            if ptr + 127 > n, break; end
                            hi   = double(rawBytes(ptr:2:ptr+126));
                            lo   = double(rawBytes(ptr+1:2:ptr+127));
                            vals = hi*256 + lo;
                            ptr  = ptr + 128;
                        end
                        tbl = zeros(8, 8);
                        for k = 1:64
                            [r, c] = ind2sub([8,8], zz(k));
                            tbl(r,c) = vals(k);
                        end
                        if tq == 0,     quantLuma   = tbl;
                        elseif tq == 1, quantChroma = tbl; end
                    end
                    i = segEnd + 1;
                else
                    i = i + 1;
                end
            end

            if isempty(quantLuma),   quantLuma   = app.defaultQuantLuma_();   end
            if isempty(quantChroma), quantChroma = app.defaultQuantChroma_(); end
        end

        function idx = zigzagOrder_(~)
            p = [0,0;0,1;1,0;2,0;1,1;0,2;0,3;1,2;2,1;3,0;4,0;3,1;2,2;1,3;0,4;0,5; ...
                 1,4;2,3;3,2;4,1;5,0;6,0;5,1;4,2;3,3;2,4;1,5;0,6;0,7;1,6;2,5;3,4; ...
                 4,3;5,2;6,1;7,0;7,1;6,2;5,3;4,4;3,5;2,6;1,7;2,7;3,6;4,5;5,4;6,3; ...
                 7,2;7,3;6,4;5,5;4,6;3,7;4,7;5,6;6,5;7,4;7,5;6,6;5,7;6,7;7,6;7,7];
            idx = sub2ind([8,8], p(:,1)+1, p(:,2)+1);
        end

        function Q = defaultQuantLuma_(~)
            Q = [16,11,10,16,24, 40, 51, 61; ...
                 12,12,14,19,26, 58, 60, 55; ...
                 14,13,16,24,40, 57, 69, 56; ...
                 14,17,22,29,51, 87, 80, 62; ...
                 18,22,37,56,68, 109,103,77; ...
                 24,35,55,64,81, 104,113,92; ...
                 49,64,78,87,103,121,120,101; ...
                 72,92,95,98,112,100,103,99];
        end

        function Q = defaultQuantChroma_(~)
            Q = [17,18,24,47,99,99,99,99; ...
                 18,21,26,66,99,99,99,99; ...
                 24,26,56,99,99,99,99,99; ...
                 47,66,99,99,99,99,99,99; ...
                 99,99,99,99,99,99,99,99; ...
                 99,99,99,99,99,99,99,99; ...
                 99,99,99,99,99,99,99,99; ...
                 99,99,99,99,99,99,99,99];
        end

        function fmt = detectFormat_(~, bytes)
            if numel(bytes) >= 2 && bytes(1)==0xFF && bytes(2)==0xD8
                fmt = 'JPEG  (SOI = FF D8)';
            elseif numel(bytes) >= 4 && all(bytes(1:4)==[0x89;0x50;0x4E;0x47])
                fmt = 'PNG   (magic = 89 50 4E 47)';
            elseif numel(bytes) >= 2 && bytes(1)==0x42 && bytes(2)==0x4D
                fmt = 'BMP   (magic = 42 4D)';
            elseif numel(bytes) >= 4 && (all(bytes(1:4)==[0x49;0x49;0x2A;0x00]) || ...
                                          all(bytes(1:4)==[0x4D;0x4D;0x00;0x2A]))
                fmt = 'TIFF  (II or MM byte order)';
            elseif numel(bytes) >= 6 && all(bytes(1:6)==[0x47;0x49;0x46;0x38;0x39;0x61])
                fmt = 'GIF89a';
            else
                fmt = 'Unknown';
            end
        end

        function drawHeatmap_(~, ax, M, ttl, cmapName)
            imagesc(ax, M);
            if strcmp(cmapName, 'rdbu')
                % Red-White-Blue diverging colourmap (built-in approximation)
                n   = 64;
                r   = [linspace(0.7,1,n/2), linspace(1,0.1,n/2)]';
                g   = [linspace(0.1,1,n/2), linspace(1,0.1,n/2)]';
                b   = [linspace(0.1,1,n/2), linspace(1,0.7,n/2)]';
                cmap = [r, g, b];
            else
                cmap = feval(cmapName, 64);
            end
            colormap(ax, cmap);
            colorbar(ax);
            title(ax, ttl, 'FontSize', 10);
            ax.XTick = 1:8; ax.YTick = 1:8;
            ax.XTickLabel = 1:8; ax.YTickLabel = 1:8;
            axis(ax, 'image');
        end

        function lines = matToLines_(~, M, fmt)
            lines = {};
            for r = 1:size(M, 1)
                lines{end+1} = ['  [', sprintf(fmt, M(r,:)), ' ]']; %#ok<AGROW>
            end
        end

        function s = humanBytes_(~, n)
            if n < 1024, s = sprintf('%d B', n);
            elseif n < 1024^2, s = sprintf('%.1f KB', n/1024);
            else, s = sprintf('%.2f MB', n/1024^2);
            end
        end

        function setStatus_(app, msg)
            app.LblStatus_.Text = msg;
            drawnow limitrate;
        end

        % Sidebar helpers
        function sideSection_(app, row, text)
            lbl = uilabel(app.SideGrid_, 'Text', text, ...
                'FontWeight', 'bold', 'FontSize', 10, 'FontColor', [0.3 0.3 0.3]);
            lbl.Layout.Row = row; lbl.Layout.Column = 1;
        end

        function lbl = sideInfo_(app, row, text)
            lbl = uilabel(app.SideGrid_, 'Text', text, ...
                'FontSize', 10, 'Interpreter', 'none', 'WordWrap', 'on');
            lbl.Layout.Row = row; lbl.Layout.Column = 1;
        end
    end
end
