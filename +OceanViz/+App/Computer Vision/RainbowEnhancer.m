classdef RainbowEnhancer < handle
    % RAINBOWENHANCER  Step-by-step interactive app for rainbow separation.
    %
    %   Exposes the entire mathematical pipeline of Exercise 3.29, from
    %   ingestion/linearization to mask definition, background synthesis
    %   via Laplacian interpolation, background subtraction, and final
    %   re-encoding.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-07-17
    
    properties (Access = private)
        Fig_
        Sidebar_
        
        % Dashboard Axes (2x3 Grid)
        SrcAx_        % Ax1: Input/Linearized (L_total)
        MaskAx_       % Ax2: Binary Mask (M)
        BgAx_         % Ax3: Synthesized Background (L_bg)
        RainbowAx_    % Ax4: Isolated Rainbow (L_rainbow)
        EnhancedAx_   % Ax5: Final Output (P_isolated)
        InfoPanel_    % Slot 6: Math info panel
        InfoText_     % Label inside InfoPanel_
        
        % Controls
        BtnAction_    % The Wizard Step button
        BtnReset_     % Reset pipeline button
        SliderAlpha_  % Enhancement Gain (k)
        LabelAlphaVal_% Value label for Gain
        
        % Image Data
        ImgSRGB_      % P_total (sRGB)
        ImgLin_       % L_total (linear)
        Mask_         % M (logical mask)
        ImgBg_        % L_bg (linear)
        ImgRainbow_   % L_rainbow (linear)
        ImgEnhanced_  % P_isolated (sRGB)
        
        % State
        StateStep_ = 1  % 1: Load, 2: Linearize, 3: Mask, 4: Inpaint, 5: Isolate, 6: Complete
        Roi_          % Interactive ROI handle
        Colors_
        VizCore_
    end
    
    properties (Access = private, Constant)
        FIG_WIDTH       = 1200
        FIG_HEIGHT      = 700
        SIDEBAR_W       = 280
        
        COLOR_BG        = [0.08, 0.08, 0.10]  % deep gunmetal background
        COLOR_PANEL     = [0.13, 0.15, 0.19]  % sidebar / card panel
        COLOR_CANVAS    = [0.05, 0.06, 0.08]  % black canvas background
        COLOR_BTN       = [0.18, 0.21, 0.28]  % button background
        COLOR_GRID      = [0.22, 0.24, 0.30]  % borders/lines
        COLOR_TEXT      = [0.92, 0.92, 0.94]  % primary text
        COLOR_ACCENT    = [0.00, 0.55, 1.00]  % active blue accent
        COLOR_SUCCESS   = [0.10, 0.70, 0.30]  % green success indicators
        COLOR_MUTED     = [0.50, 0.53, 0.58]
    end
    
    methods
        function obj = RainbowEnhancer(varargin)
            p = inputParser;
            addParameter(p, 'Image', []);
            parse(p, varargin{:});
            imgInput = p.Results.Image;
            
            % Setup styling
            try
                obj.VizCore_ = OceanViz.Core('Theme', 'Dark');
                obj.Colors_ = obj.VizCore_.Colors;
            catch
                obj.Colors_.Background = obj.COLOR_BG;
                obj.Colors_.Text = obj.COLOR_TEXT;
                obj.Colors_.Primary = obj.COLOR_ACCENT;
            end
            
            % Build UI elements
            obj.buildUI_();
            
            % If an image was passed, pre-load it and advance to Step 2
            if ~isempty(imgInput)
                if ischar(imgInput) || isstring(imgInput)
                    imgInput = im2double(imread(imgInput));
                end
                obj.ImgSRGB_ = double(imgInput);
                obj.ImgLin_ = max(0, obj.ImgSRGB_) .^ 2.2;
                obj.StateStep_ = 2;
            else
                obj.StateStep_ = 1;
            end
            obj.updateViews_();
        end
        
        function show(obj)
            if ~isempty(obj.Fig_) && isvalid(obj.Fig_)
                figure(obj.Fig_);
            end
        end
        
        function delete(obj)
            if ~isempty(obj.Fig_) && isvalid(obj.Fig_)
                delete(obj.Fig_);
            end
        end
    end
    
    methods (Access = private)
        function buildUI_(obj)
            scr = get(0, 'ScreenSize');
            x0 = (scr(3) - obj.FIG_WIDTH) / 2;
            y0 = (scr(4) - obj.FIG_HEIGHT) / 2;
            
            obj.Fig_ = uifigure(...
                'Name', 'Exercise 3.29: Step-by-Step Rainbow Separation Dashboard', ...
                'Position', [x0, y0, obj.FIG_WIDTH, obj.FIG_HEIGHT], ...
                'Color', obj.COLOR_BG);
            
            % --- Sidebar ---
            obj.Sidebar_ = uipanel(obj.Fig_, ...
                'Position', [10, 10, obj.SIDEBAR_W, obj.FIG_HEIGHT - 20], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none');
            
            % Sidebar Title
            uilabel(obj.Sidebar_, ...
                'Text', 'Rainbow Separator', ...
                'Position', [15, obj.FIG_HEIGHT - 55, obj.SIDEBAR_W - 30, 25], ...
                'FontSize', 15, ...
                'FontWeight', 'bold', ...
                'FontColor', obj.COLOR_ACCENT);
            
            % Step Checklist Panel
            y_start = obj.FIG_HEIGHT - 120;
            steps_desc = {
                '1. Load Image (Ingest)';
                '2. Linearize Image (Gamma)';
                '3. Draw Rainbow Mask (ROI)';
                '4. Inpaint Background (Laplace)';
                '5. Isolate Spectrum (Subtract)';
                '6. Final Enhanced Output'
            };
            for i = 1:6
                uilabel(obj.Sidebar_, ...
                    'Text', steps_desc{i}, ...
                    'Position', [20, y_start - (i-1)*30, obj.SIDEBAR_W - 40, 22], ...
                    'FontSize', 11, ...
                    'FontColor', obj.COLOR_MUTED, ...
                    'Tag', sprintf('StepLabel_%d', i));
            end
            
            % Slider for Alpha/Gain
            uilabel(obj.Sidebar_, ...
                'Text', 'Enhancement Gain (k):', ...
                'Position', [15, 230, 180, 22], ...
                'FontSize', 11, ...
                'FontColor', obj.COLOR_TEXT);
            obj.LabelAlphaVal_ = uilabel(obj.Sidebar_, ...
                'Text', '3.0', ...
                'Position', [200, 230, 50, 22], ...
                'FontSize', 11, ...
                'FontColor', obj.COLOR_ACCENT, ...
                'HorizontalAlignment', 'right');
            obj.SliderAlpha_ = uislider(obj.Sidebar_, ...
                'Limits', [1.0, 10.0], ...
                'Value', 4.0, ...
                'Position', [20, 205, obj.SIDEBAR_W - 40, 3], ...
                'ValueChangingFcn', @(src,e) obj.onAlphaChanging_(e), ...
                'ValueChangedFcn', @(src,e) obj.onAlphaChanged_(e));
            
            % Action Wizard Button
            obj.BtnAction_ = uibutton(obj.Sidebar_, ...
                'Text', '1. Load Image', ...
                'Position', [15, 120, obj.SIDEBAR_W - 30, 42], ...
                'BackgroundColor', obj.COLOR_ACCENT, ...
                'FontColor', [1, 1, 1], ...
                'FontWeight', 'bold', ...
                'ButtonPushedFcn', @(~,~) obj.executeNextStep_());
                
            % Reset Button
            obj.BtnReset_ = uibutton(obj.Sidebar_, ...
                'Text', 'Reset Pipeline', ...
                'Position', [15, 60, obj.SIDEBAR_W - 30, 35], ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'FontWeight', 'bold', ...
                'ButtonPushedFcn', @(~,~) obj.resetPipeline_());
            
            % --- Dashboard 2x3 Grid Axes ---
            % Ax 1: Input / Linearized
            obj.SrcAx_ = uiaxes(obj.Fig_, 'Position', [310, 370, 260, 260], 'Color', obj.COLOR_CANVAS);
            disableDefaultInteractivity(obj.SrcAx_);
            
            % Ax 2: Binary Mask
            obj.MaskAx_ = uiaxes(obj.Fig_, 'Position', [600, 370, 260, 260], 'Color', obj.COLOR_CANVAS);
            disableDefaultInteractivity(obj.MaskAx_);
            
            % Ax 3: Background
            obj.BgAx_ = uiaxes(obj.Fig_, 'Position', [890, 370, 260, 260], 'Color', obj.COLOR_CANVAS);
            disableDefaultInteractivity(obj.BgAx_);
            
            % Ax 4: Isolated Linear Rainbow
            obj.RainbowAx_ = uiaxes(obj.Fig_, 'Position', [310, 70, 260, 260], 'Color', obj.COLOR_CANVAS);
            disableDefaultInteractivity(obj.RainbowAx_);
            
            % Ax 5: Final Enhanced Output
            obj.EnhancedAx_ = uiaxes(obj.Fig_, 'Position', [600, 70, 260, 260], 'Color', obj.COLOR_CANVAS);
            disableDefaultInteractivity(obj.EnhancedAx_);
            
            % Info Panel (Slot 6)
            obj.InfoPanel_ = uipanel(obj.Fig_, ...
                'Position', [890, 70, 260, 260], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'line', ...
                'BorderWidth', 1, ...
                'HighlightColor', obj.COLOR_GRID);
            
            uilabel(obj.InfoPanel_, ...
                'Text', 'Math & Operation Log', ...
                'Position', [10, 230, 240, 20], ...
                'FontWeight', 'bold', ...
                'FontSize', 11, ...
                'FontColor', obj.COLOR_ACCENT);
                
            obj.InfoText_ = uilabel(obj.InfoPanel_, ...
                'Text', '', ...
                'Position', [10, 10, 240, 210], ...
                'FontSize', 10, ...
                'FontColor', obj.COLOR_TEXT, ...
                'WordWrap', 'on', ...
                'VerticalAlignment', 'top');
        end
        
        function styleAx_(obj, ax, titleStr)
            ax.XColor = 'none';
            ax.YColor = 'none';
            ax.Color = obj.COLOR_CANVAS;
            ax.Title.String = titleStr;
            ax.Title.Color = obj.COLOR_TEXT;
            ax.Title.FontSize = 10;
        end
        
        function clearAx_(obj, ax, titleStr)
            cla(ax);
            ax.XColor = 'none';
            ax.YColor = 'none';
            ax.Color = obj.COLOR_CANVAS;
            text(ax, 0.5, 0.5, 'Pending...', ...
                'Color', obj.COLOR_MUTED, ...
                'HorizontalAlignment', 'center', ...
                'Units', 'normalized', ...
                'FontSize', 9);
            ax.Title.String = titleStr;
            ax.Title.Color = obj.COLOR_MUTED;
            ax.Title.FontSize = 10;
        end
        
        function executeNextStep_(obj)
            switch obj.StateStep_
                case 1
                    obj.loadImage_();
                case 2
                    obj.linearizeImage_();
                case 3
                    obj.drawMask_();
                case 4
                    obj.synthesizeBackground_();
                case 5
                    obj.isolateRainbow_();
            end
        end
        
        function loadImage_(obj)
            [file, path] = uigetfile({'*.jpg;*.png;*.bmp;*.jpeg;*.tif', 'Image Files (*.jpg, *.png, *.bmp, *.jpeg, *.tif)'});
            if isequal(file, 0) || isequal(path, 0)
                return;
            end
            
            try
                img = im2double(imread(fullfile(path, file)));
                obj.ImgSRGB_ = double(img);
                obj.ImgLin_ = [];
                obj.Mask_ = [];
                obj.ImgBg_ = [];
                obj.ImgRainbow_ = [];
                obj.ImgEnhanced_ = [];
                
                obj.StateStep_ = 2;
                obj.updateViews_();
            catch ME
                uialert(obj.Fig_, ['Failed to load image: ', ME.message], 'Error Loading Image');
            end
        end
        
        function linearizeImage_(obj)
            if isempty(obj.ImgSRGB_)
                return;
            end
            obj.ImgLin_ = max(0, obj.ImgSRGB_) .^ 2.2;
            obj.StateStep_ = 3;
            obj.updateViews_();
        end
        
        function drawMask_(obj)
            if isempty(obj.ImgLin_)
                return;
            end
            
            uialert(obj.Fig_, 'Click and drag on the top-left viewport to outline the rainbow. Double-click or press Enter to close the boundary.', 'Action Required');
            
            try
                obj.Roi_ = drawfreehand(obj.SrcAx_, 'Color', 'g', 'LineWidth', 1.5);
                
                if isempty(obj.Roi_) || ~isvalid(obj.Roi_)
                    return;
                end
                
                obj.Mask_ = createMask(obj.Roi_);
                
                % Overlay static boundary on SrcAx_ and delete interactive ROI
                pts = obj.Roi_.Position;
                delete(obj.Roi_);
                hold(obj.SrcAx_, 'on');
                plot(obj.SrcAx_, [pts(:,1); pts(1,1)], [pts(:,2); pts(1,2)], 'g-', 'LineWidth', 1.5);
                
                obj.StateStep_ = 4;
                obj.updateViews_();
            catch ME
                uialert(obj.Fig_, ['Drawing failed: ', ME.message], 'Drawing Error');
            end
        end
        
        function synthesizeBackground_(obj)
            if isempty(obj.Mask_)
                return;
            end
            
            obj.BtnAction_.Text = 'Synthesizing...';
            obj.BtnAction_.Enable = 'off';
            drawnow;
            
            try
                % Inpaint each channel via Laplacian interpolation
                obj.ImgBg_ = zeros(size(obj.ImgLin_));
                for c = 1:3
                    obj.ImgBg_(:,:,c) = regionfill(obj.ImgLin_(:,:,c), obj.Mask_);
                end
                
                obj.StateStep_ = 5;
                obj.updateViews_();
            catch ME
                obj.BtnAction_.Enable = 'on';
                obj.updateViews_();
                uialert(obj.Fig_, ['Background synthesis failed: ', ME.message], 'Processing Error');
            end
        end
        
        function isolateRainbow_(obj)
            if isempty(obj.ImgBg_)
                return;
            end
            
            % Subtract background in linear space and clamp
            obj.ImgRainbow_ = max(obj.ImgLin_ - obj.ImgBg_, 0);
            
            % Re-encode final enhanced view
            k = obj.SliderAlpha_.Value;
            I_enhanced_lin = obj.ImgBg_ + k * obj.ImgRainbow_;
            I_enhanced_lin = max(0, min(1, I_enhanced_lin));
            obj.ImgEnhanced_ = max(0, I_enhanced_lin) .^ (1/2.2);
            
            obj.StateStep_ = 6;
            obj.updateViews_();
        end
        
        function resetPipeline_(obj)
            obj.ImgSRGB_ = [];
            obj.ImgLin_ = [];
            obj.Mask_ = [];
            obj.ImgBg_ = [];
            obj.ImgRainbow_ = [];
            obj.ImgEnhanced_ = [];
            obj.StateStep_ = 1;
            obj.updateViews_();
        end
        
        function onAlphaChanging_(obj, event)
            obj.LabelAlphaVal_.Text = sprintf('%.1f', event.Value);
        end
        
        function onAlphaChanged_(obj, event)
            if obj.StateStep_ == 6
                k = event.Source.Value;
                I_enhanced_lin = obj.ImgBg_ + k * obj.ImgRainbow_;
                I_enhanced_lin = max(0, min(1, I_enhanced_lin));
                obj.ImgEnhanced_ = max(0, I_enhanced_lin) .^ (1/2.2);
                
                imshow(obj.ImgEnhanced_, 'Parent', obj.EnhancedAx_);
                axis(obj.EnhancedAx_, 'image');
                obj.styleAx_(obj.EnhancedAx_, '5. Final Output P_isolated');
            end
        end
        
        function updateViews_(obj)
            % 1. Update wizard action button
            switch obj.StateStep_
                case 1
                    obj.BtnAction_.Text = '1. Load Image';
                    obj.BtnAction_.Enable = 'on';
                    obj.BtnAction_.BackgroundColor = obj.COLOR_ACCENT;
                    obj.SliderAlpha_.Enable = 'off';
                case 2
                    obj.BtnAction_.Text = '2. Linearize Image';
                    obj.BtnAction_.Enable = 'on';
                    obj.BtnAction_.BackgroundColor = obj.COLOR_ACCENT;
                    obj.SliderAlpha_.Enable = 'off';
                case 3
                    obj.BtnAction_.Text = '3. Draw Rainbow Mask';
                    obj.BtnAction_.Enable = 'on';
                    obj.BtnAction_.BackgroundColor = obj.COLOR_ACCENT;
                    obj.SliderAlpha_.Enable = 'off';
                case 4
                    obj.BtnAction_.Text = '4. Synthesize Background';
                    obj.BtnAction_.Enable = 'on';
                    obj.BtnAction_.BackgroundColor = obj.COLOR_ACCENT;
                    obj.SliderAlpha_.Enable = 'off';
                case 5
                    obj.BtnAction_.Text = '5. Isolate & Boost';
                    obj.BtnAction_.Enable = 'on';
                    obj.BtnAction_.BackgroundColor = obj.COLOR_ACCENT;
                    obj.SliderAlpha_.Enable = 'off';
                case 6
                    obj.BtnAction_.Text = 'Pipeline Complete';
                    obj.BtnAction_.Enable = 'off';
                    obj.BtnAction_.BackgroundColor = obj.COLOR_SUCCESS;
                    obj.SliderAlpha_.Enable = 'on';
            end
            
            % 2. Update checklist panel labels
            for i = 1:6
                lbl = findobj(obj.Sidebar_, 'Tag', sprintf('StepLabel_%d', i));
                if ~isempty(lbl)
                    if i < obj.StateStep_
                        lbl.FontColor = obj.COLOR_SUCCESS;
                        lbl.FontWeight = 'normal';
                    elseif i == obj.StateStep_
                        lbl.FontColor = obj.COLOR_ACCENT;
                        lbl.FontWeight = 'bold';
                    else
                        lbl.FontColor = obj.COLOR_MUTED;
                        lbl.FontWeight = 'normal';
                    end
                end
            end
            
            % 3. Update active text description & axes render
            switch obj.StateStep_
                case 1
                    obj.InfoText_.Text = sprintf('Step 1: Data Ingestion\n\nLoad an image containing a rainbow arc (e.g., Data/rainbow_dark.jpg).\n\nEquations:\nNo mathematical conversions are applied yet. The image is displayed in non-linear sRGB space.');
                    obj.clearAx_(obj.SrcAx_, '1. Input View (sRGB)');
                    obj.clearAx_(obj.MaskAx_, '2. Binary Mask M(x,y)');
                    obj.clearAx_(obj.BgAx_, '3. Synthesized Background L_bg');
                    obj.clearAx_(obj.RainbowAx_, '4. Isolated Rainbow (sRGB)');
                    obj.clearAx_(obj.EnhancedAx_, '5. Final Output P_isolated');
                case 2
                    obj.InfoText_.Text = sprintf('Step 2: Linearization\n\nConvert sRGB pixels P_total back into linear radiometric energy space.\n\nEquations:\nL_total = P_total ^ 2.2\n\nNotice how the image appears darker. This is because display gamma is removed, leaving physical light intensities.');
                    imshow(obj.ImgSRGB_, 'Parent', obj.SrcAx_);
                    axis(obj.SrcAx_, 'image');
                    obj.styleAx_(obj.SrcAx_, '1. Input View (sRGB)');
                    obj.clearAx_(obj.MaskAx_, '2. Binary Mask M(x,y)');
                    obj.clearAx_(obj.BgAx_, '3. Synthesized Background L_bg');
                    obj.clearAx_(obj.RainbowAx_, '4. Isolated Rainbow (sRGB)');
                    obj.clearAx_(obj.EnhancedAx_, '5. Final Output P_isolated');
                case 3
                    obj.InfoText_.Text = sprintf('Step 3: Define Arc Mask\n\nOutline the rainbow arc. Click ''Draw Rainbow Mask'' and trace the rainbow region in the left viewport.\n\nEquations:\nGenerates a logical matrix M(x,y) where:\nM = 1 inside the boundary\nM = 0 outside (pure background sky)');
                    imshow(obj.ImgLin_, 'Parent', obj.SrcAx_);
                    axis(obj.SrcAx_, 'image');
                    obj.styleAx_(obj.SrcAx_, '1. Linearized Image L_total');
                    obj.clearAx_(obj.MaskAx_, '2. Binary Mask M(x,y)');
                    obj.clearAx_(obj.BgAx_, '3. Synthesized Background L_bg');
                    obj.clearAx_(obj.RainbowAx_, '4. Isolated Rainbow (sRGB)');
                    obj.clearAx_(obj.EnhancedAx_, '5. Final Output P_isolated');
                case 4
                    obj.InfoText_.Text = sprintf('Step 4: Background Synthesis\n\nEstimate the sky behind the rainbow by solving Laplace''s equation:\ngrad^2 L_bg = 0\n\nEquations:\nThe region fill algorithm interpolates boundary pixel values inward to produce a smooth, clean sky background L_bg.');
                    imshow(obj.ImgLin_, 'Parent', obj.SrcAx_);
                    axis(obj.SrcAx_, 'image');
                    obj.styleAx_(obj.SrcAx_, '1. Linearized Image L_total');
                    
                    imshow(double(obj.Mask_), 'Parent', obj.MaskAx_);
                    axis(obj.MaskAx_, 'image');
                    obj.styleAx_(obj.MaskAx_, '2. Binary Mask M(x,y)');
                    
                    obj.clearAx_(obj.BgAx_, '3. Synthesized Background L_bg');
                    obj.clearAx_(obj.RainbowAx_, '4. Isolated Rainbow (sRGB)');
                    obj.clearAx_(obj.EnhancedAx_, '5. Final Output P_isolated');
                case 5
                    obj.InfoText_.Text = sprintf('Step 5: Subtract Background\n\nExtract the pure physical rainbow energy by subtracting background light from the linear image.\n\nEquations:\nL_rainbow = max(L_total - L_bg, 0)\n\nZero clamping ensures no negative energy values exist.');
                    imshow(obj.ImgLin_, 'Parent', obj.SrcAx_);
                    axis(obj.SrcAx_, 'image');
                    obj.styleAx_(obj.SrcAx_, '1. Linearized Image L_total');
                    
                    imshow(double(obj.Mask_), 'Parent', obj.MaskAx_);
                    axis(obj.MaskAx_, 'image');
                    obj.styleAx_(obj.MaskAx_, '2. Binary Mask M(x,y)');
                    
                    imshow(max(0, obj.ImgBg_) .^ (1/2.2), 'Parent', obj.BgAx_);
                    axis(obj.BgAx_, 'image');
                    obj.styleAx_(obj.BgAx_, '3. Synthesized Background L_bg');
                    
                    obj.clearAx_(obj.RainbowAx_, '4. Isolated Rainbow (sRGB)');
                    obj.clearAx_(obj.EnhancedAx_, '5. Final Output P_isolated');
                case 6
                    obj.InfoText_.Text = sprintf('Step 6: Enhance & Gamma Compress\n\nBoost the isolated signal and map the result back to sRGB display space.\n\nEquations:\nP_isolated = (k * L_rainbow) ^ (1 / 2.2)\n\nThe final output features pure, vibrant spectral colors floating against a perfectly black canvas.');
                    imshow(obj.ImgLin_, 'Parent', obj.SrcAx_);
                    axis(obj.SrcAx_, 'image');
                    obj.styleAx_(obj.SrcAx_, '1. Linearized Image L_total');
                    
                    imshow(double(obj.Mask_), 'Parent', obj.MaskAx_);
                    axis(obj.MaskAx_, 'image');
                    obj.styleAx_(obj.MaskAx_, '2. Binary Mask M(x,y)');
                    
                    imshow(max(0, obj.ImgBg_) .^ (1/2.2), 'Parent', obj.BgAx_);
                    axis(obj.BgAx_, 'image');
                    obj.styleAx_(obj.BgAx_, '3. Synthesized Background L_bg');
                    
                    imshow(max(0, obj.ImgRainbow_) .^ (1/2.2), 'Parent', obj.RainbowAx_);
                    axis(obj.RainbowAx_, 'image');
                    obj.styleAx_(obj.RainbowAx_, '4. Isolated Rainbow (sRGB)');
                    
                    imshow(obj.ImgEnhanced_, 'Parent', obj.EnhancedAx_);
                    axis(obj.EnhancedAx_, 'image');
                    obj.styleAx_(obj.EnhancedAx_, '5. Final Output P_isolated');
            end
        end
    end
end
