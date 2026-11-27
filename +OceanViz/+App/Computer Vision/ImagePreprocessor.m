classdef ImagePreprocessor < handle
    % IMAGEPREPROCESSOR  Interactive Image Preprocessing Sandbox.
    %
    % Allows composition of multiple sequential pixel transformations
    % with real-time parametric slider updates, side-by-side viewports,
    % live luma intensity histogram analysis, and pipeline JSON configuration persistence.
    %
    % Author: Antigravity AI
    % Date: 2026-06-11
    
    properties (Access = private)
        % Main window handles
        Fig_
        Sidebar_
        Panel1_
        Panel2_
        CanvasAx1_
        CanvasAx2_
        CanvasAx3_
        CanvasAx4_
        
        % Controls
        DropdownPreset_
        BtnLoadCustom_
        ListBoxTransforms_
        DropdownAdd_
        DropdownViewType_
        DropdownColormap_
        DropdownFFTView_
        BtnRemove_
        BtnMoveUp_
        BtnMoveDown_
        ChkAutoRun_
        BtnRun_
        PanelParams_
        BtnExportImage_
        
        % Data & State
        OriginalImage_       % Original raw image array
        ProcessedImage_      % Resulting image array
        TransformsList_ = {} % Cell array of structs: {Type, Params, Instance}
        ActiveStepIdx_ = -1  % Current selected step in ListBox
        AutoRun_ = true      % Auto-apply transformations on parameter change/add
        
        % Styling
        Colors_
        VizCore_
    end
    
    properties (Access = private, Constant)
        COLOR_BG        = [0.11, 0.11, 0.13]  % gunmetal background
        COLOR_PANEL     = [0.17, 0.19, 0.23]  % sidebar panel
        COLOR_CANVAS    = [0.10, 0.12, 0.16]  % canvas background
        COLOR_BTN       = [0.22, 0.25, 0.32]  % button background
        COLOR_GRID      = [0.25, 0.27, 0.33]  % grid lines
        COLOR_TEXT      = [0.90, 0.90, 0.90]  % primary text
        COLOR_ACCENT    = [0.20, 0.60, 1.00]  % active accent
        COLOR_SECONDARY = [1.00, 0.40, 0.40]  % inactive accent
        COLOR_MUTED     = [0.55, 0.58, 0.63]
        
        FIG_WIDTH       = 1100
        FIG_HEIGHT      = 720
        SIDEBAR_W       = 320
    end
    
    methods
        function obj = ImagePreprocessor()
            % Construct the app and prepare styling
            try
                obj.VizCore_ = OceanViz.Core('Theme', 'Dark');
                obj.Colors_ = obj.VizCore_.Colors;
            catch
                % Fallback theme
                obj.Colors_.Background = obj.COLOR_BG;
                obj.Colors_.Text = obj.COLOR_TEXT;
                obj.Colors_.Primary = obj.COLOR_ACCENT;
                obj.Colors_.Secondary = obj.COLOR_SECONDARY;
            end
            
            % Setup UI
            obj.buildUI_();
            
            % Load default preset image
            obj.loadPresetImage_('ocean_img.jpg');
        end
        
        function show(obj)
            % SHOW Make the application window visible.
            if ~isempty(obj.Fig_) && isvalid(obj.Fig_)
                figure(obj.Fig_);
            end
        end
        
        function delete(obj)
            % Clean teardown of figure window
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
                'Name', 'Image Preprocessing Sandbox — Ocean Library', ...
                'Position', [x0, y0, obj.FIG_WIDTH, obj.FIG_HEIGHT], ...
                'Color', obj.COLOR_BG);
            
            % --- Sidebar control panel ---
            obj.Sidebar_ = uipanel(obj.Fig_, ...
                'Position', [10, 10, obj.SIDEBAR_W, obj.FIG_HEIGHT - 20], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none');
            
            % Title
            uilabel(obj.Sidebar_, ...
                'Text', 'Image Source', ...
                'Position', [15, obj.FIG_HEIGHT - 55, obj.SIDEBAR_W - 30, 25], ...
                'FontSize', 14, ...
                'FontWeight', 'bold', ...
                'FontColor', obj.COLOR_ACCENT);
            
            % Preset dropdown
            obj.DropdownPreset_ = uidropdown(obj.Sidebar_, ...
                'Items', {'ocean_img.jpg', 'ocean_img_gray.jpg', 'elon_musk.jpg', 'Squid-game-symbols.jpg', 'conde.jpg', 'consu.jpg', 'cónu.jpg'}, ...
                'Position', [15, obj.FIG_HEIGHT - 90, 150, 28], ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(src, ~) obj.loadPresetImage_(src.Value));
            
            % Load custom file button
            obj.BtnLoadCustom_ = uibutton(obj.Sidebar_, ...
                'Text', 'Load Custom...', ...
                'Position', [175, obj.FIG_HEIGHT - 90, 130, 28], ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~, ~) obj.loadCustomImage_());
            
            % Divider 1
            uilabel(obj.Sidebar_, ...
                'Text', '', ...
                'Position', [15, obj.FIG_HEIGHT - 105, obj.SIDEBAR_W - 30, 1], ...
                'BackgroundColor', obj.COLOR_GRID);
            
            % Pipeline section title
            uilabel(obj.Sidebar_, ...
                'Text', 'Transformation Steps', ...
                'Position', [15, obj.FIG_HEIGHT - 135, obj.SIDEBAR_W - 30, 25], ...
                'FontSize', 14, ...
                'FontWeight', 'bold', ...
                'FontColor', obj.COLOR_ACCENT);
            
            % Active Steps ListBox
            obj.ListBoxTransforms_ = uilistbox(obj.Sidebar_, ...
                'Position', [15, obj.FIG_HEIGHT - 250, obj.SIDEBAR_W - 30, 100], ...
                'BackgroundColor', obj.COLOR_CANVAS, ...
                'FontColor', obj.COLOR_TEXT, ...
                'Items', {}, ...
                'ValueChangedFcn', @(~, ~) obj.onStepSelectionChanged_());
            
            % Dropdown to Add Strategy
            addItems = {'+ Add Step...', ...
                        '--- Point Operations ---', ...
                        '  Linear Operator', '  Gamma Correction', '  Color Balance', ...
                        '  Histogram Equalization', '  Adaptive HistEq (CLAHE)', ...
                        '  Color Space Converter', '  MinMax Stretch', '  MeanMatch Stretch', ...
                        '  Lookup Table (LUT)', '  Local Laplacian', ...
                        '--- Spatial Filtering ---', ...
                        '  Spatial Filter', ...
                        '--- Image Denoising ---', ...
                        '  Gaussian Filter', '  Median Filter', '  Bilateral Filter', '  Non-Local Means', ...
                        '--- Geometric Transforms ---', ...
                        '  Decimation (Downsample)', '  Inject Zeros (Expand)', ...
                        '--- Domain Transforms ---', ...
                        '  Lifting Scheme', ...
                        '--- Reconstruction ---', ...
                        '  Synthesis Filter'};
            obj.DropdownAdd_ = uidropdown(obj.Sidebar_, ...
                'Items', addItems, ...
                'Position', [15, obj.FIG_HEIGHT - 290, 140, 28], ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(src, ~) obj.onAddDropdownSelected_(src));
            
            % Pipeline management buttons
            obj.BtnRemove_ = uibutton(obj.Sidebar_, ...
                'Text', 'Remove', ...
                'Position', [165, obj.FIG_HEIGHT - 290, 70, 28], ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~, ~) obj.removeTransformStep_());
            
            obj.BtnMoveUp_ = uibutton(obj.Sidebar_, ...
                'Text', 'U', ...
                'Position', [243, obj.FIG_HEIGHT - 290, 30, 28], ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'Tooltip', 'Move step up', ...
                'ButtonPushedFcn', @(~, ~) obj.moveTransformStep_(-1));
            
            obj.BtnMoveDown_ = uibutton(obj.Sidebar_, ...
                'Text', 'D', ...
                'Position', [278, obj.FIG_HEIGHT - 290, 30, 28], ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'Tooltip', 'Move step down', ...
                'ButtonPushedFcn', @(~, ~) obj.moveTransformStep_(1));
            
            % Auto Apply Checkbox
            obj.ChkAutoRun_ = uicheckbox(obj.Sidebar_, ...
                'Text', 'Auto Apply', ...
                'Value', obj.AutoRun_, ...
                'Position', [15, obj.FIG_HEIGHT - 325, 120, 22], ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(src, ~) obj.toggleAutoRun_(src.Value));
            
            % Manual Run Button
            obj.BtnRun_ = uibutton(obj.Sidebar_, ...
                'Text', 'Run Pipeline', ...
                'Position', [150, obj.FIG_HEIGHT - 327, 155, 28], ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'Enable', 'off', ...
                'ButtonPushedFcn', @(~, ~) obj.manualRun_());
            
            % Divider 2
            uilabel(obj.Sidebar_, ...
                'Text', '', ...
                'Position', [15, obj.FIG_HEIGHT - 340, obj.SIDEBAR_W - 30, 1], ...
                'BackgroundColor', obj.COLOR_GRID);
            
            % Dynamic parameters panel area
            obj.PanelParams_ = uipanel(obj.Sidebar_, ...
                'Title', 'Parameters', ...
                'Position', [15, 100, obj.SIDEBAR_W - 30, obj.FIG_HEIGHT - 450], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'ForegroundColor', obj.COLOR_TEXT, ...
                'BorderType', 'line');
            
            % Initial panel state
            uilabel(obj.PanelParams_, ...
                'Text', 'No step selected', ...
                'Position', [15, 100, 250, 25], ...
                'FontColor', obj.COLOR_MUTED, ...
                'HorizontalAlignment', 'center');
            
            % Export & Persistence buttons (Phase 5)
            obj.BtnExportImage_ = uibutton(obj.Sidebar_, ...
                'Text', 'Export Processed Image', ...
                'Position', [15, 55, obj.SIDEBAR_W - 30, 28], ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'FontWeight', 'bold', ...
                'ButtonPushedFcn', @(~, ~) obj.exportProcessedImage_());
            
            uibutton(obj.Sidebar_, ...
                'Text', 'Export Pipeline', ...
                'Position', [15, 15, 135, 28], ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~, ~) obj.exportPipelineJSON_());
            
            uibutton(obj.Sidebar_, ...
                'Text', 'Import Pipeline', ...
                'Position', [160, 15, 145, 28], ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~, ~) obj.importPipelineJSON_());
            
            % --- Viewport Axes (Side-by-Side & Histogram) ---
            canvasW = (obj.FIG_WIDTH - obj.SIDEBAR_W - 50) / 2;
            
            % Original Viewport panel & axes
            obj.Panel1_ = uipanel(obj.Fig_, ...
                'Title', 'Original Image', ...
                'Position', [obj.SIDEBAR_W + 20, 280, canvasW, 420], ...
                'BackgroundColor', obj.COLOR_CANVAS, ...
                'ForegroundColor', obj.COLOR_TEXT);
            obj.CanvasAx1_ = uiaxes(obj.Panel1_, ...
                'Position', [5, 5, canvasW - 10, 385], ...
                'Color', obj.COLOR_CANVAS, ...
                'XColor', 'none', ...
                'YColor', 'none');
            disableDefaultInteractivity(obj.CanvasAx1_);
            
            % Processed Viewport panel & axes
            obj.Panel2_ = uipanel(obj.Fig_, ...
                'Title', 'Processed Image', ...
                'Position', [obj.SIDEBAR_W + 30 + canvasW, 280, canvasW, 420], ...
                'BackgroundColor', obj.COLOR_CANVAS, ...
                'ForegroundColor', obj.COLOR_TEXT);
                
            % View Type Dropdown
            obj.DropdownViewType_ = uidropdown(obj.Panel2_, ...
                'Position', [canvasW - 170, 395, 160, 22], ...
                'Items', {'Image View', '3D Surface View', 'Channel Split View'}, ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(~, ~) obj.renderViewports_());
                
            obj.CanvasAx2_ = uiaxes(obj.Panel2_, ...
                'Position', [5, 5, canvasW - 10, 385], ...
                'Color', obj.COLOR_CANVAS, ...
                'XColor', 'none', ...
                'YColor', 'none');
            disableDefaultInteractivity(obj.CanvasAx2_);
            
            % Histogram & Frequency Analysis panel
            panel3 = uipanel(obj.Fig_, ...
                'Title', 'Frequency & Intensity Analysis', ...
                'Position', [obj.SIDEBAR_W + 20, 20, obj.FIG_WIDTH - obj.SIDEBAR_W - 30, 240], ...
                'BackgroundColor', obj.COLOR_CANVAS, ...
                'ForegroundColor', obj.COLOR_TEXT);
            
            % Left axes for Histogram
            obj.CanvasAx3_ = uiaxes(panel3, ...
                'Position', [15, 10, 350, 200], ...
                'Color', obj.COLOR_CANVAS, ...
                'XColor', obj.COLOR_MUTED, ...
                'YColor', obj.COLOR_MUTED);
            disableDefaultInteractivity(obj.CanvasAx3_);
            hold(obj.CanvasAx3_, 'on');
            
            % Right axes for FFT Spectrum
            obj.CanvasAx4_ = uiaxes(panel3, ...
                'Position', [385, 10, 350, 200], ...
                'Color', obj.COLOR_CANVAS, ...
                'XColor', 'none', ...
                'YColor', 'none');
            disableDefaultInteractivity(obj.CanvasAx4_);
            
            % View Type Dropdown for FFT
            obj.DropdownFFTView_ = uidropdown(panel3, ...
                'Position', [590, 205, 120, 22], ...
                'Items', {'2D Heatmap', '3D Surface'}, ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(~, ~) obj.updateFFTSpectrum_());
                
            % Colormap Dropdown
            obj.DropdownColormap_ = uidropdown(panel3, ...
                'Position', [460, 205, 120, 22], ...
                'Items', {'parula', 'turbo', 'hsv', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter', 'gray', 'bone', 'copper', 'pink', 'lines', 'colorcube', 'prism', 'flag', 'white'}, ...
                'Value', 'parula', ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(~, ~) obj.renderViewports_());
        end
        
        function loadPresetImage_(obj, name)
            script_dir = fileparts(mfilename('fullpath'));
            project_root = fullfile(script_dir, '..', '..', '..');
            image_path = fullfile(project_root, 'Data', 'images', name);
            
            if exist(image_path, 'file')
                obj.loadImageFromFile_(image_path);
            else
                uialert(obj.Fig_, sprintf('Preset file not found:\n%s', image_path), 'Error');
            end
        end
        
        function loadCustomImage_(obj)
            [file, path] = uigetfile({'*.jpg;*.png;*.bmp;*.tiff', 'Image Files (*.jpg, *.png, *.bmp, *.tiff)'}, ...
                                     'Load Image');
            if isequal(file, 0) || isequal(path, 0)
                return;
            end
            obj.loadImageFromFile_(fullfile(path, file));
        end
        
        function loadImageFromFile_(obj, path)
            try
                img = imread(path);
                obj.OriginalImage_ = img;
                
                % Clear axes and render original
                cla(obj.CanvasAx1_);
                imshow(obj.OriginalImage_, 'Parent', obj.CanvasAx1_);
                
                % Update panel1 title with size info
                [h, w, c] = size(obj.OriginalImage_);
                obj.Panel1_.Title = sprintf('Original Image (%dx%dx%d)', h, w, c);
                
                if obj.AutoRun_
                    obj.applyPipeline_();
                    obj.renderViewports_();
                else
                    obj.ProcessedImage_ = [];
                    obj.renderViewports_();
                end
            catch ME
                uialert(obj.Fig_, sprintf('Failed to load image:\n%s', ME.message), 'Error');
            end
        end
        
        % --- Pipeline step manipulation callbacks ---
        function onAddDropdownSelected_(obj, src)
            val = src.Value;
            % Reset selection back to "+ Add Step..."
            src.Value = '+ Add Step...';
            
            if startsWith(val, '+') || startsWith(val, '---')
                return;
            end
            
            type = strtrim(val);
            
            % Add step default parameters and instantiate
            step = [];
            step.Type = type;
            
            switch type
                case 'Linear Operator'
                    step.Params = struct('Scale', 1.0, 'Bias', 0.0, 'Clip', true);
                    step.Instance = OceanMath.Image.Pixels.LinearOperator(...
                        'Scale', step.Params.Scale, 'Bias', step.Params.Bias, 'Clip', step.Params.Clip);
                case 'Gamma Correction'
                    step.Params = struct('Gamma', 1.0, 'C', 1.0);
                    step.Instance = OceanMath.Image.Pixels.GammaCorrection(...
                        'Gamma', step.Params.Gamma, 'C', step.Params.C);
                case 'Color Balance'
                    step.Params = struct('RedGain', 1.0, 'GreenGain', 1.0, 'BlueGain', 1.0, 'Gamma', 1.0, 'ApplyGammaFirst', false);
                    step.Instance = OceanMath.Image.Pixels.ColorBalance(...
                        'RedGain', step.Params.RedGain, 'GreenGain', step.Params.GreenGain, ...
                        'BlueGain', step.Params.BlueGain, 'Gamma', step.Params.Gamma, ...
                        'ApplyGammaFirst', step.Params.ApplyGammaFirst);
                case 'Histogram Equalization'
                    step.Params = struct('Alpha', 1.0);
                    step.Instance = OceanMath.Image.Pixels.HistogramEqualization('Alpha', step.Params.Alpha);
                case 'Adaptive HistEq (CLAHE)'
                    step.Params = struct('NumTilesRows', 8, 'NumTilesCols', 8, 'ClipLimit', 0.02);
                    step.Instance = OceanMath.Image.Pixels.AdaptiveHistogramEqualization(...
                        'NumTiles', [step.Params.NumTilesRows, step.Params.NumTilesCols], 'ClipLimit', step.Params.ClipLimit);
                case 'Color Space Converter'
                    step.Params = struct('Type', 'RGB2Gray');
                    step.Instance = OceanMath.Image.Pixels.ColorSpaceConverter(step.Params.Type);
                case 'Spatial Filter'
                    step.Params = struct('Preset', 'Average Box Blur 3x3', 'Boundary', 'replicate');
                    kernel = 1/9 * ones(3,3);
                    step.Instance = OceanMath.Image.Pixels.SpatialFilter(...
                        'Kernel', kernel, 'Boundary', step.Params.Boundary);
                case 'MinMax Stretch'
                    step.Params = struct('Min', -1.0, 'Max', -1.0);
                    step.Instance = OceanMath.Image.Pixels.MinMaxStretch('TargetRange', [step.Params.Min, step.Params.Max]);
                case 'MeanMatch Stretch'
                    step.Params = struct('TargetMean', -1.0, 'Scale', 1.0);
                    step.Instance = OceanMath.Image.Pixels.MeanMatchStretch('TargetMean', step.Params.TargetMean, 'Scale', step.Params.Scale);
                case 'Inject Zeros (Expand)'
                    step.Params = struct('Factor', 2);
                    step.Instance = OceanMath.Image.Pixels.ZeroStuffing('Factor', step.Params.Factor);
                case 'Synthesis Filter'
                    step.Params = struct('Preset', 'Haar', 'Mode', 'Diagonal High-Pass (HH)');
                    step.Instance = OceanMath.Image.Pixels.SynthesisFilter(...
                        'Preset', step.Params.Preset, 'Mode', step.Params.Mode);
                case 'Decimation (Downsample)'
                    step.Params = struct('Factor', 2, 'AntiAlias', true, 'FilterType', 'Gaussian');
                    step.Instance = OceanMath.Image.Pixels.Decimation(...
                        'Factor', step.Params.Factor, ...
                        'AntiAlias', step.Params.AntiAlias, ...
                        'FilterType', step.Params.FilterType);
                case 'Lifting Scheme'
                    step.Params = struct('WaveletType', 'CDF53', 'Levels', 1);
                    step.Instance = OceanMath.Signal.Lifting.LiftingScheme(...
                        'WaveletType', step.Params.WaveletType, 'Levels', step.Params.Levels);
                case 'Lookup Table (LUT)'
                    step.Params = struct('Preset', 'Solarize', 'ParamValue', 4.0);
                    step.Instance = obj.createLutTransform_(step.Params.Preset, step.Params.ParamValue);
                case 'Local Laplacian'
                    step.Params = struct('NumLevels', 4, 'NumBins', 8, 'Sigma', 0.1, 'Alpha', 1.0, 'Beta', 1.0);
                    step.Instance = OceanMath.Image.MultiResolution.LocalLaplacianFilter(...
                        'NumLevels', step.Params.NumLevels, ...
                        'NumBins', step.Params.NumBins, ...
                        'Sigma', step.Params.Sigma, ...
                        'Alpha', step.Params.Alpha, ...
                        'Beta', step.Params.Beta);
                case 'Gaussian Filter'
                    step.Params = struct('Sigma', 1.0);
                    step.Instance = OceanMath.Image.Filters.GaussianFilter('Sigma', step.Params.Sigma);
                case 'Median Filter'
                    step.Params = struct('WindowSize', 3);
                    step.Instance = OceanMath.Image.Filters.MedianFilter('WindowSize', step.Params.WindowSize);
                case 'Bilateral Filter'
                    step.Params = struct('SpatialSigma', 3.0, 'RangeSigma', 0.1, 'Method', 'grid');
                    step.Instance = OceanMath.Image.Filters.BilateralFilter(...
                        'SpatialSigma', step.Params.SpatialSigma, ...
                        'RangeSigma', step.Params.RangeSigma, ...
                        'Method', step.Params.Method);
                case 'Non-Local Means'
                    step.Params = struct('SearchRadius', 7, 'PatchRadius', 2, 'FilteringParameter', 0.1);
                    step.Instance = OceanMath.Image.Filters.NonLocalMeansFilter(...
                        'SearchRadius', step.Params.SearchRadius, ...
                        'PatchRadius', step.Params.PatchRadius, ...
                        'FilteringParameter', step.Params.FilteringParameter);
            end
            
            obj.TransformsList_{end+1} = step;
            obj.updateListBox_();
            
            % Select the newly added step
            obj.ListBoxTransforms_.Value = numel(obj.TransformsList_);
            obj.onStepSelectionChanged_();
            
            if obj.AutoRun_
                obj.applyPipeline_();
                obj.renderViewports_();
            end
        end
        
        function removeTransformStep_(obj)
            idx = obj.ActiveStepIdx_;
            if idx < 1 || idx > numel(obj.TransformsList_)
                return;
            end
            
            obj.TransformsList_(idx) = [];
            obj.ActiveStepIdx_ = -1;
            obj.updateListBox_();
            obj.onStepSelectionChanged_();
            
            if obj.AutoRun_
                obj.applyPipeline_();
                obj.renderViewports_();
            end
        end
        
        function moveTransformStep_(obj, direction)
            idx = obj.ActiveStepIdx_;
            if idx < 1 || idx > numel(obj.TransformsList_)
                return;
            end
            
            newIdx = idx + direction;
            if newIdx < 1 || newIdx > numel(obj.TransformsList_)
                return; % Out of bounds
            end
            
            % Swap elements
            temp = obj.TransformsList_{idx};
            obj.TransformsList_{idx} = obj.TransformsList_{newIdx};
            obj.TransformsList_{newIdx} = temp;
            
            obj.updateListBox_();
            obj.ListBoxTransforms_.Value = newIdx;
            obj.onStepSelectionChanged_();
            
            if obj.AutoRun_
                obj.applyPipeline_();
                obj.renderViewports_();
            end
        end
        
        function updateListBox_(obj)
            items = cell(1, numel(obj.TransformsList_));
            for i = 1:numel(obj.TransformsList_)
                items{i} = sprintf('%d. %s', i, obj.TransformsList_{i}.Type);
            end
            obj.ListBoxTransforms_.Items = items;
            obj.ListBoxTransforms_.ItemsData = 1:numel(obj.TransformsList_);
        end
        
        function onStepSelectionChanged_(obj)
            val = obj.ListBoxTransforms_.Value;
            if isempty(val)
                obj.ActiveStepIdx_ = -1;
            else
                obj.ActiveStepIdx_ = val;
            end
            obj.drawParameterControls_();
        end
        
        % --- Dynamic parameter panel rendering ---
        function drawParameterControls_(obj)
            % Clean parameters panel
            delete(obj.PanelParams_.Children);
            
            idx = obj.ActiveStepIdx_;
            if idx < 1 || idx > numel(obj.TransformsList_)
                uilabel(obj.PanelParams_, ...
                    'Text', 'No step selected', ...
                    'Position', [15, 100, 250, 25], ...
                    'FontColor', obj.COLOR_MUTED, ...
                    'HorizontalAlignment', 'center');
                return;
            end
            
            step = obj.TransformsList_{idx};
            type = step.Type;
            params = step.Params;
            
            % Draw sliders and dropdowns based on transform type
            switch type
                case 'Linear Operator'
                    % Scale slider
                    uilabel(obj.PanelParams_, 'Text', 'Scale (Contrast):', 'Position', [15, 230, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblScaleVal = uilabel(obj.PanelParams_, 'Text', sprintf('%.2f', params.Scale), 'Position', [220, 230, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    sldScale = uislider(obj.PanelParams_, ...
                        'Limits', [0.1, 5.0], ...
                        'Value', params.Scale, ...
                        'Position', [25, 205, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateScaleParam_(src, lblScaleVal));
                    
                    % Bias slider
                    is_uint8 = isa(obj.OriginalImage_, 'uint8');
                    biasLimits = [-1.0, 1.0];
                    if is_uint8
                        biasLimits = [-255.0, 255.0];
                    end
                    
                    uilabel(obj.PanelParams_, 'Text', 'Bias (Brightness):', 'Position', [15, 155, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblBiasVal = uilabel(obj.PanelParams_, 'Text', sprintf('%.2f', params.Bias), 'Position', [220, 155, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    sldBias = uislider(obj.PanelParams_, ...
                        'Limits', biasLimits, ...
                        'Value', params.Bias, ...
                        'Position', [25, 130, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateBiasParam_(src, lblBiasVal));
                    
                    % Clip checkbox
                    chkClip = uicheckbox(obj.PanelParams_, ...
                        'Text', 'Clip Intensities', ...
                        'Value', params.Clip, ...
                        'Position', [15, 80, 200, 22], ...
                        'FontColor', obj.COLOR_TEXT, ...
                        'ValueChangedFcn', @(src, ~) obj.updateClipParam_(src));
                    
                case 'Gamma Correction'
                    % Gamma slider
                    uilabel(obj.PanelParams_, 'Text', 'Gamma:', 'Position', [15, 230, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblGammaVal = uilabel(obj.PanelParams_, 'Text', sprintf('%.2f', params.Gamma), 'Position', [220, 230, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    sldGamma = uislider(obj.PanelParams_, ...
                        'Limits', [0.1, 5.0], ...
                        'Value', params.Gamma, ...
                        'Position', [25, 205, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateGammaParam_(src, lblGammaVal));
                    
                    % C scalar slider
                    uilabel(obj.PanelParams_, 'Text', 'Gain (C):', 'Position', [15, 155, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblCVal = uilabel(obj.PanelParams_, 'Text', sprintf('%.2f', params.C), 'Position', [220, 155, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    sldC = uislider(obj.PanelParams_, ...
                        'Limits', [0.1, 3.0], ...
                        'Value', params.C, ...
                        'Position', [25, 130, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateCParam_(src, lblCVal));
                    
                case 'Color Balance'
                    % Red Gain slider
                    uilabel(obj.PanelParams_, 'Text', 'Red Gain:', 'Position', [15, 230, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblRedVal = uilabel(obj.PanelParams_, 'Text', sprintf('%.2f', params.RedGain), 'Position', [220, 230, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    uislider(obj.PanelParams_, ...
                        'Limits', [0.0, 3.0], ...
                        'Value', params.RedGain, ...
                        'Position', [25, 212, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateRedGainParam_(src, lblRedVal));
                    
                    % Green Gain slider
                    uilabel(obj.PanelParams_, 'Text', 'Green Gain:', 'Position', [15, 175, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblGreenVal = uilabel(obj.PanelParams_, 'Text', sprintf('%.2f', params.GreenGain), 'Position', [220, 175, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    uislider(obj.PanelParams_, ...
                        'Limits', [0.0, 3.0], ...
                        'Value', params.GreenGain, ...
                        'Position', [25, 157, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateGreenGainParam_(src, lblGreenVal));
                    
                    % Blue Gain slider
                    uilabel(obj.PanelParams_, 'Text', 'Blue Gain:', 'Position', [15, 120, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblBlueVal = uilabel(obj.PanelParams_, 'Text', sprintf('%.2f', params.BlueGain), 'Position', [220, 120, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    uislider(obj.PanelParams_, ...
                        'Limits', [0.0, 3.0], ...
                        'Value', params.BlueGain, ...
                        'Position', [25, 102, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateBlueGainParam_(src, lblBlueVal));
                    
                    % Gamma value slider (first, so checkbox callback can reference it)
                    lblGammaText = uilabel(obj.PanelParams_, 'Text', 'Linear Space Gamma:', 'Position', [15, 22, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblGammaVal = uilabel(obj.PanelParams_, 'Text', sprintf('%.2f', params.Gamma), 'Position', [220, 22, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    sldGamma = uislider(obj.PanelParams_, ...
                        'Limits', [0.1, 5.0], ...
                        'Value', params.Gamma, ...
                        'Position', [25, 5, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateColorBalanceGammaParam_(src, lblGammaVal));
                    
                    % Apply Gamma First checkbox
                    chkGammaFirst = uicheckbox(obj.PanelParams_, ...
                        'Text', 'Linearize (Gamma Decode First)', ...
                        'Value', params.ApplyGammaFirst, ...
                        'Position', [15, 60, 250, 22], ...
                        'FontColor', obj.COLOR_TEXT);
                    
                    % Sync the initial enable state
                    if params.ApplyGammaFirst
                        sldGamma.Enable = 'on';
                        lblGammaText.FontColor = obj.COLOR_TEXT;
                        lblGammaVal.FontColor = obj.COLOR_ACCENT;
                    else
                        sldGamma.Enable = 'off';
                        lblGammaText.FontColor = obj.COLOR_MUTED;
                        lblGammaVal.FontColor = obj.COLOR_MUTED;
                    end
                    
                    chkGammaFirst.ValueChangedFcn = @(src, ~) obj.updateApplyGammaFirst_(src, sldGamma, lblGammaText, lblGammaVal);
                    
                case 'Histogram Equalization'
                    % Alpha blending slider
                    uilabel(obj.PanelParams_, 'Text', 'Alpha (Blending):', 'Position', [15, 230, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblAlphaVal = uilabel(obj.PanelParams_, 'Text', sprintf('%.2f', params.Alpha), 'Position', [220, 230, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    sldAlpha = uislider(obj.PanelParams_, ...
                        'Limits', [0.0, 1.0], ...
                        'Value', params.Alpha, ...
                        'Position', [25, 205, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateAlphaParam_(src, lblAlphaVal));
                    
                case 'Adaptive HistEq (CLAHE)'
                    % ClipLimit slider
                    uilabel(obj.PanelParams_, 'Text', 'Clip Limit (Contrast Limit):', 'Position', [15, 230, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblClipLimit = uilabel(obj.PanelParams_, 'Text', sprintf('%.3f', params.ClipLimit), 'Position', [220, 230, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    sldClipLimit = uislider(obj.PanelParams_, ...
                        'Limits', [0.001, 0.2], ...
                        'Value', params.ClipLimit, ...
                        'Position', [25, 205, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateClipLimitParam_(src, lblClipLimit));
                    
                    % Rows input
                    uilabel(obj.PanelParams_, 'Text', 'Num Tiles Rows:', 'Position', [15, 155, 120, 20], 'FontColor', obj.COLOR_TEXT);
                    uieditfield(obj.PanelParams_, 'numeric', ...
                        'Value', params.NumTilesRows, ...
                        'Position', [15, 120, 100, 28], ...
                        'BackgroundColor', obj.COLOR_CANVAS, ...
                        'FontColor', obj.COLOR_TEXT, ...
                        'ValueChangedFcn', @(src, ~) obj.updateNumTilesRows_(src.Value));
                    
                    % Cols input
                    uilabel(obj.PanelParams_, 'Text', 'Num Tiles Cols:', 'Position', [150, 155, 120, 20], 'FontColor', obj.COLOR_TEXT);
                    uieditfield(obj.PanelParams_, 'numeric', ...
                        'Value', params.NumTilesCols, ...
                        'Position', [150, 120, 100, 28], ...
                        'BackgroundColor', obj.COLOR_CANVAS, ...
                        'FontColor', obj.COLOR_TEXT, ...
                        'ValueChangedFcn', @(src, ~) obj.updateNumTilesCols_(src.Value));
                    
                case 'Color Space Converter'
                    uilabel(obj.PanelParams_, 'Text', 'Conversion Type:', 'Position', [15, 230, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    spaces = {'RGB2Gray', 'Gray2RGB', 'RGB2HSV', 'HSV2RGB', 'RGB2YCbCr', 'YCbCr2RGB'};
                    uidropdown(obj.PanelParams_, ...
                        'Items', spaces, ...
                        'Value', params.Type, ...
                        'Position', [15, 190, 250, 28], ...
                        'BackgroundColor', obj.COLOR_BTN, ...
                        'FontColor', obj.COLOR_TEXT, ...
                        'ValueChangedFcn', @(src, ~) obj.updateColorSpaceType_(src.Value));
                    
                case 'Spatial Filter'
                    % Preset kernel selection
                    uilabel(obj.PanelParams_, 'Text', 'Kernel Preset:', 'Position', [15, 230, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    kernels = {'Average Box Blur 3x3', 'Gaussian Blur 5x5', 'Sobel X Edge', 'Sobel Y Edge', 'Laplacian Sharpen'};
                    uidropdown(obj.PanelParams_, ...
                        'Items', kernels, ...
                        'Value', params.Preset, ...
                        'Position', [15, 195, 250, 28], ...
                        'BackgroundColor', obj.COLOR_BTN, ...
                        'FontColor', obj.COLOR_TEXT, ...
                        'ValueChangedFcn', @(src, ~) obj.updateKernelPreset_(src.Value));
                    
                    % Boundary dropdown
                    uilabel(obj.PanelParams_, 'Text', 'Boundary Handling:', 'Position', [15, 145, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    boundaries = {'replicate', 'symmetric', 'circular', '0 (Zero Padding)'};
                    uidropdown(obj.PanelParams_, ...
                        'Items', boundaries, ...
                        'Value', params.Boundary, ...
                        'Position', [15, 110, 250, 28], ...
                        'BackgroundColor', obj.COLOR_BTN, ...
                        'FontColor', obj.COLOR_TEXT, ...
                        'ValueChangedFcn', @(src, ~) obj.updateBoundaryPreset_(src.Value));
                    
                case 'MinMax Stretch'
                    uilabel(obj.PanelParams_, 'Text', 'Target Min Bound (-1 for Auto):', 'Position', [15, 230, 250, 20], 'FontColor', obj.COLOR_TEXT);
                    uieditfield(obj.PanelParams_, 'numeric', ...
                        'Value', params.Min, ...
                        'Position', [15, 195, 100, 28], ...
                        'BackgroundColor', obj.COLOR_CANVAS, ...
                        'FontColor', obj.COLOR_TEXT, ...
                        'ValueChangedFcn', @(src, ~) obj.updateMinBound_(src.Value));
                        
                    uilabel(obj.PanelParams_, 'Text', 'Target Max Bound (-1 for Auto):', 'Position', [15, 145, 250, 20], 'FontColor', obj.COLOR_TEXT);
                    uieditfield(obj.PanelParams_, 'numeric', ...
                        'Value', params.Max, ...
                        'Position', [15, 110, 100, 28], ...
                        'BackgroundColor', obj.COLOR_CANVAS, ...
                        'FontColor', obj.COLOR_TEXT, ...
                        'ValueChangedFcn', @(src, ~) obj.updateMaxBound_(src.Value));
                        
                case 'MeanMatch Stretch'
                    % Target mean slider
                    uilabel(obj.PanelParams_, 'Text', 'Target Mean (-1 for Auto):', 'Position', [15, 230, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblMeanVal = uilabel(obj.PanelParams_, 'Text', sprintf('%.1f', params.TargetMean), 'Position', [220, 230, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    
                    is_uint8 = isa(obj.OriginalImage_, 'uint8');
                    meanLimits = [-1.0, 1.0];
                    if is_uint8
                        meanLimits = [-1.0, 255.0];
                    end
                    
                    uislider(obj.PanelParams_, ...
                        'Limits', meanLimits, ...
                        'Value', params.TargetMean, ...
                        'Position', [25, 205, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateMeanParam_(src, lblMeanVal));
                    
                    % Contrast scale slider
                    uilabel(obj.PanelParams_, 'Text', 'Contrast Scale:', 'Position', [15, 155, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblMMScaleVal = uilabel(obj.PanelParams_, 'Text', sprintf('%.2f', params.Scale), 'Position', [220, 155, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    uislider(obj.PanelParams_, ...
                        'Limits', [0.1, 5.0], ...
                        'Value', params.Scale, ...
                        'Position', [25, 130, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateMeanMatchScaleParam_(src, lblMMScaleVal));
                case 'Inject Zeros (Expand)'
                    % Factor slider
                    uilabel(obj.PanelParams_, 'Text', 'Expansion Factor:', 'Position', [15, 230, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblFactorVal = uilabel(obj.PanelParams_, 'Text', sprintf('%d', params.Factor), 'Position', [220, 230, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    uislider(obj.PanelParams_, ...
                        'Limits', [2, 8], ...
                        'Value', params.Factor, ...
                        'Position', [25, 205, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateFactorParam_(src, lblFactorVal));
                case 'Synthesis Filter'
                    % Preset dropdown
                    uilabel(obj.PanelParams_, 'Text', 'Preset:', 'Position', [15, 230, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    uidropdown(obj.PanelParams_, ...
                        'Items', {'Haar', 'LeGall 5/3'}, ...
                        'Value', params.Preset, ...
                        'Position', [15, 195, 250, 28], ...
                        'BackgroundColor', obj.COLOR_BTN, ...
                        'FontColor', obj.COLOR_TEXT, ...
                        'ValueChangedFcn', @(src, ~) obj.updateSynthesisPreset_(src.Value));
                    % Mode dropdown
                    uilabel(obj.PanelParams_, 'Text', 'Filter Mode:', 'Position', [15, 145, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    uidropdown(obj.PanelParams_, ...
                        'Items', {'Low-Pass (LL)', 'Horizontal High-Pass (LH)', 'Vertical High-Pass (HL)', 'Diagonal High-Pass (HH)'}, ...
                        'Value', params.Mode, ...
                        'Position', [15, 110, 250, 28], ...
                        'BackgroundColor', obj.COLOR_BTN, ...
                        'FontColor', obj.COLOR_TEXT, ...
                        'ValueChangedFcn', @(src, ~) obj.updateSynthesisMode_(src.Value));
                case 'Decimation (Downsample)'
                    % Factor slider
                    uilabel(obj.PanelParams_, 'Text', 'Decimation Factor:', 'Position', [15, 230, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblFactorVal = uilabel(obj.PanelParams_, 'Text', sprintf('%d', params.Factor), 'Position', [220, 230, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    uislider(obj.PanelParams_, ...
                        'Limits', [2, 8], ...
                        'Value', params.Factor, ...
                        'Position', [25, 205, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateDecimFactorParam_(src, lblFactorVal));
                    
                    % Anti-Alias Checkbox
                    uicheckbox(obj.PanelParams_, ...
                        'Text', 'Anti-Aliasing Filter', ...
                        'Value', params.AntiAlias, ...
                        'Position', [15, 145, 200, 22], ...
                        'FontColor', obj.COLOR_TEXT, ...
                        'ValueChangedFcn', @(src, ~) obj.updateDecimAAParam_(src.Value));
                    
                    % Filter Type Dropdown
                    uilabel(obj.PanelParams_, 'Text', 'Filter Type:', 'Position', [15, 95, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    enableState = 'on';
                    if ~params.AntiAlias
                        enableState = 'off';
                    end
                    uidropdown(obj.PanelParams_, ...
                        'Items', {'Gaussian', 'Average'}, ...
                        'Value', params.FilterType, ...
                        'Position', [15, 60, 250, 28], ...
                        'BackgroundColor', obj.COLOR_BTN, ...
                        'FontColor', obj.COLOR_TEXT, ...
                        'Enable', enableState, ...
                        'ValueChangedFcn', @(src, ~) obj.updateDecimFilterTypeParam_(src.Value));
                case 'Lifting Scheme'
                    % Wavelet Type Dropdown
                    uilabel(obj.PanelParams_, 'Text', 'Wavelet Type:', 'Position', [15, 230, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    uidropdown(obj.PanelParams_, ...
                        'Items', {'CDF53', 'Haar'}, ...
                        'Value', params.WaveletType, ...
                        'Position', [15, 195, 250, 28], ...
                        'BackgroundColor', obj.COLOR_BTN, ...
                        'FontColor', obj.COLOR_TEXT, ...
                        'ValueChangedFcn', @(src, ~) obj.updateLiftingWavelet_(src.Value));
                        
                    % Levels Slider
                    uilabel(obj.PanelParams_, 'Text', 'Decomposition Levels:', 'Position', [15, 155, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblLevelsVal = uilabel(obj.PanelParams_, 'Text', sprintf('%d', params.Levels), 'Position', [220, 155, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    uislider(obj.PanelParams_, ...
                        'Limits', [1, 5], ...
                        'Value', params.Levels, ...
                        'Position', [25, 130, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateLiftingLevels_(src, lblLevelsVal));
                case 'Lookup Table (LUT)'
                    % Preset Dropdown (Solarize, Posterize, Invert, Threshold)
                    uilabel(obj.PanelParams_, 'Text', 'LUT Effect Preset:', 'Position', [15, 230, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    uidropdown(obj.PanelParams_, ...
                        'Items', {'Solarize', 'Posterize', 'Invert', 'Threshold'}, ...
                        'Value', params.Preset, ...
                        'Position', [15, 195, 250, 28], ...
                        'BackgroundColor', obj.COLOR_BTN, ...
                        'FontColor', obj.COLOR_TEXT, ...
                        'ValueChangedFcn', @(src, ~) obj.updateLutPreset_(src.Value));
                    
                    % Parameter value slider
                    uilabel(obj.PanelParams_, 'Text', 'Parameter Value:', 'Position', [15, 145, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblParamVal = uilabel(obj.PanelParams_, 'Text', sprintf('%.2f', params.ParamValue), 'Position', [220, 145, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    
                    % Adjust slider limits/labels dynamically based on current preset
                    limits = [0.0, 255.0];
                    if strcmp(params.Preset, 'Solarize')
                        limits = [1.0, 10.0];
                    elseif strcmp(params.Preset, 'Posterize')
                        limits = [2.0, 32.0];
                    elseif strcmp(params.Preset, 'Invert')
                        limits = [0.0, 1.0]; % dummy/unused
                    end
                    
                    sldParam = uislider(obj.PanelParams_, ...
                        'Limits', limits, ...
                        'Value', params.ParamValue, ...
                        'Position', [25, 120, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateLutParamValue_(src, lblParamVal));
                    
                    % Disable parameter slider if preset is 'Invert' (doesn't take parameters)
                    if strcmp(params.Preset, 'Invert')
                        sldParam.Enable = 'off';
                        lblParamVal.Text = 'N/A';
                    end
                case 'Local Laplacian'
                    % NumLevels slider
                    uilabel(obj.PanelParams_, 'Text', 'Pyramid Levels:', 'Position', [15, 230, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblLevelsVal = uilabel(obj.PanelParams_, 'Text', sprintf('%d', params.NumLevels), 'Position', [220, 230, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    uislider(obj.PanelParams_, ...
                        'Limits', [1, 6], ...
                        'Value', params.NumLevels, ...
                        'Position', [25, 212, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateLLFLevels_(src, lblLevelsVal));
                    
                    % NumBins slider
                    uilabel(obj.PanelParams_, 'Text', 'Intensity Bins:', 'Position', [15, 175, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblBinsVal = uilabel(obj.PanelParams_, 'Text', sprintf('%d', params.NumBins), 'Position', [220, 175, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    uislider(obj.PanelParams_, ...
                        'Limits', [4, 16], ...
                        'Value', params.NumBins, ...
                        'Position', [25, 157, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateLLFBins_(src, lblBinsVal));
                    
                    % Sigma slider
                    uilabel(obj.PanelParams_, 'Text', 'Edge Threshold (Sigma):', 'Position', [15, 120, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblSigmaVal = uilabel(obj.PanelParams_, 'Text', sprintf('%.2f', params.Sigma), 'Position', [220, 120, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    uislider(obj.PanelParams_, ...
                        'Limits', [0.01, 0.5], ...
                        'Value', params.Sigma, ...
                        'Position', [25, 102, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateLLFSigma_(src, lblSigmaVal));
                        
                    % Alpha slider
                    uilabel(obj.PanelParams_, 'Text', 'Detail Control (Alpha):', 'Position', [15, 65, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblAlphaVal = uilabel(obj.PanelParams_, 'Text', sprintf('%.2f', params.Alpha), 'Position', [220, 65, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    uislider(obj.PanelParams_, ...
                        'Limits', [0.1, 4.0], ...
                        'Value', params.Alpha, ...
                        'Position', [25, 47, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateLLFAlpha_(src, lblAlphaVal));
                        
                    % Beta slider
                    uilabel(obj.PanelParams_, 'Text', 'Edge Control (Beta):', 'Position', [15, 20, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblBetaVal = uilabel(obj.PanelParams_, 'Text', sprintf('%.2f', params.Beta), 'Position', [220, 20, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    uislider(obj.PanelParams_, ...
                        'Limits', [0.1, 4.0], ...
                        'Value', params.Beta, ...
                        'Position', [25, 2, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateLLFBeta_(src, lblBetaVal));
                case 'Gaussian Filter'
                    uilabel(obj.PanelParams_, 'Text', 'Gaussian Sigma:', 'Position', [15, 230, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblSigmaVal = uilabel(obj.PanelParams_, 'Text', sprintf('%.2f', params.Sigma), 'Position', [220, 230, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    uislider(obj.PanelParams_, ...
                        'Limits', [0.1, 5.0], ...
                        'Value', params.Sigma, ...
                        'Position', [25, 205, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateGaussianSigma_(src, lblSigmaVal));
                case 'Median Filter'
                    uilabel(obj.PanelParams_, 'Text', 'Window Size (Odd):', 'Position', [15, 230, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblSizeVal = uilabel(obj.PanelParams_, 'Text', sprintf('%d', params.WindowSize), 'Position', [220, 230, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    uislider(obj.PanelParams_, ...
                        'Limits', [3, 11], ...
                        'Value', params.WindowSize, ...
                        'Position', [25, 205, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateMedianWindowSize_(src, lblSizeVal));
                case 'Bilateral Filter'
                    uilabel(obj.PanelParams_, 'Text', 'Spatial Sigma:', 'Position', [15, 230, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblSpatialVal = uilabel(obj.PanelParams_, 'Text', sprintf('%.2f', params.SpatialSigma), 'Position', [220, 230, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    uislider(obj.PanelParams_, ...
                        'Limits', [0.5, 10.0], ...
                        'Value', params.SpatialSigma, ...
                        'Position', [25, 212, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateBilateralSpatialSigma_(src, lblSpatialVal));
                    
                    uilabel(obj.PanelParams_, 'Text', 'Range Similarity Sigma:', 'Position', [15, 175, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblRangeVal = uilabel(obj.PanelParams_, 'Text', sprintf('%.2f', params.RangeSigma), 'Position', [220, 175, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    uislider(obj.PanelParams_, ...
                        'Limits', [0.01, 0.5], ...
                        'Value', params.RangeSigma, ...
                        'Position', [25, 157, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateBilateralRangeSigma_(src, lblRangeVal));
                    
                    uilabel(obj.PanelParams_, 'Text', 'Execution Method:', 'Position', [15, 120, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    uidropdown(obj.PanelParams_, ...
                        'Items', {'grid', 'direct'}, ...
                        'Value', params.Method, ...
                        'Position', [15, 85, 250, 28], ...
                        'BackgroundColor', obj.COLOR_BTN, ...
                        'FontColor', obj.COLOR_TEXT, ...
                        'ValueChangedFcn', @(src, ~) obj.updateBilateralMethod_(src.Value));
                case 'Non-Local Means'
                    uilabel(obj.PanelParams_, 'Text', 'Filtering Parameter (h):', 'Position', [15, 230, 200, 20], 'FontColor', obj.COLOR_TEXT);
                    lblHVal = uilabel(obj.PanelParams_, 'Text', sprintf('%.3f', params.FilteringParameter), 'Position', [220, 230, 50, 20], 'FontColor', obj.COLOR_ACCENT, 'FontWeight', 'bold');
                    uislider(obj.PanelParams_, ...
                        'Limits', [0.01, 0.5], ...
                        'Value', params.FilteringParameter, ...
                        'Position', [25, 205, 230, 3], ...
                        'ValueChangedFcn', @(src, ~) obj.updateNlmFilteringParameter_(src, lblHVal));
                    
                    uilabel(obj.PanelParams_, 'Text', 'Search Radius:', 'Position', [15, 155, 120, 20], 'FontColor', obj.COLOR_TEXT);
                    uieditfield(obj.PanelParams_, 'numeric', ...
                        'Value', params.SearchRadius, ...
                        'Position', [15, 120, 100, 28], ...
                        'BackgroundColor', obj.COLOR_CANVAS, ...
                        'FontColor', obj.COLOR_TEXT, ...
                        'ValueChangedFcn', @(src, ~) obj.updateNlmSearchRadius_(src.Value));
                    
                    uilabel(obj.PanelParams_, 'Text', 'Patch Radius:', 'Position', [150, 155, 120, 20], 'FontColor', obj.COLOR_TEXT);
                    uieditfield(obj.PanelParams_, 'numeric', ...
                        'Value', params.PatchRadius, ...
                        'Position', [150, 120, 100, 28], ...
                        'BackgroundColor', obj.COLOR_CANVAS, ...
                        'FontColor', obj.COLOR_TEXT, ...
                        'ValueChangedFcn', @(src, ~) obj.updateNlmPatchRadius_(src.Value));
            end
            
            % If the step instance has a Description property, show an information box in the remaining space
            if isprop(step.Instance, 'Description')
                children = obj.PanelParams_.Children;
                if ~isempty(children)
                    bottoms = zeros(1, numel(children));
                    for c_idx = 1:numel(children)
                        bottoms(c_idx) = children(c_idx).Position(2);
                    end
                    min_y = min(bottoms);
                else
                    min_y = 270;
                end
                
                desc_height = min_y - 25;
                if desc_height > 25
                    uilabel(obj.PanelParams_, ...
                        'Text', step.Instance.Description, ...
                        'Position', [15, 10, obj.SIDEBAR_W - 60, desc_height], ...
                        'FontSize', 9.5, ...
                        'FontAngle', 'italic', ...
                        'FontColor', obj.COLOR_TEXT, ...
                        'WordWrap', 'on', ...
                        'HorizontalAlignment', 'left');
                end
            end
        end
        
        % --- Dynamic parameter updater functions ---
        function updateLutPreset_(obj, val)
            idx = obj.ActiveStepIdx_;
            obj.TransformsList_{idx}.Params.Preset = val;
            
            % Reset default parameter values for the selected preset
            if strcmp(val, 'Solarize')
                obj.TransformsList_{idx}.Params.ParamValue = 4.0;
            elseif strcmp(val, 'Posterize')
                obj.TransformsList_{idx}.Params.ParamValue = 4.0; % 4 levels
            elseif strcmp(val, 'Threshold')
                obj.TransformsList_{idx}.Params.ParamValue = 127.0; % middle cutoff
            else
                obj.TransformsList_{idx}.Params.ParamValue = 0.0; % Invert
            end
            
            obj.reinstantiateSelectedStep_();
            obj.drawParameterControls_(); % Redraw parameters to update slider limits/state
            
            if obj.AutoRun_
                obj.applyPipeline_();
                obj.renderViewports_();
            end
        end
        
        function updateLutParamValue_(obj, src, label)
            idx = obj.ActiveStepIdx_;
            val = src.Value;
            obj.TransformsList_{idx}.Params.ParamValue = val;
            label.Text = sprintf('%.2f', val);
            
            obj.reinstantiateSelectedStep_();
        end
        
        function updateScaleParam_(obj, src, label)
            label.Text = sprintf('%.2f', src.Value);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.Scale = src.Value;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateRedGainParam_(obj, src, label)
            label.Text = sprintf('%.2f', src.Value);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.RedGain = src.Value;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateGreenGainParam_(obj, src, label)
            label.Text = sprintf('%.2f', src.Value);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.GreenGain = src.Value;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateBlueGainParam_(obj, src, label)
            label.Text = sprintf('%.2f', src.Value);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.BlueGain = src.Value;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateColorBalanceGammaParam_(obj, src, label)
            label.Text = sprintf('%.2f', src.Value);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.Gamma = src.Value;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateApplyGammaFirst_(obj, src, sldGamma, lblGammaText, lblGammaVal)
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.ApplyGammaFirst = src.Value;
            if src.Value
                sldGamma.Enable = 'on';
                lblGammaText.FontColor = obj.COLOR_TEXT;
                lblGammaVal.FontColor = obj.COLOR_ACCENT;
            else
                sldGamma.Enable = 'off';
                lblGammaText.FontColor = obj.COLOR_MUTED;
                lblGammaVal.FontColor = obj.COLOR_MUTED;
            end
            obj.reinstantiateSelectedStep_();
        end
        
        function updateBiasParam_(obj, src, label)
            label.Text = sprintf('%.2f', src.Value);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.Bias = src.Value;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateClipParam_(obj, src)
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.Clip = src.Value;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateDecimFactorParam_(obj, src, label)
            val = round(src.Value);
            src.Value = val;
            label.Text = sprintf('%d', val);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.Factor = val;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateDecimAAParam_(obj, val)
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.AntiAlias = val;
            obj.reinstantiateSelectedStep_();
            obj.drawParameterControls_();
        end
        
        function updateDecimFilterTypeParam_(obj, val)
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.FilterType = val;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateFactorParam_(obj, src, label)
            val = round(src.Value);
            src.Value = val;
            label.Text = sprintf('%d', val);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.Factor = val;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateLiftingWavelet_(obj, val)
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.WaveletType = val;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateLiftingLevels_(obj, src, label)
            val = round(src.Value);
            src.Value = val;
            label.Text = sprintf('%d', val);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.Levels = val;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateSynthesisPreset_(obj, val)
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.Preset = val;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateSynthesisMode_(obj, val)
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.Mode = val;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateGammaParam_(obj, src, label)
            label.Text = sprintf('%.2f', src.Value);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.Gamma = src.Value;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateCParam_(obj, src, label)
            label.Text = sprintf('%.2f', src.Value);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.C = src.Value;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateAlphaParam_(obj, src, label)
            label.Text = sprintf('%.2f', src.Value);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.Alpha = src.Value;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateClipLimitParam_(obj, src, label)
            label.Text = sprintf('%.3f', src.Value);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.ClipLimit = src.Value;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateNumTilesRows_(obj, val)
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.NumTilesRows = max(1, round(val));
            obj.reinstantiateSelectedStep_();
        end
        
        function updateNumTilesCols_(obj, val)
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.NumTilesCols = max(1, round(val));
            obj.reinstantiateSelectedStep_();
        end
        
        function updateColorSpaceType_(obj, type)
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.Type = type;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateKernelPreset_(obj, preset)
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.Preset = preset;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateBoundaryPreset_(obj, boundary)
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.Boundary = boundary;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateMinBound_(obj, val)
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.Min = val;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateMaxBound_(obj, val)
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.Max = val;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateMeanParam_(obj, src, label)
            label.Text = sprintf('%.1f', src.Value);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.TargetMean = src.Value;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateMeanMatchScaleParam_(obj, src, label)
            label.Text = sprintf('%.2f', src.Value);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.Scale = src.Value;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateLLFLevels_(obj, src, label)
            val = round(src.Value);
            src.Value = val;
            label.Text = sprintf('%d', val);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.NumLevels = val;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateLLFBins_(obj, src, label)
            val = round(src.Value);
            src.Value = val;
            label.Text = sprintf('%d', val);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.NumBins = val;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateLLFSigma_(obj, src, label)
            label.Text = sprintf('%.2f', src.Value);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.Sigma = src.Value;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateLLFAlpha_(obj, src, label)
            label.Text = sprintf('%.2f', src.Value);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.Alpha = src.Value;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateLLFBeta_(obj, src, label)
            label.Text = sprintf('%.2f', src.Value);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.Beta = src.Value;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateGaussianSigma_(obj, src, label)
            label.Text = sprintf('%.2f', src.Value);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.Sigma = src.Value;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateMedianWindowSize_(obj, src, label)
            val = round(src.Value);
            if mod(val, 2) == 0
                val = val + 1;
            end
            src.Value = val;
            label.Text = sprintf('%d', val);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.WindowSize = val;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateBilateralSpatialSigma_(obj, src, label)
            label.Text = sprintf('%.2f', src.Value);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.SpatialSigma = src.Value;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateBilateralRangeSigma_(obj, src, label)
            label.Text = sprintf('%.2f', src.Value);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.RangeSigma = src.Value;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateBilateralMethod_(obj, val)
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.Method = val;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateNlmFilteringParameter_(obj, src, label)
            label.Text = sprintf('%.3f', src.Value);
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.FilteringParameter = src.Value;
            obj.reinstantiateSelectedStep_();
        end
        
        function updateNlmSearchRadius_(obj, val)
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.SearchRadius = max(1, round(val));
            obj.reinstantiateSelectedStep_();
        end
        
        function updateNlmPatchRadius_(obj, val)
            obj.TransformsList_{obj.ActiveStepIdx_}.Params.PatchRadius = max(0, round(val));
            obj.reinstantiateSelectedStep_();
        end
        
        function reinstantiateSelectedStep_(obj)
            idx = obj.ActiveStepIdx_;
            step = obj.TransformsList_{idx};
            params = step.Params;
            
            switch step.Type
                case 'Linear Operator'
                    step.Instance = OceanMath.Image.Pixels.LinearOperator(...
                        'Scale', params.Scale, 'Bias', params.Bias, 'Clip', params.Clip);
                case 'Gamma Correction'
                    step.Instance = OceanMath.Image.Pixels.GammaCorrection(...
                        'Gamma', params.Gamma, 'C', params.C);
                case 'Color Balance'
                    step.Instance = OceanMath.Image.Pixels.ColorBalance(...
                        'RedGain', params.RedGain, 'GreenGain', params.GreenGain, ...
                        'BlueGain', params.BlueGain, 'Gamma', params.Gamma, ...
                        'ApplyGammaFirst', params.ApplyGammaFirst);
                case 'Histogram Equalization'
                    step.Instance = OceanMath.Image.Pixels.HistogramEqualization('Alpha', params.Alpha);
                case 'Adaptive HistEq (CLAHE)'
                    step.Instance = OceanMath.Image.Pixels.AdaptiveHistogramEqualization(...
                        'NumTiles', [params.NumTilesRows, params.NumTilesCols], 'ClipLimit', params.ClipLimit);
                case 'Color Space Converter'
                    step.Instance = OceanMath.Image.Pixels.ColorSpaceConverter(params.Type);
                case 'Spatial Filter'
                    switch params.Preset
                        case 'Average Box Blur 3x3'
                            kernel = 1/9 * ones(3,3);
                        case 'Gaussian Blur 5x5'
                            kernel = 1/256 * [1 4 6 4 1; 4 16 24 16 4; 6 24 36 24 6; 4 16 24 16 4; 1 4 6 4 1];
                        case 'Sobel X Edge'
                            kernel = [-1 0 1; -2 0 2; -1 0 1];
                        case 'Sobel Y Edge'
                            kernel = [-1 -2 -1; 0 0 0; 1 2 1];
                        case 'Laplacian Sharpen'
                            kernel = [0 1 0; 1 -4 1; 0 1 0];
                    end
                    boundary = params.Boundary;
                    if strcmp(boundary, '0 (Zero Padding)')
                        boundary = 0;
                    end
                    step.Instance = OceanMath.Image.Pixels.SpatialFilter(...
                        'Kernel', kernel, 'Boundary', boundary);
                case 'MinMax Stretch'
                    step.Instance = OceanMath.Image.Pixels.MinMaxStretch('TargetRange', [params.Min, params.Max]);
                case 'MeanMatch Stretch'
                    step.Instance = OceanMath.Image.Pixels.MeanMatchStretch(...
                        'TargetMean', params.TargetMean, 'Scale', params.Scale);
                case 'Inject Zeros (Expand)'
                    step.Instance = OceanMath.Image.Pixels.ZeroStuffing('Factor', params.Factor);
                case 'Synthesis Filter'
                    step.Instance = OceanMath.Image.Pixels.SynthesisFilter(...
                        'Preset', params.Preset, 'Mode', params.Mode);
                case 'Decimation (Downsample)'
                    step.Instance = OceanMath.Image.Pixels.Decimation(...
                        'Factor', params.Factor, ...
                        'AntiAlias', params.AntiAlias, ...
                        'FilterType', params.FilterType);
                case 'Lookup Table (LUT)'
                    step.Instance = obj.createLutTransform_(params.Preset, params.ParamValue);
                case 'Lifting Scheme'
                    step.Instance = OceanMath.Signal.Lifting.LiftingScheme(...
                        'WaveletType', params.WaveletType, 'Levels', params.Levels);
                case 'Local Laplacian'
                    step.Instance = OceanMath.Image.MultiResolution.LocalLaplacianFilter(...
                        'NumLevels', params.NumLevels, ...
                        'NumBins', params.NumBins, ...
                        'Sigma', params.Sigma, ...
                        'Alpha', params.Alpha, ...
                        'Beta', params.Beta);
                case 'Gaussian Filter'
                    step.Instance = OceanMath.Image.Filters.GaussianFilter('Sigma', params.Sigma);
                case 'Median Filter'
                    step.Instance = OceanMath.Image.Filters.MedianFilter('WindowSize', params.WindowSize);
                case 'Bilateral Filter'
                    step.Instance = OceanMath.Image.Filters.BilateralFilter(...
                        'SpatialSigma', params.SpatialSigma, ...
                        'RangeSigma', params.RangeSigma, ...
                        'Method', params.Method);
                case 'Non-Local Means'
                    step.Instance = OceanMath.Image.Filters.NonLocalMeansFilter(...
                        'SearchRadius', params.SearchRadius, ...
                        'PatchRadius', params.PatchRadius, ...
                        'FilteringParameter', params.FilteringParameter);
            end
            
            obj.TransformsList_{idx} = step;
            if obj.AutoRun_
                obj.applyPipeline_();
                obj.renderViewports_();
            end
        end
        
        function lutObj = createLutTransform_(obj, preset, paramVal)
            x = (0:255)';
            switch preset
                case 'Solarize'
                    % Parabolic mapping: I_out = paramVal * x * (255 - x) / 255
                    % With paramVal typically around 4.0 for standard solarization
                    lut = paramVal * x .* (255 - x) / 255;
                    lut = max(0, min(255, lut));
                case 'Posterize'
                    % Quantization staircase: levels = paramVal (e.g. 4)
                    levels = max(2, round(paramVal));
                    lut = 255 * (floor((x/255) * levels) / (levels - 1));
                    lut = max(0, min(255, lut));
                case 'Invert'
                    lut = 255 - x;
                case 'Threshold'
                    lut = zeros(256, 1);
                    lut(x >= paramVal) = 255;
                otherwise
                    lut = x;
            end
            lutObj = OceanMath.Image.Pixels.LookupTableTransform(lut);
        end
        
        % --- Computational Execution ---
        function applyPipeline_(obj)
            if isempty(obj.OriginalImage_)
                return;
            end
            
            % Rebuild and apply the sequential pipeline
            try
                if isempty(obj.TransformsList_)
                    obj.ProcessedImage_ = [];
                else
                    img = obj.OriginalImage_;
                    for i = 1:numel(obj.TransformsList_)
                        inst = obj.TransformsList_{i}.Instance;
                        if isa(inst, 'OceanMath.Image.Pixels.IPixelTransform') || isa(inst, 'OceanMath.Image.MultiResolution.LocalLaplacianFilter')
                            img = inst.apply(img);
                        elseif isa(inst, 'OceanMath.Signal.Lifting.ILiftingTransform')
                            img = inst.forward(img);
                        else
                            error('Unknown transform type in pipeline.');
                        end
                    end
                    obj.ProcessedImage_ = img;
                end
            catch ME
                uialert(obj.Fig_, sprintf('Type Mismatch or Pipeline Error:\n%s', ME.message), 'Pipeline Error');
                obj.ProcessedImage_ = [];
            end
        end
        
        function renderViewports_(obj)
            if isempty(obj.ProcessedImage_)
                cla(obj.CanvasAx2_);
                cla(obj.CanvasAx3_);
                cla(obj.CanvasAx4_);
                colorbar(obj.CanvasAx4_, 'off');
                obj.CanvasAx4_.XColor = 'none';
                obj.CanvasAx4_.YColor = 'none';
                obj.Panel2_.Title = 'Processed Image (Empty)';
                return;
            end
            
            % Dynamic Dropdown View Type Update
            isStruct = isstruct(obj.ProcessedImage_);
            if isStruct
                if ~isequal(obj.DropdownViewType_.Items, {'Wavelet Subbands'})
                    obj.DropdownViewType_.Items = {'Wavelet Subbands'};
                    obj.DropdownViewType_.Value = 'Wavelet Subbands';
                end
            else
                % If it was previously Wavelet, reset to Image View
                currentItems = obj.DropdownViewType_.Items;
                if numel(currentItems) == 1 && strcmp(currentItems{1}, 'Wavelet Subbands')
                    obj.DropdownViewType_.Items = {'Image View', '3D Surface View', 'Channel Split View'};
                    obj.DropdownViewType_.Value = 'Image View';
                end
            end
            
            % Clear and reset the processed image viewport properties
            cla(obj.CanvasAx2_);
            view(obj.CanvasAx2_, 2); % Reset to flat 2D
            disableDefaultInteractivity(obj.CanvasAx2_);
            rotate3d(obj.CanvasAx2_, 'off');
            obj.CanvasAx2_.XColor = 'none';
            obj.CanvasAx2_.YColor = 'none';
            obj.CanvasAx2_.ZColor = 'none';
            
            viewType = obj.DropdownViewType_.Value;
            
            if strcmp(viewType, 'Wavelet Subbands')
                obj.renderWaveletView_();
                obj.Panel2_.Title = 'Processed Image (Wavelet Domain)';
            elseif strcmp(viewType, '3D Surface View')
                obj.renderSurfaceView_();
                [h, w, c] = size(obj.ProcessedImage_);
                obj.Panel2_.Title = sprintf('Processed Surface (%dx%dx%d)', h, w, c);
            elseif strcmp(viewType, 'Channel Split View')
                obj.renderChannelSplitView_();
                [h, w, c] = size(obj.ProcessedImage_);
                obj.Panel2_.Title = sprintf('Processed Channels (%dx%dx%d)', h, w, c);
            else
                imshow(obj.ProcessedImage_, 'Parent', obj.CanvasAx2_);
                [h, w, c] = size(obj.ProcessedImage_);
                obj.Panel2_.Title = sprintf('Processed Image (%dx%dx%d)', h, w, c);
            end
            
            if isStruct
                % Clear FFT and Histogram
                cla(obj.CanvasAx3_);
                cla(obj.CanvasAx4_);
                colorbar(obj.CanvasAx4_, 'off');
            else
                % Update live overlay histograms (Phase 4)
                obj.updateHistogram_();
                
                % Update live FFT magnitude spectrum
                obj.updateFFTSpectrum_();
            end
        end
        
        function renderWaveletView_(obj)
            coeffs = obj.ProcessedImage_;
            if isfield(coeffs, 'Levels')
                lvl = coeffs.Levels{1};
                
                % High-pass details from the first level
                HL = mat2gray(lvl.HL);
                LH = mat2gray(lvl.LH);
                HH = mat2gray(lvl.HH);
                
                % Low-pass approximation is at the root
                LL = mat2gray(coeffs.LL);
                
                % If multi-level, LL will be smaller. Resize to match Level 1 details for the 2x2 grid preview.
                [h, w, ~] = size(HL);
                if size(LL, 1) ~= h || size(LL, 2) ~= w
                    LL = imresize(LL, [h, w], 'nearest');
                end
                
                top = [LL, HL];
                bot = [LH, HH];
                gridImg = [top; bot];
                
                imshow(gridImg, 'Parent', obj.CanvasAx2_);
            end
        end
        
        function renderSurfaceView_(obj)
            I_double = double(obj.ProcessedImage_);
            if size(I_double, 3) == 3
                Y = 0.2989 * I_double(:,:,1) + 0.587 * I_double(:,:,2) + 0.114 * I_double(:,:,3);
            else
                Y = I_double;
            end
            
            surf(obj.CanvasAx2_, Y, 'EdgeColor', 'none');
            view(obj.CanvasAx2_, -45, 45);
            colormap(obj.CanvasAx2_, obj.DropdownColormap_.Value);
            
            % Enable Interactivity (Rotation)
            enableDefaultInteractivity(obj.CanvasAx2_);
            rotate3d(obj.CanvasAx2_, 'on');
            
            % Axis styling
            obj.CanvasAx2_.XColor = obj.COLOR_MUTED;
            obj.CanvasAx2_.YColor = obj.COLOR_MUTED;
            obj.CanvasAx2_.ZColor = obj.COLOR_MUTED;
            xlabel(obj.CanvasAx2_, 'X (Pixels)', 'Color', obj.COLOR_TEXT);
            ylabel(obj.CanvasAx2_, 'Y (Pixels)', 'Color', obj.COLOR_TEXT);
            zlabel(obj.CanvasAx2_, 'Luma Intensity', 'Color', obj.COLOR_TEXT);
        end
        
        function renderChannelSplitView_(obj)
            if size(obj.ProcessedImage_, 3) == 3
                R = obj.ProcessedImage_(:,:,1);
                G = obj.ProcessedImage_(:,:,2);
                B = obj.ProcessedImage_(:,:,3);
                
                % Create tinted channel images
                z = zeros(size(R), class(R));
                imgR = cat(3, R, z, z);
                imgG = cat(3, z, G, z);
                imgB = cat(3, z, z, B);
                
                % Concatenate horizontally
                gridImg = [imgR, imgG, imgB];
                imshow(gridImg, 'Parent', obj.CanvasAx2_);
                
                % Add professional annotations
                [h, w] = size(R);
                text(obj.CanvasAx2_, w/2, h - 20, 'Red Channel', 'Color', 'white', ...
                    'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
                    'BackgroundColor', [0 0 0 0.5], 'Margin', 5);
                text(obj.CanvasAx2_, w + w/2, h - 20, 'Green Channel', 'Color', 'white', ...
                    'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
                    'BackgroundColor', [0 0 0 0.5], 'Margin', 5);
                text(obj.CanvasAx2_, 2*w + w/2, h - 20, 'Blue Channel', 'Color', 'white', ...
                    'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
                    'BackgroundColor', [0 0 0 0.5], 'Margin', 5);
            else
                imshow(obj.ProcessedImage_, 'Parent', obj.CanvasAx2_);
                [h, w] = size(obj.ProcessedImage_);
                text(obj.CanvasAx2_, w/2, h - 20, 'Grayscale Channel', 'Color', 'white', ...
                    'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
                    'BackgroundColor', [0 0 0 0.5], 'Margin', 5);
            end
        end
        
        function updateFFTSpectrum_(obj)
            if isempty(obj.ProcessedImage_)
                return;
            end
            
            % Clear axes and turn off any existing colorbar to redraw fresh
            cla(obj.CanvasAx4_);
            colorbar(obj.CanvasAx4_, 'off');
            
            % Convert to grayscale double for FFT
            I_double = double(obj.ProcessedImage_);
            if size(I_double, 3) == 3
                Y = 0.298936 * I_double(:,:,1) + 0.587043 * I_double(:,:,2) + 0.114021 * I_double(:,:,3);
            else
                Y = I_double;
            end
            
            % Downsample for 3D FFT rendering performance if image is huge
            [h, w] = size(Y);
            if h > 256 || w > 256
                scale = 256 / max(h, w);
                Y = imresize(Y, scale);
            end
            
            % Compute 2D FFT and shift DC to center
            Y_fft = fftshift(fft2(Y));
            Y_mag = log(1.0 + abs(Y_fft));
            
            viewType = obj.DropdownFFTView_.Value;
            
            if strcmp(viewType, '3D Surface')
                % Create coordinate grid from -0.5 to 0.5
                [h2, w2] = size(Y_mag);
                [U, V] = meshgrid(linspace(-0.5, 0.5, w2), linspace(-0.5, 0.5, h2));
                
                % Render spectrum as a 3D surface
                surf(obj.CanvasAx4_, U, V, Y_mag, 'EdgeColor', 'none');
                view(obj.CanvasAx4_, -45, 45);
                colormap(obj.CanvasAx4_, obj.DropdownColormap_.Value);
                
                % Enable Interactivity (Rotation)
                enableDefaultInteractivity(obj.CanvasAx4_);
                rotate3d(obj.CanvasAx4_, 'on');
                
                % Professional Axis Styling
                obj.CanvasAx4_.XColor = obj.COLOR_MUTED;
                obj.CanvasAx4_.YColor = obj.COLOR_MUTED;
                obj.CanvasAx4_.ZColor = obj.COLOR_MUTED;
                obj.CanvasAx4_.XTick = [-0.5, -0.25, 0, 0.25, 0.5];
                obj.CanvasAx4_.YTick = [-0.5, -0.25, 0, 0.25, 0.5];
                
                xlabel(obj.CanvasAx4_, 'u (cycles/pixel)', 'Color', obj.COLOR_TEXT);
                ylabel(obj.CanvasAx4_, 'v (cycles/pixel)', 'Color', obj.COLOR_TEXT);
                zlabel(obj.CanvasAx4_, 'Log Magnitude', 'Color', obj.COLOR_TEXT);
                title(obj.CanvasAx4_, '3D FFT Magnitude Spectrum', 'Color', obj.COLOR_TEXT);
            else
                % Render spectrum as a flat 2D Heatmap
                imagesc(obj.CanvasAx4_, [-0.5, 0.5], [-0.5, 0.5], Y_mag);
                view(obj.CanvasAx4_, 2);
                colormap(obj.CanvasAx4_, obj.DropdownColormap_.Value);
                
                % Disable Interactivity
                disableDefaultInteractivity(obj.CanvasAx4_);
                rotate3d(obj.CanvasAx4_, 'off');
                
                % Professional Axis Styling
                obj.CanvasAx4_.XColor = obj.COLOR_MUTED;
                obj.CanvasAx4_.YColor = obj.COLOR_MUTED;
                obj.CanvasAx4_.ZColor = 'none';
                obj.CanvasAx4_.XTick = [-0.5, -0.25, 0, 0.25, 0.5];
                obj.CanvasAx4_.YTick = [-0.5, -0.25, 0, 0.25, 0.5];
                
                xlabel(obj.CanvasAx4_, 'u (cycles/pixel)', 'Color', obj.COLOR_TEXT);
                ylabel(obj.CanvasAx4_, 'v (cycles/pixel)', 'Color', obj.COLOR_TEXT);
                title(obj.CanvasAx4_, '2D FFT Magnitude Spectrum', 'Color', obj.COLOR_TEXT);
            end
            
            % Add scientific colorbar indicating log magnitude scaling
            cb = colorbar(obj.CanvasAx4_);
            cb.Color = obj.COLOR_MUTED;
            cb.Label.String = 'log(1 + |F(u,v)|)';
            cb.Label.Color = obj.COLOR_TEXT;
        end
        
        % --- Histogram Computation & Rendering (Phase 4) ---
        function updateHistogram_(obj)
            if isempty(obj.OriginalImage_) || isempty(obj.ProcessedImage_)
                return;
            end
            
            % Clear axes
            cla(obj.CanvasAx3_);
            
            % Compute luma distributions
            h_orig = obj.computeLumaHist_(obj.OriginalImage_);
            h_proc = obj.computeLumaHist_(obj.ProcessedImage_);
            
            % Draw overlay curves on CanvasAx3_
            hold(obj.CanvasAx3_, 'on');
            plot(obj.CanvasAx3_, 0:255, h_orig, '--', 'Color', [0.5, 0.5, 0.5], 'LineWidth', 1.2);
            plot(obj.CanvasAx3_, 0:255, h_proc, '-', 'Color', obj.COLOR_ACCENT, 'LineWidth', 1.8);
            hold(obj.CanvasAx3_, 'off');
            
            % Set y-limit based on maximum count excluding bin 0 (first element) if it is a massive outlier
            max_others = max([h_orig(2:end); h_proc(2:end)]);
            max_all = max([h_orig; h_proc]);
            if max_all > 5 * max_others && max_others > 0
                ylim(obj.CanvasAx3_, [0, max_others * 1.15]);
            else
                ylim(obj.CanvasAx3_, [0, max_all * 1.05]);
            end
            
            % Axis styling
            xlim(obj.CanvasAx3_, [0, 255]);
            grid(obj.CanvasAx3_, 'on');
            legend(obj.CanvasAx3_, 'Original Luma', 'Processed Luma', ...
                'Location', 'northeast', 'TextColor', obj.COLOR_TEXT);
            xlabel(obj.CanvasAx3_, 'Intensity Value', 'Color', obj.COLOR_TEXT);
            ylabel(obj.CanvasAx3_, 'Pixel Count', 'Color', obj.COLOR_TEXT);
        end
        
        function h = computeLumaHist_(obj, I)
            is_uint8 = isa(I, 'uint8');
            I_double = double(I);
            
            if size(I_double, 3) == 3
                Y = 0.299 * I_double(:,:,1) + 0.587 * I_double(:,:,2) + 0.114 * I_double(:,:,3);
            else
                Y = I_double;
            end
            
            if ~is_uint8 && max(I_double(:)) <= 1.0
                Y = Y * 255.0;
            end
            
            Y_rounded = round(Y);
            Y_rounded = max(0, min(255, Y_rounded));
            
            % Optimized blitting histogram calculation (Phase 4 performance tuning)
            h = histcounts(Y_rounded(:), 0:256)';
        end
        
        % --- Exporting & Configuration Persistence (Phase 5) ---
        function exportProcessedImage_(obj)
            if isempty(obj.ProcessedImage_)
                uialert(obj.Fig_, 'No processed image to export.', 'Export Image');
                return;
            end
            
            [file, path] = uiputfile({'*.png', 'Portable Network Graphics (*.png)'; ...
                                     '*.jpg', 'JPEG Image (*.jpg)'; ...
                                     '*.bmp', 'Windows Bitmap (*.bmp)'}, ...
                                    'Export Processed Image', 'Results/processed_image.png');
            if isequal(file, 0) || isequal(path, 0)
                return;
            end
            
            % Export
            try
                imwrite(obj.ProcessedImage_, fullfile(path, file));
                uialert(obj.Fig_, 'Image exported successfully!', 'Success', 'Icon', 'info');
            catch ME
                uialert(obj.Fig_, sprintf('Failed to export image:\n%s', ME.message), 'Error');
            end
        end
        
        function exportPipelineJSON_(obj)
            if isempty(obj.TransformsList_)
                uialert(obj.Fig_, 'No pipeline steps to export.', 'Export Pipeline');
                return;
            end
            
            [file, path] = uiputfile('*.json', 'Export Pipeline Config', 'Results/pipeline_config.json');
            if isequal(file, 0) || isequal(path, 0)
                return;
            end
            
            % Construct serializable structure array
            saveData = struct('Type', {}, 'Params', {});
            for i = 1:numel(obj.TransformsList_)
                saveData(i).Type = obj.TransformsList_{i}.Type;
                saveData(i).Params = obj.TransformsList_{i}.Params;
            end
            
            try
                jsonText = jsonencode(saveData, 'PrettyPrint', true);
                fid = fopen(fullfile(path, file), 'w');
                if fid == -1, error('Could not open file.'); end
                fprintf(fid, '%s', jsonText);
                fclose(fid);
                uialert(obj.Fig_, 'Pipeline configuration exported successfully!', 'Success', 'Icon', 'info');
            catch ME
                uialert(obj.Fig_, sprintf('Failed to export configuration:\n%s', ME.message), 'Error');
            end
        end
        
        function importPipelineJSON_(obj)
            [file, path] = uigetfile('*.json', 'Import Pipeline Config', 'Results/');
            if isequal(file, 0) || isequal(path, 0)
                return;
            end
            
            try
                fid = fopen(fullfile(path, file), 'r');
                if fid == -1, error('Could not open file.'); end
                rawText = fread(fid, '*char')';
                fclose(fid);
                
                loadedData = jsondecode(rawText);
                if ~isstruct(loadedData)
                    error('Invalid JSON file structure. Must be a structure array.');
                end
                
                newTransforms = {};
                for i = 1:numel(loadedData)
                    step = [];
                    step.Type = loadedData(i).Type;
                    step.Params = loadedData(i).Params;
                    
                    % Reinstantiate based on Type and Params
                    switch step.Type
                        case 'Linear Operator'
                            step.Instance = OceanMath.Image.Pixels.LinearOperator(...
                                'Scale', step.Params.Scale, 'Bias', step.Params.Bias, 'Clip', step.Params.Clip);
                        case 'Gamma Correction'
                            step.Instance = OceanMath.Image.Pixels.GammaCorrection(...
                                'Gamma', step.Params.Gamma, 'C', step.Params.C);
                        case 'Color Balance'
                            step.Instance = OceanMath.Image.Pixels.ColorBalance(...
                                'RedGain', step.Params.RedGain, 'GreenGain', step.Params.GreenGain, ...
                                'BlueGain', step.Params.BlueGain, 'Gamma', step.Params.Gamma, ...
                                'ApplyGammaFirst', step.Params.ApplyGammaFirst);
                        case 'Histogram Equalization'
                            step.Instance = OceanMath.Image.Pixels.HistogramEqualization('Alpha', step.Params.Alpha);
                        case 'Adaptive HistEq (CLAHE)'
                            step.Instance = OceanMath.Image.Pixels.AdaptiveHistogramEqualization(...
                                'NumTiles', [step.Params.NumTilesRows, step.Params.NumTilesCols], 'ClipLimit', step.Params.ClipLimit);
                        case 'Color Space Converter'
                            step.Instance = OceanMath.Image.Pixels.ColorSpaceConverter(step.Params.Type);
                        case 'Spatial Filter'
                            switch step.Params.Preset
                                case 'Average Box Blur 3x3'
                                    kernel = 1/9 * ones(3,3);
                                case 'Gaussian Blur 5x5'
                                    kernel = 1/256 * [1 4 6 4 1; 4 16 24 16 4; 6 24 36 24 6; 4 16 24 16 4; 1 4 6 4 1];
                                case 'Sobel X Edge'
                                    kernel = [-1 0 1; -2 0 2; -1 0 1];
                                case 'Sobel Y Edge'
                                    kernel = [-1 -2 -1; 0 0 0; 1 2 1];
                                case 'Laplacian Sharpen'
                                    kernel = [0 1 0; 1 -4 1; 0 1 0];
                            end
                            boundary = step.Params.Boundary;
                            if strcmp(boundary, '0 (Zero Padding)')
                                boundary = 0;
                            end
                            step.Instance = OceanMath.Image.Pixels.SpatialFilter(...
                                'Kernel', kernel, 'Boundary', boundary);
                        case 'MinMax Stretch'
                            step.Instance = OceanMath.Image.Pixels.MinMaxStretch('TargetRange', [step.Params.Min, step.Params.Max]);
                        case 'MeanMatch Stretch'
                            step.Instance = OceanMath.Image.Pixels.MeanMatchStretch(...
                                'TargetMean', step.Params.TargetMean, 'Scale', step.Params.Scale);
                        case 'Inject Zeros (Expand)'
                            step.Instance = OceanMath.Image.Pixels.ZeroStuffing('Factor', step.Params.Factor);
                        case 'Synthesis Filter'
                            step.Instance = OceanMath.Image.Pixels.SynthesisFilter(...
                                'Preset', step.Params.Preset, 'Mode', step.Params.Mode);
                        case 'Decimation (Downsample)'
                            step.Instance = OceanMath.Image.Pixels.Decimation(...
                                'Factor', step.Params.Factor, ...
                                'AntiAlias', step.Params.AntiAlias, ...
                                'FilterType', step.Params.FilterType);
                        case 'Lookup Table (LUT)'
                            step.Instance = obj.createLutTransform_(step.Params.Preset, step.Params.ParamValue);
                    end
                    newTransforms{end+1} = step;
                end
                
                obj.TransformsList_ = newTransforms;
                obj.updateListBox_();
                
                % Select the first step if list is not empty
                if ~isempty(obj.TransformsList_)
                    obj.ListBoxTransforms_.Value = 1;
                    obj.ActiveStepIdx_ = 1;
                else
                    obj.ActiveStepIdx_ = -1;
                end
                
                obj.drawParameterControls_();
                obj.applyPipeline_();
                obj.renderViewports_();
                uialert(obj.Fig_, 'Pipeline configuration imported successfully!', 'Success', 'Icon', 'info');
            catch ME
                uialert(obj.Fig_, sprintf('Failed to import configuration:\n%s', ME.message), 'Error');
            end
        end
        
        function toggleAutoRun_(obj, value)
            obj.AutoRun_ = value;
            if obj.AutoRun_
                obj.BtnRun_.Enable = 'off';
                obj.applyPipeline_();
                obj.renderViewports_();
            else
                obj.BtnRun_.Enable = 'on';
            end
        end
        
        function manualRun_(obj)
            obj.applyPipeline_();
            obj.renderViewports_();
        end
    end
end
